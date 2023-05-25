import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from typing import Tuple

from src.utils.geometry import Geometry
from src.utils.shape import build_shapes, assert_tensor_match, shape

from .operator import Operator, make_operator


class EdgeProj(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
        cutoff: float = 10,
        step: float = 0.1,
        bias: bool = False,
    ):
        super(EdgeProj, self).__init__()

        if output_dim is None:
            output_dim = features

        self.cutoff = cutoff
        self.step = step
        self.mu = nn.Parameter(
            torch.arange(0, self.cutoff, self.step, dtype=torch.float32)
        )

        layers = [
            nn.Linear(2 * features + self.mu.shape[0], hidden_dim, bias=False),
            nn.SiLU(),
        ]
        for i in range(n_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=False), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)

    def get_last_layer(self):
        last_layer = None
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                last_layer = layer
        return last_layer

    def forward(self, h, src, dst, edge_norm):
        d_ij_emb = torch.exp(
            -1 / self.step * (self.mu[None, :] - edge_norm[:, None]).pow(2)
        )
        inputs = torch.cat((h[src], h[dst], d_ij_emb), dim=-1)

        return self.mlp(inputs)


class FaceProj(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
        cutoff: float = 10,
        step: float = 0.1,
        bias: bool = False,
    ):
        super(FaceProj, self).__init__()

        if output_dim is None:
            output_dim = features

        self.cutoff = cutoff
        self.step = step
        self.mu = nn.Parameter(
            torch.arange(0, self.cutoff, self.step, dtype=torch.float32)
        )

        layers = [
            nn.Linear(3 * features + self.mu.shape[0] * 2 + 2, hidden_dim, bias=False),
            nn.SiLU(),
        ]
        for i in range(n_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=False), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)

    def get_last_layer(self):
        last_layer = None
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                last_layer = layer
        return last_layer

    def forward(self, h, triplets, norm_ij, norm_ik, cos_ijk, sin_ijk):
        d_ij_emb = torch.exp(
            -1 / self.step * (self.mu[None, :] - norm_ij[:, None]).pow(2)
        )
        d_ik_emb = torch.exp(
            -1 / self.step * (self.mu[None, :] - norm_ik[:, None]).pow(2)
        )

        inputs = torch.cat(
            (
                h[triplets.src],
                h[triplets.dst_i],
                h[triplets.dst_j],
                d_ij_emb,
                d_ik_emb,
                cos_ijk.unsqueeze(1),
                sin_ijk.unsqueeze(1),
            ),
            dim=1,
        )

        return self.mlp(inputs)


class UpdateFeatures(nn.GRU):
    def __init__(self, features: int):
        super(UpdateFeatures, self).__init__(features, features, 1, batch_first=False)

    def forward(self, h: torch.FloatTensor, mi: torch.FloatTensor):
        _, h_prime = super().forward(mi.unsqueeze(0), h.unsqueeze(0))

        return h_prime.squeeze(0)


class Actions(nn.Module):
    def __init__(
        self,
        features: int,
        ops_config: dict,
        hidden_dim: int,
        n_layers: int,
        limit_actions: float,
        reduce_rho: str,
    ):
        super(Actions, self).__init__()

        self.ops = make_operator(ops_config)

        self.limit_actions = limit_actions
        self.reduce_rho = reduce_rho

        self.interact_edges = EdgeProj(
            features,
            output_dim=self.ops.edges_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bias=True,
        )
        self.interact_triplets = FaceProj(
            features,
            output_dim=self.ops.triplets_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bias=True,
        )

        self.I = nn.Parameter(torch.eye(3), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.interact_edges.reset_parameters()
        self.interact_triplets.reset_parameters()

    def apply(
        self,
        geometry: Geometry,
        edges_weights: torch.FloatTensor,
        triplets_weights: torch.FloatTensor,
        check_tensor: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # type checking and size evaluation
        if check_tensor:
            shapes = assert_tensor_match(
                (geometry.cell, shape("b", 3, 3, dtype=torch.float32)),
                (geometry.batch, shape("n", dtype=torch.long)),
                (geometry.edges.src, shape("e", dtype=torch.long)),
                (geometry.edges.dst, shape("e", dtype=torch.long)),
                (geometry.edges.cell, shape("e", 3, dtype=torch.long)),
                (geometry.triplets.src, shape("t", dtype=torch.long)),
                (geometry.triplets.dst_i, shape("t", dtype=torch.long)),
                (geometry.triplets.cell_i, shape("t", 3, dtype=torch.long)),
                (geometry.triplets.dst_j, shape("t", dtype=torch.long)),
                (geometry.triplets.cell_j, shape("t", 3, dtype=torch.long)),
            )
        else:
            shapes = build_shapes(
                {
                    "b": geometry.cell.shape[0],
                    "n": geometry.batch.shape[0],
                    "e": geometry.edges.src.shape[0],
                    "t": geometry.triplets.src.shape[0],
                }
            )

        # calculating actions
        edges_ops, triplets_ops = self.ops.forward(geometry)

        # aggregation
        if edges_ops is not None:
            weighted_ops = (edges_ops * edges_weights[:, :, None, None]).sum(dim=1)
            actions_edges = scatter(
                weighted_ops,
                geometry.batch_edges,
                dim=0,
                dim_size=shapes.b,
                reduce=self.reduce_rho,
            )
        else:
            actions_edges = None
        if triplets_ops is not None:
            weighted_ops = (triplets_ops * triplets_weights[:, :, None, None]).sum(
                dim=1
            )
            actions_triplets = scatter(
                weighted_ops,
                geometry.batch_triplets,
                dim=0,
                dim_size=shapes.b,
                reduce=self.reduce_rho,
            )
        else:
            actions_triplets = None

        # action
        if (actions_edges is not None) and (actions_triplets is not None):
            actions_rho = actions_edges + actions_triplets
        elif actions_edges is not None:
            actions_rho = actions_edges
        elif actions_triplets is not None:
            actions_rho = actions_triplets

        if self.limit_actions != 0.0:
            actions_rho = self.limit_actions * torch.tanh(
                actions_rho / self.limit_actions
            )

        actions_rho = self.I + actions_rho

        rho_prime = torch.bmm(actions_rho, geometry.cell)

        return rho_prime, actions_rho

    def forward(
        self,
        geometry: Geometry,
        h: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if self.ops.edges_dim > 0:
            edges_weights = self.interact_edges(
                h, geometry.edges.src, geometry.edges.dst, geometry.edges_r_ij
            )
        else:
            edges_weights = None

        if self.ops.triplets_dim > 0:
            triplets_weights = self.interact_triplets(
                h,
                geometry.triplets,
                geometry.triplets_r_ij,
                geometry.triplets_r_ik,
                geometry.triplets_cos_ijk,
                geometry.triplets_sin_ijk,
            )
        else:
            triplets_weights = None

        return edges_weights, triplets_weights


class MPNN(nn.Module):
    def __init__(self, features: int):
        super(MPNN, self).__init__()

        self.message_f = EdgeProj(
            features, hidden_dim=features, n_layers=0, output_dim=features
        )
        self.update_f = UpdateFeatures(features)

        self.reset_parameters()

    def reset_parameters(self):
        self.message_f.reset_parameters()
        self.update_f.reset_parameters()

    def forward(self, geometry: Geometry, h: torch.FloatTensor):
        # message passing
        mij = self.message_f(
            h, geometry.edges.src, geometry.edges.dst, geometry.edges_r_ij
        )
        mi = scatter(mij, geometry.edges.src, dim=0, reduce="mean", dim_size=h.shape[0])
        h_prime = self.update_f(h, mi)

        return h_prime


class ActionsPos(nn.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.interact_edges = EdgeProj(
            features,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bias=True,
        )

        self.I = nn.Parameter(torch.eye(3), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.interact_edges.reset_parameters()

    def apply(
        self,
        geometry: Geometry,
        edges_weights: torch.FloatTensor,
        check_tensor: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # type checking and size evaluation
        if check_tensor:
            shapes = assert_tensor_match(
                (geometry.cell, shape("b", 3, 3, dtype=torch.float32)),
                (geometry.batch, shape("n", dtype=torch.long)),
                (geometry.edges.src, shape("e", dtype=torch.long)),
                (geometry.edges.dst, shape("e", dtype=torch.long)),
                (geometry.edges.cell, shape("e", 3, dtype=torch.long)),
            )
        else:
            shapes = build_shapes(
                {
                    "b": geometry.cell.shape[0],
                    "n": geometry.batch.shape[0],
                }
            )

        # aggregation
        weighted_ops = geometry.edges_e_ij * edges_weights
        x_traj = scatter(
            weighted_ops,
            geometry.edges.src,
            dim=0,
            dim_size=shapes.n,
            reduce="mean",
        )

        x_traj_euc = torch.bmm(
            geometry.cell[geometry.batch], x_traj.unsqueeze(2)
        ).squeeze(2)

        return (geometry.x + x_traj) % 1.0, x_traj, x_traj_euc

    def forward(
        self,
        geometry: Geometry,
        h: torch.FloatTensor,
    ) -> torch.FloatTensor:

        edges_weights = self.interact_edges(
            h, geometry.edges.src, geometry.edges.dst, geometry.edges_r_ij
        )

        return edges_weights
