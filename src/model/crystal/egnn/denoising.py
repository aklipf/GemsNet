import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.operator import gnn as ops
from src.utils.geometry import Geometry

from torch_scatter import scatter_mean

from typing import Tuple


class EGNNDenoiser(nn.Module):
    def __init__(
        self,
        features: int,
        knn: int,
        vector_fields: dict,
        layers: int,
        limit_actions,
        scale_hidden_dim: int,
        scale_reduce_rho: str,
    ):
        super().__init__()

        self.knn = knn

        self.layers = layers

        self.embedding = nn.Embedding(100, features)

        self.mpnn = nn.ModuleList([ops.MPNN(features=features) for _ in range(layers)])

        self.I = nn.Parameter(torch.eye(3), requires_grad=False)

        self.update = nn.ModuleList(
            [ops.MPNN(features=features) for _ in range(layers)]
        )

        self.actions = nn.ModuleList(
            [
                ops.Actions(
                    features=features,
                    ops_config=vector_fields,
                    hidden_dim=scale_hidden_dim,
                    n_layers=1,
                    limit_actions=limit_actions,
                    reduce_rho=scale_reduce_rho,
                )
                for _ in range(layers)
            ]
        )

        self.actions_pos = nn.ModuleList(
            [
                ops.ActionsPos(
                    features,
                    hidden_dim=scale_hidden_dim,
                    n_layers=1,
                )
                for _ in range(layers)
            ]
        )

    def actions_init(self, cell: torch.FloatTensor) -> torch.FloatTensor:
        return self.I.unsqueeze(0).repeat(cell.shape[0], 1, 1)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        struct_size: torch.LongTensor,
    ):
        cell = self.actions_init(cell)

        geometry = Geometry(cell, struct_size, x % 1, knn=self.knn)

        geometry.filter_triplets(geometry.triplets_sin_ijk.abs() > 1e-3)

        h = self.embedding(z)

        for l in self.mpnn:
            h = l(geometry, h)

        action_rho = self.actions_init(cell)

        rho_list = []
        actions_list = []
        traj_sum = 0

        for i in range(self.layers):
            actions = self.actions[i]
            actions_pos = self.actions_pos[i]
            update = self.update[i]

            h = update(geometry, h)
            edges_weights, triplets_weights = actions(geometry, h)

            rho_prime, action = actions.apply(geometry, edges_weights, triplets_weights)

            action_rho = torch.bmm(action, action_rho)
            rho_prime = torch.bmm(action_rho, cell)

            rho_list.append(rho_prime)
            actions_list.append(action_rho)

            edges_weights = actions_pos(geometry, h)
            _, x_traj, _ = actions_pos.apply(geometry, edges_weights)

            traj_sum += x_traj

            geometry.x += x_traj
            geometry.cell = rho_prime
            geometry.update_vectors()

        x_prime = geometry.x%1.0

        return x_prime, traj_sum, rho_prime
