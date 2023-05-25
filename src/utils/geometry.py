import torch
import torch.nn.functional as F

from .shape import build_shapes, assert_tensor_match, shape

from dataclasses import dataclass

import crystallographic_graph


@dataclass(init=False)
class Geometry:
    batch: torch.LongTensor
    batch_edges: torch.LongTensor
    batch_triplets: torch.LongTensor

    num_atoms: torch.LongTensor

    cell: torch.FloatTensor
    x: torch.FloatTensor

    lengths: torch.FloatTensor
    angles: torch.FloatTensor

    edges: crystallographic_graph.Edges

    edges_e_ij: torch.FloatTensor
    edges_v_ij: torch.FloatTensor
    edges_u_ij: torch.FloatTensor

    edges_r_ij: torch.FloatTensor

    triplets: crystallographic_graph.Triplets

    triplets_e_ij: torch.FloatTensor
    triplets_e_ik: torch.FloatTensor
    triplets_v_ij: torch.FloatTensor
    triplets_v_ik: torch.FloatTensor
    triplets_u_ij: torch.FloatTensor
    triplets_u_ik: torch.FloatTensor

    triplets_r_ij: torch.FloatTensor
    triplets_r_ik: torch.FloatTensor
    triplets_angle_ijk: torch.FloatTensor
    triplets_cos_ijk: torch.FloatTensor
    triplets_sin_ijk: torch.FloatTensor

    def __init__(
        self,
        cell: torch.FloatTensor,
        num_atoms: torch.LongTensor,
        x: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        knn: int = 0,
        cutoff: float = 0,
        check_tensor: bool = True,
        edges: bool = True,
        triplets: bool = True,
        symetric: bool = False,
        compute_reverse_idx: bool = False,
        edges_idx: torch.LongTensor = None,
        edges_attr: torch.LongTensor = None,
    ):
        assert knn > 0 or cutoff > 0 or (edges_idx is not None)

        if check_tensor:
            shapes = assert_tensor_match(
                (cell, shape("b", 3, 3, dtype=torch.float32)),
                (num_atoms, shape("b", dtype=torch.long)),
                (x, shape("n", 3, dtype=torch.float32)),
            )
        else:
            shapes = build_shapes(
                {
                    "b": cell.shape[0],
                    "n": x.shape[0],
                }
            )

        assert (edges_idx is None) == (edges_attr is None) or isinstance(
            edges_idx, crystallographic_graph.Edges
        )

        self.num_atoms = num_atoms

        self.cell = cell
        self.x = x

        self.edges = None
        self.batch_edges = None

        self.triplets = None
        self.batch_triplets = None

        struct_idx = torch.arange(shapes.b, device=x.device)
        self.batch = struct_idx.repeat_interleave(num_atoms)

        if edges:
            if edges_idx is None:
                self.edges = crystallographic_graph.make_graph(
                    self.cell,
                    self.x,
                    self.num_atoms,
                    knn=knn,
                    cutoff=cutoff,
                    symetric=symetric,
                    compute_reverse_idx=compute_reverse_idx,
                )

                self.batch_edges = self.batch[self.edges.src]
            elif isinstance(edges_idx, crystallographic_graph.Edges):
                self.edges = edges_idx
                self.batch_edges = self.batch[self.edges.src]
            else:
                self.edges = crystallographic_graph.Edges(
                    src=edges_idx[0], dst=edges_idx[1], cell=edges_attr
                )
                self.batch_edges = self.batch[self.edges.src]

        if triplets:
            self.triplets = crystallographic_graph.make_triplets(
                self.num_atoms, self.edges, check_tensor=check_tensor
            )
            self.batch_triplets = self.batch[self.triplets.src]

        self.update_vectors()

    def get_cell_parameters(self, cell=None):
        if cell is None:
            cell = self.cell

        lengths = cell.norm(dim=2)

        cross = torch.cross(cell[:, [1, 2, 0]], cell[:, [2, 0, 1]], dim=2)
        dot = (cell[:, [1, 2, 0]] * cell[:, [2, 0, 1]]).sum(dim=2)
        angles = torch.atan2(cross.norm(dim=2), dot)

        return lengths, angles

    def filter_edges(self, mask: torch.BoolTensor):
        assert mask.shape == self.edges.src.shape

        self.batch_edges = self.batch_edges[mask]

        self.edges.src = self.edges.src[mask]
        self.edges.dst = self.edges.dst[mask]
        self.edges.cell = self.edges.cell[mask]

        self.edges_e_ij = self.edges_e_ij[mask]
        self.edges_v_ij = self.edges_v_ij[mask]
        self.edges_r_ij = self.edges_r_ij[mask]
        self.edges_u_ij = self.edges_u_ij[mask]

    def filter_triplets(self, mask: torch.BoolTensor):
        assert mask.shape == self.triplets.src.shape

        self.batch_triplets = self.batch_triplets[mask]

        self.triplets.src = self.triplets.src[mask]
        self.triplets.dst_i = self.triplets.dst_i[mask]
        self.triplets.cell_i = self.triplets.cell_i[mask]
        self.triplets.dst_j = self.triplets.dst_j[mask]
        self.triplets.cell_j = self.triplets.cell_j[mask]

        self.triplets_e_ij = self.triplets_e_ij[mask]
        self.triplets_v_ij = self.triplets_v_ij[mask]
        self.triplets_r_ij = self.triplets_r_ij[mask]
        self.triplets_u_ij = self.triplets_u_ij[mask]

        self.triplets_e_ik = self.triplets_e_ik[mask]
        self.triplets_v_ik = self.triplets_v_ik[mask]
        self.triplets_r_ik = self.triplets_r_ik[mask]
        self.triplets_u_ik = self.triplets_u_ik[mask]

        self.triplets_cos_ijk = self.triplets_cos_ijk[mask]
        self.triplets_sin_ijk = self.triplets_sin_ijk[mask]
        self.triplets_angle_ijk = self.triplets_angle_ijk[mask]

    def update_vectors(self, cell=None, x=None):
        if cell is None:
            cell = self.cell

        if x is None:
            x = self.x

        self.lengths, self.angles = self.get_cell_parameters()

        if self.edges is not None:
            self.edges_e_ij = (
                x[self.edges.dst, :] - x[self.edges.src, :] + self.edges.cell
            )

            edges_batch = self.batch[self.edges.src]

            self.edges_v_ij = torch.bmm(
                cell[edges_batch], self.edges_e_ij.unsqueeze(2)
            ).squeeze(2)

            self.edges_r_ij = self.edges_v_ij.norm(dim=1)
            self.edges_u_ij = self.edges_v_ij / self.edges_r_ij[:, None]

            if self.edges_r_ij.isinf().any():
                raise Exception("infinite edges")
        else:
            empty_scalar = torch.empty(
                (0,), dtype=torch.float32, device=self.cell.device
            )
            empty_vector = torch.empty(
                (0, 3), dtype=torch.float32, device=self.cell.device
            )

            self.edges_e_ij = empty_vector

            self.edges_v_ij = empty_vector

            self.edges_r_ij = empty_scalar
            self.edges_u_ij = empty_vector

        if (self.triplets is not None) and (self.triplets.src.shape[0] > 0):
            self.triplets_e_ij = (
                x[self.triplets.dst_i, :]
                - x[self.triplets.src, :]
                + self.triplets.cell_i
            )
            self.triplets_e_ik = (
                x[self.triplets.dst_j, :]
                - x[self.triplets.src, :]
                + self.triplets.cell_j
            )

            triplets_batch = self.batch[self.triplets.src]

            self.triplets_v_ij = torch.bmm(
                cell[triplets_batch], self.triplets_e_ij.unsqueeze(2)
            ).squeeze(2)
            self.triplets_v_ik = torch.bmm(
                cell[triplets_batch], self.triplets_e_ik.unsqueeze(2)
            ).squeeze(2)

            self.triplets_r_ij = self.triplets_v_ij.norm(dim=1)
            self.triplets_r_ik = self.triplets_v_ik.norm(dim=1)

            self.triplets_u_ij = self.triplets_v_ij / (
                self.triplets_r_ij[:, None] + 1e-12
            )
            self.triplets_u_ik = self.triplets_v_ik / (
                self.triplets_r_ik[:, None] + 1e-12
            )

            self.triplets_cos_ijk = (self.triplets_u_ij * self.triplets_u_ik).sum(dim=1)
            self.triplets_sin_ijk = torch.cross(
                self.triplets_u_ij, self.triplets_u_ik
            ).norm(dim=1)

            self.triplets_angle_ijk = torch.atan2(
                self.triplets_sin_ijk, self.triplets_cos_ijk
            )
        else:
            empty_scalar = torch.empty(
                (0,), dtype=torch.float32, device=self.cell.device
            )
            empty_vector = torch.empty(
                (0, 3), dtype=torch.float32, device=self.cell.device
            )

            self.triplets_e_ij = empty_vector
            self.triplets_e_ik = empty_vector

            self.triplets_v_ij = empty_vector
            self.triplets_v_ik = empty_vector

            self.triplets_u_ij = empty_vector
            self.triplets_u_ik = empty_vector

            self.triplets_r_ij = empty_scalar
            self.triplets_r_ik = empty_scalar

            self.triplets_cos_ijk = empty_scalar
            self.triplets_sin_ijk = empty_scalar
            self.triplets_angle_ijk = empty_scalar


if __name__ == "__main__":
    from ase.neighborlist import neighbor_list
    from ase.spacegroup import crystal
    import torch.nn as nn
    import numpy as np

    class RandomCrystal(nn.Module):
        def __init__(
            self,
            size_pdf: torch.FloatTensor,
            std_lattice: float = 0.2,
            scale_lattice: float = 5.0,
            features: int = 128,
        ):
            super().__init__()

            self.features = features
            self.scale_lattice = scale_lattice
            self.std_lattice = std_lattice

            size_cdf = torch.cumsum(size_pdf, dim=0)
            assert size_cdf[-1] >= 1.0

            self.size_cdf = nn.Parameter(size_cdf, requires_grad=False)

        @property
        def device(self):
            return self.size_cdf.data.device

        def forward(self, batch_size: int):
            size = torch.bucketize(
                torch.rand(batch_size, device=self.device), self.size_cdf
            )
            num_atoms = size.sum()

            cells = self.scale_lattice * torch.matrix_exp(
                self.std_lattice * torch.randn(batch_size, 3, 3, device=self.device)
            )

            x = torch.rand(num_atoms, 3, device=self.device)
            z = torch.rand(num_atoms, self.features, device=self.device)

            return cells, x, z, size

    batch_size = 256

    pdf = torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    rand = RandomCrystal(pdf).to("cuda")
    cell, x, z, size = rand(batch_size=batch_size)

    batch = torch.arange(batch_size, device=size.device)
    batch = batch.repeat_interleave(size)

    geometry = Geometry(cell, size, x, knn=128, triplets=False)

    distance = []
    lengths = []
    angles = []
    for i in range(batch_size):
        mask = batch == i

        size_i = size[i].item()
        cell_i = cell[i].clone().detach().cpu().numpy()
        x_i = x[mask].clone().detach().cpu().numpy()

        cry = crystal("C" * size_i, [tuple(x) for x in x_i], cell=cell_i)

        [a, b, c, alpha, beta, gamma] = cry.get_cell_lengths_and_angles()
        lengths.append([a, b, c])
        angles.append([alpha, beta, gamma])

        dist = neighbor_list("d", cry, cutoff=5.0)
        distance.append(dist)

    distance = np.concatenate(distance)
    lengths = np.array(lengths)
    angles = np.array(angles)

    print(
        "lengths mean absolut error",
        np.max(np.abs(geometry.lengths.cpu().numpy() - lengths)),
    )
    print(
        "angles mean absolut error",
        np.max(np.abs(geometry.angles.cpu().numpy() - (angles * np.pi / 180))),
    )

    mask = geometry.edges_r_ij <= 5.0
    geom_dist = geometry.edges_r_ij[mask].detach().cpu().numpy()

    import matplotlib.pyplot as plt

    hist1, bins = np.histogram(distance, bins=32, range=(0.0, 5.0))
    hist2, _ = np.histogram(geom_dist, bins=32, range=(0.0, 5.0))
    bins = 0.5 * (bins[1:] + bins[:-1])

    plt.plot(bins, hist1)
    plt.plot(bins, hist2)
    plt.savefig("out.png")
