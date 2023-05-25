import torch
import torch.nn as nn

import numpy as np

from torch_geometric.loader import DataLoader

import tqdm

from typing import Tuple


class LatticeScaler(nn.Module):
    def __init__(self):
        super(LatticeScaler, self).__init__()

        self.mean = nn.Parameter(
            torch.zeros(6, dtype=torch.float32), requires_grad=False
        )
        self.std = nn.Parameter(torch.ones(6, dtype=torch.float32), requires_grad=False)

    @staticmethod
    def get_lattices_parameters(
        lattices: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        lengths = lattices.norm(dim=2)

        i = torch.tensor([0, 1, 2], dtype=torch.long, device=lattices.device)
        j = torch.tensor([1, 2, 0], dtype=torch.long, device=lattices.device)
        k = torch.tensor([2, 0, 1], dtype=torch.long, device=lattices.device)

        cross = torch.cross(lattices[:, j], lattices[:, k], dim=2)
        dot = (lattices[:, j] * lattices[:, k]).sum(dim=2)

        angles = torch.atan2(cross.norm(dim=2), dot) * 180 / torch.pi

        inv_mask = (cross * lattices[:, i]).sum(dim=2) < 0
        angles[inv_mask] *= -1

        return lengths, angles

    def get_lattices(
        self, lengths: torch.FloatTensor, angles: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Converts lattice from abc, angles to matrix.
        https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
        """

        a, b, c = lengths
        alpha, beta, gamma = angles

        angles_r = torch.deg2rad(torch.tensor([alpha, beta, gamma]))
        cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
        sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

        val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
        # Sometimes rounding errors result in values slightly > 1.
        # val = max(min(val, val), -val)
        gamma_star = torch.arccos(val)

        vector_a = [a * sin_beta, 0.0, a * cos_beta]
        vector_b = [
            -b * sin_alpha * np.cos(gamma_star),
            b * sin_alpha * np.sin(gamma_star),
            b * cos_alpha,
        ]
        vector_c = [0.0, 0.0, float(c)]
        return torch.tensor([vector_a, vector_b, vector_c])

    @torch.no_grad()
    def fit(self, dataloader: DataLoader, verbose: bool = True):
        lengths, angles = [], []

        if verbose:
            iterator = tqdm.tqdm(
                dataloader, desc="calculating normalization paremeters"
            )
        else:
            iterator = dataloader

        for batch in iterator:
            current_lengths, current_angles = LatticeScaler.get_lattices_parameters(batch.cell)
            lengths.append(current_lengths)
            angles.append(current_angles)

        lengths = torch.cat(lengths, dim=0)
        angles = torch.cat(angles, dim=0)
        params = torch.cat((lengths, angles), dim=1)

        self.mean.data = params.mean(dim=0)
        self.std.data = params.std(dim=0)

        # flitered in case of very similar shape (e.g. perovskite dataset)
        self.std.data[self.std.data < 1e-3] = 1.0

    def normalise_lattice(
        self, lattices: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        lengths, angles = LatticeScaler.get_lattices_parameters(lattices)

        return self.normalise(lengths, angles)

    def normalise(
        self, lengths: torch.FloatTensor, angles: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        lengths_scaled = (lengths - self.mean[:3]) / (self.std[:3] + 1e-6)
        angles_scaled = (angles - self.mean[3:]) / (self.std[3:] + 1e-6)

        return lengths_scaled, angles_scaled

    def denormalise(
        self, lengths: torch.FloatTensor, angles: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        lengths_scaled = lengths * self.std[:3] + self.mean[:3]
        angles_scaled = angles * self.std[3:] + self.mean[3:]

        return lengths_scaled, angles_scaled
