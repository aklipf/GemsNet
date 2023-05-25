import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .gemsnet import GemsNetT

class GemsNetDenoiser(nn.Module):
    def __init__(
        self,
        features: int,
        knn:int,
        num_blocks:int,
        vector_fields: dict,
    ):
        super().__init__()

        self.gemsnet = GemsNetT(features,knn=knn,vector_fields=vector_fields)

    def forward(
        self,
        cell: torch.FloatTensor,
        x_thild: torch.FloatTensor,
        z: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:
        
        x_prime, x_traj, _, rho_prime, _ = self.gemsnet(cell, x_thild, z, num_atoms)
        return x_prime, x_traj, rho_prime
