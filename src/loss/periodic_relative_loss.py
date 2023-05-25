import torch
import torch.nn as nn
import torch.nn.functional as F

import crystallographic_graph

from .min_distance_loss import MinDistanceLoss


class PeriodicRelativeLoss(nn.Module):
    def __init__(self, knn: int = 4):
        super().__init__()

        self.min_distance = MinDistanceLoss()
        self.knn = knn

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_tilde: torch.FloatTensor,
        num_atoms: torch.FloatTensor,
    ) -> torch.FloatTensor:
        edges = crystallographic_graph.make_graph(cell, x, num_atoms, knn=self.knn)
        e_ij = x[edges.dst] + edges.cell - x[edges.src]
        e_tilde_ij = x_tilde[edges.dst] + edges.cell - x_tilde[edges.src]

        struct_idx = torch.arange(cell.shape[0], device=cell.device)
        batch = struct_idx.repeat_interleave(num_atoms)
        batch_edges = batch[edges.src]

        _, num_edges = torch.unique_consecutive(batch_edges, return_counts=True)

        return self.min_distance(cell, e_tilde_ij, e_ij, num_edges)
