import torch
import torch.nn as nn
import torch.nn.functional as F

import crystallographic_graph


class RelativeLoss(nn.Module):
    def __init__(self, knn: int = 4):
        super().__init__()

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

        return F.l1_loss(e_tilde_ij, e_ij), (e_tilde_ij - e_ij).abs()
