import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean

import crystallographic_graph


class MinDistanceLoss(nn.Module):
    def __init__(self, center: bool = True):
        super().__init__()

        self.center = center

        self.offset = nn.Parameter(
            torch.tensor(
                [
                    [-1, -1, -1],
                    [-1, -1, 0],
                    [-1, -1, 1],
                    [-1, 0, -1],
                    [-1, 0, 0],
                    [-1, 0, 1],
                    [-1, 1, -1],
                    [-1, 1, 0],
                    [-1, 1, 1],
                    [0, -1, -1],
                    [0, -1, 0],
                    [0, -1, 1],
                    [0, 0, -1],
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, -1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, -1, -1],
                    [1, -1, 0],
                    [1, -1, 1],
                    [1, 0, -1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, -1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        x_tilde: torch.FloatTensor,
        num_atoms: torch.LongTensor,
    ) -> torch.FloatTensor:

        struct_idx = torch.arange(cell.shape[0], device=cell.device)
        batch = struct_idx.repeat_interleave(num_atoms)

        euc_x_tilde = torch.einsum(cell[batch], [0, 1, 2], x_tilde, [0, 2], [0, 1])

        euc_x = torch.einsum(
            cell[batch],
            [0, 2, 3],
            x[:, None, :] + self.offset[None, :, :],
            [0, 1, 3],
            [0, 1, 2],
        )

        min_idx = (euc_x_tilde[:, None] - euc_x).norm(dim=2).argmin(dim=1)

        if self.center:
            center = scatter_mean(
                x + self.offset[min_idx] - x_tilde, batch, dim=0, dim_size=cell.shape[0]
            )

            return F.l1_loss(x_tilde, x + self.offset[min_idx] - center[batch])
        else:
            return F.l1_loss(x_tilde, x + self.offset[min_idx])
