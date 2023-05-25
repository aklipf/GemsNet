import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.scaler import LatticeScaler

from typing import Dict, Tuple, Union


class LatticeParametersLoss(nn.Module):
    def __init__(self, lattice_scaler: LatticeScaler = None, distance: str = "l1"):
        super().__init__()
        assert distance in ["l1", "mse"]

        if lattice_scaler is None:
            lattice_scaler = LatticeScaler()

        assert isinstance(lattice_scaler, LatticeScaler)

        self.lattice_scaler = lattice_scaler
        self.distance = distance

    def forward(
        self,
        source: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]],
        target: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]],
    ) -> torch.FloatTensor:
        if isinstance(target, tuple):
            param_src = self.lattice_scaler.normalise(source)
        else:
            param_src = self.lattice_scaler.normalise_lattice(source)

        if isinstance(target, tuple):
            param_tgt = self.lattice_scaler.normalise(target)
        else:
            param_tgt = self.lattice_scaler.normalise_lattice(target)

        y_src = torch.cat(param_src, dim=1)
        y_tgt = torch.cat(param_tgt, dim=1)

        if self.distance == "l1":
            return F.l1_loss(y_src, y_tgt)
        if self.distance == "mse":
            return F.mse_loss(y_src, y_tgt)

        raise Exception(f"unkown distance {self.distance}")
