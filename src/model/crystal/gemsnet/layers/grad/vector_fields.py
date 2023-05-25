import torch
import torch.nn as nn

from .grad import Grad

import enum
from typing import List
import abc


class VectorFields(nn.Module):
    def __init__(self, vector_fields_triplets, normalize: bool = True):
        super().__init__()

        self.vector_fields_triplets = vector_fields_triplets
        self.normalize = normalize

    @property
    def triplets_dim(self):
        return len(self.vector_fields_triplets)

    def forward(
        self,
        cell: torch.FloatTensor,
        batch_triplets: torch.LongTensor,
        e_ij: torch.FloatTensor,
        e_ik: torch.FloatTensor,
    ):
        raise NotImplementedError


class VFGrad(VectorFields):
    class AbstractVectorFields(metaclass=abc.ABCMeta):
        def __init__(self):
            self.grad = None

        def set_grad(self, grad):
            self.grad = grad

        @abc.abstractclassmethod
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ) -> torch.FloatTensor:
            pass

    class VectorFieldsNormij(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance(cell, x_ij)[0]

    class VectorFieldsNormijSym(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance_sym(cell, x_ij)[0]

    class VectorFieldsNormik(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance(cell, x_ik)[0]

    class VectorFieldsNormikSym(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance_sym(cell, x_ik)[0]

    class VectorFieldsAngle(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_angle(cell, x_ij, x_ik)[0]

    class VectorFieldsAngleSym(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_angle_sym(cell, x_ij, x_ik)[0]

    class VectorFieldsArea(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_area(cell, x_ij, x_ik)[0]

    class VectorFieldsAreaSym(AbstractVectorFields):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_area_sym(cell, x_ij, x_ik)[0]

    def __init__(self, vector_fields_triplets, normalize: bool = True):
        assert isinstance(normalize, bool)
        assert isinstance(vector_fields_triplets, set)
        assert len(vector_fields_triplets) > 0

        for op in vector_fields_triplets:
            assert isinstance(op, VFGrad.AbstractVectorFields)

        super().__init__(
            vector_fields_triplets=vector_fields_triplets, normalize=normalize
        )

        self.grad = Grad()

        for op in self.vector_fields_triplets:
            op.set_grad(self.grad)

    def forward(
        self,
        cell: torch.FloatTensor,
        batch_triplets: torch.LongTensor,
        e_ij: torch.FloatTensor,
        e_ik: torch.FloatTensor,
    ):
        ops = []

        for op in self.vector_fields_triplets:
            ops.append(op.op(cell[batch_triplets], e_ij, e_ik))

        ops = torch.stack(ops, dim=1)

        return ops


def make_vector_fields(config):
    assert "type" in config
    assert "normalize" in config
    assert "triplets" in config

    assert config["type"] in ["grad"]

    if config["type"] == "grad":
        vector_fields_triplets = set()

        ops_dict = {
            "n_ij": VFGrad.VectorFieldsNormij,
            "n_ij_sym": VFGrad.VectorFieldsNormijSym,
            "n_ik": VFGrad.VectorFieldsNormik,
            "n_ik_sym": VFGrad.VectorFieldsNormikSym,
            "angle": VFGrad.VectorFieldsAngle,
            "angle_sym": VFGrad.VectorFieldsAngleSym,
            "area": VFGrad.VectorFieldsArea,
            "area_sym": VFGrad.VectorFieldsAreaSym,
        }

        for op in config["triplets"]:
            assert op in ops_dict
            vector_fields_triplets.add(ops_dict[op]())

        ops = VFGrad(vector_fields_triplets, normalize=config["normalize"])

    return ops
