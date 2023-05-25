import torch
import torch.nn as nn

from src.utils.geometry import Geometry
from .grad import Grad

import enum
from typing import List
import abc


class Operator(nn.Module):
    def __init__(self, operators_edges, operators_triplets, normalize: bool = True):
        super().__init__()

        self.operators_edges = operators_edges
        self.operators_triplets = operators_triplets
        self.normalize = normalize

    @property
    def edges_dim(self):
        return len(self.operators_edges)

    @property
    def triplets_dim(self):
        return len(self.operators_triplets)

    def forward(self, geometry):
        raise NotImplementedError


class OpKetBra(Operator):
    class AbstractOperator(metaclass=abc.ABCMeta):
        @abc.abstractclassmethod
        def op(
            self, u: torch.FloatTensor, v: torch.FloatTensor = None
        ) -> torch.FloatTensor:
            pass

    class OperatorVij(AbstractOperator):
        def op(self, u, v=None):
            return torch.bmm(u.unsqueeze(2), u.unsqueeze(1))

    class OperatorVik(AbstractOperator):
        def op(self, u, v=None):
            return torch.bmm(v.unsqueeze(2), v.unsqueeze(1))

    class OperatorVijk(AbstractOperator):
        def op(self, u, v=None):
            return torch.bmm(u.unsqueeze(2), v.unsqueeze(1))

    class OperatorVikj(AbstractOperator):
        def op(self, u, v=None):
            return torch.bmm(v.unsqueeze(2), u.unsqueeze(1))

    class OperatorVijkSym(AbstractOperator):
        def op(self, u, v=None):
            return 0.5 * (
                torch.bmm(u.unsqueeze(2), v.unsqueeze(1))
                + torch.bmm(v.unsqueeze(2), u.unsqueeze(1))
            )

    def __init__(self, operators_edges, operators_triplets, normalize: bool = True):
        assert isinstance(normalize, bool)
        assert isinstance(operators_edges, set)
        assert isinstance(operators_triplets, set)

        for op in operators_edges:
            assert isinstance(op, (OpKetBra.OperatorVij,))
        for op in operators_triplets:
            assert isinstance(op, OpKetBra.AbstractOperator)

        super().__init__(
            operators_edges=operators_edges,
            operators_triplets=operators_triplets,
            normalize=normalize,
        )

    def forward(self, geometry):
        edges_ops = []
        triplets_ops = []

        for op in self.operators_edges:
            if self.normalize:
                edges_ops.append(op.op(geometry.edges_u_ij))
            else:
                edges_ops.append(op.op(geometry.edges_v_ij))

        for op in self.operators_triplets:
            if self.normalize:
                triplets_ops.append(
                    op.op(geometry.triplets_u_ij, geometry.triplets_u_ik)
                )
            else:
                triplets_ops.append(
                    op.op(geometry.triplets_v_ij, geometry.triplets_v_ik)
                )

        if len(edges_ops) > 0:
            edges_ops = torch.stack(edges_ops, dim=1)
        else:
            edges_ops = None

        if len(triplets_ops) > 0:
            triplets_ops = torch.stack(triplets_ops, dim=1)
        else:
            triplets_ops = None

        return edges_ops, triplets_ops


class OpSymSkew(Operator):
    class AbstractOperator(metaclass=abc.ABCMeta):
        @abc.abstractclassmethod
        def op(
            self, u: torch.FloatTensor, v: torch.FloatTensor = None
        ) -> torch.FloatTensor:
            pass

    class OperatorVij(AbstractOperator):
        def op(self, u, v=None):
            return u[:, :, None] + u[:, None, :]

    class OperatorVik(AbstractOperator):
        def op(self, u, v=None):
            return v[:, :, None] + v[:, None, :]

    class OperatorVijk(AbstractOperator):
        def op(self, u, v=None):
            return u[:, :, None] + v[:, None, :]

    class OperatorVikj(AbstractOperator):
        def op(self, u, v=None):
            return v[:, :, None] + u[:, None, :]

    class OperatorVijkSym(AbstractOperator):
        def op(self, u, v=None):
            return 0.5 * (u[:, :, None] + v[:, None, :] + v[:, :, None] + u[:, None, :])

    def __init__(self, operators_edges, operators_triplets, normalize: bool = True):
        assert isinstance(normalize, bool)
        assert isinstance(operators_edges, set)
        assert isinstance(operators_triplets, set)

        for op in operators_edges:
            assert isinstance(op, (OpSymSkew.OperatorVij,))
        for op in operators_triplets:
            assert isinstance(op, OpSymSkew.AbstractOperator)

        super().__init__(
            operators_edges=operators_edges,
            operators_triplets=operators_triplets,
            normalize=normalize,
        )

    def forward(self, geometry):
        edges_ops = []
        triplets_ops = []

        for op in self.operators_edges:
            if self.normalize:
                edges_ops.append(op.op(geometry.edges_u_ij))
            else:
                edges_ops.append(op.op(geometry.edges_v_ij))

        for op in self.operators_triplets:
            if self.normalize:
                triplets_ops.append(
                    op.op(geometry.triplets_u_ij, geometry.triplets_u_ik)
                )
            else:
                triplets_ops.append(
                    op.op(geometry.triplets_v_ij, geometry.triplets_v_ik)
                )

        if len(edges_ops) > 0:
            edges_ops = torch.stack(edges_ops, dim=1)
        else:
            edges_ops = None

        if len(triplets_ops) > 0:
            triplets_ops = torch.stack(triplets_ops, dim=1)
        else:
            triplets_ops = None

        return edges_ops, triplets_ops


class OpGrad(Operator):
    class AbstractOperator(metaclass=abc.ABCMeta):
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

    class OperatorNormij(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance(cell, x_ij)[0]

    class OperatorNormijSym(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance_sym(cell, x_ij)[0]

    class OperatorNormik(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance(cell, x_ik)[0]

    class OperatorNormikSym(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_distance_sym(cell, x_ik)[0]

    class OperatorAngle(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_angle(cell, x_ij, x_ik)[0]

    class OperatorAngleSym(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_angle_sym(cell, x_ij, x_ik)[0]

    class OperatorArea(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_area(cell, x_ij, x_ik)[0]

    class OperatorAreaSym(AbstractOperator):
        def op(
            self,
            cell: torch.FloatTensor,
            x_ij: torch.FloatTensor,
            x_ik: torch.FloatTensor = None,
        ):
            return self.grad.grad_area_sym(cell, x_ij, x_ik)[0]

    def __init__(self, operators_edges, operators_triplets, normalize: bool = True):
        assert isinstance(normalize, bool)
        assert isinstance(operators_edges, set)
        assert isinstance(operators_triplets, set)

        for op in operators_edges:
            assert isinstance(op, (OpGrad.OperatorNormij, OpGrad.OperatorNormijSym))
        for op in operators_triplets:
            assert isinstance(op, OpGrad.AbstractOperator)

        super().__init__(
            operators_edges=operators_edges,
            operators_triplets=operators_triplets,
            normalize=normalize,
        )

        self.grad = Grad()

        for op in self.operators_edges:
            op.set_grad(self.grad)
        for op in self.operators_triplets:
            op.set_grad(self.grad)

    def forward(self, geometry):
        edges_ops = []
        triplets_ops = []

        for op in self.operators_edges:
            edges_ops.append(
                op.op(geometry.cell[geometry.batch_edges], geometry.edges_e_ij)
            )

        for op in self.operators_triplets:
            triplets_ops.append(
                op.op(
                    geometry.cell[geometry.batch_triplets],
                    geometry.triplets_e_ij,
                    geometry.triplets_e_ik,
                )
            )

        if len(edges_ops) > 0:
            edges_ops = torch.stack(edges_ops, dim=1)
        else:
            edges_ops = None

        if len(triplets_ops) > 0:
            triplets_ops = torch.stack(triplets_ops, dim=1)
        else:
            triplets_ops = None

        return edges_ops, triplets_ops


def make_operator(config):
    assert "type" in config
    assert "normalize" in config
    assert "edges" in config
    assert "triplets" in config

    assert config["type"] in ["ket-bra", "sym-skew", "grad"]

    if config["type"] == "ket-bra":
        operators_edges = set()
        operators_triplets = set()

        ops_dict = {
            "v_ij": OpKetBra.OperatorVij,
            "v_ik": OpKetBra.OperatorVik,
            "v_ijk": OpKetBra.OperatorVijk,
            "v_ikj": OpKetBra.OperatorVikj,
            "v_ijk_sym": OpKetBra.OperatorVijkSym,
        }

        for op in config["edges"]:
            assert op in ["v_ij"]
            operators_edges.add(ops_dict[op]())

        for op in config["triplets"]:
            assert op in ops_dict
            operators_triplets.add(ops_dict[op]())

        ops = OpKetBra(
            operators_edges, operators_triplets, normalize=config["normalize"]
        )
    elif config["type"] == "sym-skew":
        operators_edges = set()
        operators_triplets = set()

        ops_dict = {
            "v_ij": OpSymSkew.OperatorVij,
            "v_ik": OpSymSkew.OperatorVik,
            "v_ijk": OpSymSkew.OperatorVijk,
            "v_ikj": OpSymSkew.OperatorVikj,
            "v_ijk_sym": OpSymSkew.OperatorVijkSym,
        }

        for op in config["edges"]:
            assert op in ["v_ij"]
            operators_edges.add(ops_dict[op]())

        for op in config["triplets"]:
            assert op in ops_dict
            operators_triplets.add(ops_dict[op]())

        ops = OpSymSkew(
            operators_edges, operators_triplets, normalize=config["normalize"]
        )
    elif config["type"] == "grad":
        operators_edges = set()
        operators_triplets = set()

        ops_dict = {
            "n_ij": OpGrad.OperatorNormij,
            "n_ij_sym": OpGrad.OperatorNormijSym,
            "n_ik": OpGrad.OperatorNormik,
            "n_ik_sym": OpGrad.OperatorNormikSym,
            "angle": OpGrad.OperatorAngle,
            "angle_sym": OpGrad.OperatorAngleSym,
            "area": OpGrad.OperatorArea,
            "area_sym": OpGrad.OperatorAreaSym,
        }

        for op in config["edges"]:
            assert op in ["n_ij", "n_ij_sym"]
            operators_edges.add(ops_dict[op]())

        for op in config["triplets"]:
            assert op in ops_dict
            operators_triplets.add(ops_dict[op]())

        ops = OpGrad(operators_edges, operators_triplets, normalize=config["normalize"])

    return ops
