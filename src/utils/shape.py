import torch
from typing import Tuple, List, Union, Dict
from collections import namedtuple


class shape:
    def __init__(self, *dim: Union[int, str], dtype=None):
        assert isinstance(dim, tuple)

        for d in dim:
            assert (type(d) == int and -1 <= d) or type(d) == str

        assert (dtype is None) or isinstance(dtype, torch.dtype)

        self.dim = dim
        self.dtype = dtype

    def get_dim(self, dim: List[Union[int, str]], context: Dict[str, int] = {}):
        dim_eval = []

        for d in dim:
            if type(d) == str and (d in context):
                dim_eval.append(context[d])
            else:
                dim_eval.append(d)

        return tuple(dim_eval)

    def assert_match(self, x: torch.Tensor, context: Dict[str, int] = {}):
        assert isinstance(x, torch.Tensor), "x is not a Tensor"

        assert x.dim() == len(
            self.dim
        ), f"the dimension of x should match with {self.dim}"

        for x_dim, trg_dim in zip(x.shape, self.dim):
            if (trg_dim is None) or (trg_dim == -1):
                continue

            if type(trg_dim) == str:
                if trg_dim in context:
                    trg_dim = context[trg_dim]
                else:
                    context[trg_dim] = x_dim
                    continue

            assert (
                x_dim == trg_dim
            ), f"the shape of x {tuple(x.shape)} should match with {self.get_dim(self.dim,context)}"

        if self.dtype is not None:
            assert (
                x.dtype == self.dtype
            ), f"the data type of x ({x.dtype}) should match with {self.dtype}"

        return context


def build_shapes(context: Dict[str, int]) -> namedtuple("shapes", tuple()):
    return namedtuple("shapes", context.keys())(*context.values())


def assert_tensor_match(
    *args: Tuple[torch.Tensor, shape]
) -> namedtuple("shapes", tuple()):
    context = {}
    for x, s in args:
        context = s.assert_match(x, context=context)

    return build_shapes(context)
