import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

from .grad import Grad

import unittest
import time


class TestGrad(unittest.TestCase):
    batch_size = 1024
    verbose = True

    def log(self, *args, **kwargs):
        if TestGrad.verbose:
            print(*args, **kwargs)

    def assertAlmostEqualsTensors(self, x, y, places):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        max_error = (x-y).abs().max().item()
        self.log(f"max error: {max_error:.5f}")
        self.assertAlmostEqual(max_error, 0, places)

    def batched_jacobian(self, fn, inputs):
        lst_grad = [[] for _ in inputs]
        for vars in zip(*inputs):
            grad_vars = jacobian(fn, vars)

            for idx, g in enumerate(grad_vars):
                lst_grad[idx].append(g)

        for idx, g in enumerate(lst_grad):
            lst_grad[idx] = torch.stack(g, dim=0)

        return tuple(lst_grad)

    def test_norm(self):
        torch.manual_seed(0)

        x = torch.randn(TestGrad.batch_size, 3)

        grad_fn = Grad()

        (gt_grad_x,) = self.batched_jacobian(lambda u: u.norm(), (x,))

        grad_x = grad_fn.jacobian_norm(x)

        self.assertAlmostEqualsTensors(gt_grad_x, grad_x, places=6)

    def test_dot(self):
        torch.manual_seed(0)

        x = torch.randn(TestGrad.batch_size, 3)
        y = torch.randn(TestGrad.batch_size, 3)

        grad_fn = Grad()

        (gt_grad_x, gt_grad_y) = self.batched_jacobian(
            lambda x, y: x.dot(y), (x, y))

        grad_x, grad_y = grad_fn.jacobian_dot(x, y)

        self.assertAlmostEqualsTensors(gt_grad_x, grad_x, places=6)
        self.assertAlmostEqualsTensors(gt_grad_y, grad_y, places=6)

    def test_cross_norm(self):
        torch.manual_seed(0)

        x = torch.randn(TestGrad.batch_size, 3)
        y = torch.randn(TestGrad.batch_size, 3)

        grad_fn = Grad()

        (gt_grad_x, gt_grad_y) = self.batched_jacobian(
            lambda x, y: torch.cross(x, y).norm(), (x, y))

        grad_x, grad_y = grad_fn.jacobian_cross_norm(x, y)

        self.assertAlmostEqualsTensors(gt_grad_x, grad_x, places=6)
        self.assertAlmostEqualsTensors(gt_grad_y, grad_y, places=6)

    def test_matrix_vector(self):
        torch.manual_seed(0)

        m = torch.randn(TestGrad.batch_size, 3, 3)
        u = torch.randn(TestGrad.batch_size, 3)

        grad_fn = Grad()

        (gt_grad_m, gt_grad_u) = self.batched_jacobian(
            lambda x, y: x @ y, (m, u))

        grad_m, grad_u = grad_fn.jacobian_mu(m, u)

        self.assertAlmostEqualsTensors(gt_grad_m, grad_m, places=6)
        self.assertAlmostEqualsTensors(gt_grad_u, grad_u, places=6)

        grad_m = grad_fn.jacobian_m(u)

        self.assertAlmostEqualsTensors(gt_grad_m, grad_m, places=6)

    def test_atan2(self):
        torch.manual_seed(0)

        x = torch.randn(TestGrad.batch_size)
        y = torch.randn(TestGrad.batch_size)

        grad_fn = Grad()

        (gt_grad_y, gt_grad_x) = self.batched_jacobian(
            lambda y, x: torch.atan2(y, x), (y, x))

        grad_y, grad_x = grad_fn.jacobian_atan2(y, x)

        self.assertAlmostEqualsTensors(gt_grad_x, grad_x, places=5)
        self.assertAlmostEqualsTensors(gt_grad_y, grad_y, places=5)

    def test_atan2(self):
        torch.manual_seed(0)

        x = torch.randn(TestGrad.batch_size)
        y = torch.randn(TestGrad.batch_size)

        grad_fn = Grad()

        (gt_grad_y, gt_grad_x) = self.batched_jacobian(
            lambda y, x: torch.atan2(y, x), (y, x))

        grad_y, grad_x = grad_fn.jacobian_atan2(y, x)

        self.assertAlmostEqualsTensors(gt_grad_x, grad_x, places=5)
        self.assertAlmostEqualsTensors(gt_grad_y, grad_y, places=5)

    def test_angle_vector(self):
        torch.manual_seed(0)

        u = torch.randn(TestGrad.batch_size, 3)
        v = torch.randn(TestGrad.batch_size, 3)

        grad_fn = Grad()

        (gt_grad_u, gt_grad_v) = self.batched_jacobian(
            lambda x, y: torch.atan2(torch.cross(x, y).norm(), x.dot(y)),
            (u, v))

        grad_u, grad_v = grad_fn.jacobian_angle_vector(u, v)

        self.assertAlmostEqualsTensors(gt_grad_u, grad_u, places=5)
        self.assertAlmostEqualsTensors(gt_grad_v, grad_v, places=5)

    def test_distance(self):
        torch.manual_seed(0)

        grad_fn = Grad()

        g = torch.randn(TestGrad.batch_size, 3, 3)
        rho = torch.matrix_exp(torch.randn(TestGrad.batch_size, 3, 3))
        x_ij = torch.randn(TestGrad.batch_size, 3)

        def get_distance(g, rho, xij):
            u = g @ rho @ xij
            return u.norm()

        t0 = time.time()
        grad_g, grad_x_i, grad_x_j = grad_fn.grad_distance(
            rho, x_ij, g=g)
        t1 = time.time()
        self.log(f"grad distance {t1-t0:.6f}sec")

        (gt_grad_g, _, gt_grad_x_ij) = self.batched_jacobian(
            lambda g, rho, xij: get_distance(
                g, rho, xij), (g, rho, x_ij)
        )
        gt_grad_x_i = -gt_grad_x_ij
        gt_grad_x_j = gt_grad_x_ij

        self.assertAlmostEqualsTensors(gt_grad_g, grad_g, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_i, grad_x_i, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_j, grad_x_j, places=3)

    def test_distance_sym(self):
        torch.manual_seed(0)

        grad_fn = Grad()

        g = torch.randn(TestGrad.batch_size, 3, 3)
        rho = torch.matrix_exp(torch.randn(TestGrad.batch_size, 3, 3))
        x_ij = torch.randn(TestGrad.batch_size, 3)

        def get_distance(g, rho, xij):
            u = (g + g.t()) @ rho @ xij
            return u.norm()

        t0 = time.time()
        grad_g, grad_x_i, grad_x_j = grad_fn.grad_distance_sym(
            rho, x_ij, g=g)
        t1 = time.time()
        self.log(f"grad distance sym {t1-t0:.6f}sec")

        (gt_grad_g, _, gt_grad_x_ij) = self.batched_jacobian(
            lambda g, rho, xij: get_distance(
                g, rho, xij), (g, rho, x_ij)
        )
        gt_grad_x_i = -gt_grad_x_ij
        gt_grad_x_j = gt_grad_x_ij

        self.assertAlmostEqualsTensors(gt_grad_g, grad_g, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_i, grad_x_i, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_j, grad_x_j, places=3)

    def test_area(self):
        torch.manual_seed(0)

        grad_fn = Grad()

        g = torch.randn(TestGrad.batch_size, 3, 3)
        rho = torch.matrix_exp(torch.randn(TestGrad.batch_size, 3, 3))
        x_ij = torch.randn(TestGrad.batch_size, 3)
        x_ik = torch.randn(TestGrad.batch_size, 3)

        def get_area(g, rho, xij, xik):
            u = g @ rho @ xij
            v = g @ rho @ xik
            return 0.5 * torch.cross(u, v).norm()

        t0 = time.time()
        grad_g, grad_x_i, grad_x_j, grad_x_k = grad_fn.grad_area(
            rho, x_ij, x_ik, g=g)
        t1 = time.time()
        self.log(f"grad area {t1-t0:.6f}sec")

        (gt_grad_g, _, gt_grad_x_ij, gt_grad_x_ik) = self.batched_jacobian(
            lambda g, rho, xij, xik: get_area(
                g, rho, xij, xik), (g, rho, x_ij, x_ik)
        )
        gt_grad_x_i = -(gt_grad_x_ij+gt_grad_x_ik)
        gt_grad_x_j = gt_grad_x_ij
        gt_grad_x_k = gt_grad_x_ik

        self.assertAlmostEqualsTensors(gt_grad_g, grad_g, places=2)
        self.assertAlmostEqualsTensors(gt_grad_x_i, grad_x_i, places=2)
        self.assertAlmostEqualsTensors(gt_grad_x_j, grad_x_j, places=2)
        self.assertAlmostEqualsTensors(gt_grad_x_k, grad_x_k, places=2)

    def test_area_sym(self):
        torch.manual_seed(0)

        grad_fn = Grad()

        g = torch.randn(TestGrad.batch_size, 3, 3)
        rho = torch.matrix_exp(torch.randn(TestGrad.batch_size, 3, 3))
        x_ij = torch.randn(TestGrad.batch_size, 3)
        x_ik = torch.randn(TestGrad.batch_size, 3)

        def get_area(g, rho, xij, xik):
            u = (g + g.t()) @ rho @ xij
            v = (g + g.t()) @ rho @ xik
            return 0.5 * torch.cross(u, v).norm()

        t0 = time.time()
        grad_g, grad_x_i, grad_x_j, grad_x_k = grad_fn.grad_area_sym(
            rho, x_ij, x_ik, g=g)
        t1 = time.time()
        self.log(f"grad area sym {t1-t0:.6f}sec")

        (gt_grad_g, _, gt_grad_x_ij, gt_grad_x_ik) = self.batched_jacobian(
            lambda g, rho, xij, xik: get_area(
                g, rho, xij, xik), (g, rho, x_ij, x_ik)
        )
        gt_grad_x_i = -(gt_grad_x_ij+gt_grad_x_ik)
        gt_grad_x_j = gt_grad_x_ij
        gt_grad_x_k = gt_grad_x_ik

        self.assertAlmostEqualsTensors(gt_grad_g, grad_g, places=2)
        self.assertAlmostEqualsTensors(gt_grad_x_i, grad_x_i, places=2)
        self.assertAlmostEqualsTensors(gt_grad_x_j, grad_x_j, places=2)
        self.assertAlmostEqualsTensors(gt_grad_x_k, grad_x_k, places=2)

    def test_angle(self):
        torch.manual_seed(0)

        grad_fn = Grad()

        g = torch.randn(TestGrad.batch_size, 3, 3)
        rho = torch.matrix_exp(torch.randn(TestGrad.batch_size, 3, 3))
        x_ij = torch.randn(TestGrad.batch_size, 3)
        x_ik = torch.randn(TestGrad.batch_size, 3)

        def get_angle(g, rho, xij, xik):
            u = g @ rho @ xij
            v = g @ rho @ xik
            return torch.atan2(torch.cross(u, v).norm(), u.dot(v))

        t0 = time.time()
        grad_g, grad_x_i, grad_x_j, grad_x_k = grad_fn.grad_angle(
            rho, x_ij, x_ik, g=g)
        t1 = time.time()
        self.log(f"grad angle {t1-t0:.6f}sec")

        (gt_grad_g, _, gt_grad_x_ij, gt_grad_x_ik) = self.batched_jacobian(
            lambda g, rho, xij, xik: get_angle(
                g, rho, xij, xik), (g, rho, x_ij, x_ik)
        )
        gt_grad_x_i = -(gt_grad_x_ij+gt_grad_x_ik)
        gt_grad_x_j = gt_grad_x_ij
        gt_grad_x_k = gt_grad_x_ik

        self.assertAlmostEqualsTensors(gt_grad_g, grad_g, places=4)
        self.assertAlmostEqualsTensors(gt_grad_x_i, grad_x_i, places=4)
        self.assertAlmostEqualsTensors(gt_grad_x_j, grad_x_j, places=4)
        self.assertAlmostEqualsTensors(gt_grad_x_k, grad_x_k, places=4)

    def test_angle_sym(self):
        torch.manual_seed(0)

        grad_fn = Grad()

        g = torch.randn(TestGrad.batch_size, 3, 3)
        rho = torch.matrix_exp(torch.randn(TestGrad.batch_size, 3, 3))
        x_ij = torch.randn(TestGrad.batch_size, 3)
        x_ik = torch.randn(TestGrad.batch_size, 3)

        def get_angle(g, rho, xij, xik):
            u = (g + g.t()) @ rho @ xij
            v = (g + g.t()) @ rho @ xik
            return torch.atan2(torch.cross(u, v).norm(), u.dot(v))

        t0 = time.time()
        grad_g, grad_x_i, grad_x_j, grad_x_k = grad_fn.grad_angle_sym(
            rho, x_ij, x_ik, g=g)
        t1 = time.time()
        self.log(f"grad angle sym {t1-t0:.6f}sec")

        (gt_grad_g, _, gt_grad_x_ij, gt_grad_x_ik) = self.batched_jacobian(
            lambda g, rho, xij, xik: get_angle(
                g, rho, xij, xik), (g, rho, x_ij, x_ik)
        )
        gt_grad_x_i = -(gt_grad_x_ij+gt_grad_x_ik)
        gt_grad_x_j = gt_grad_x_ij
        gt_grad_x_k = gt_grad_x_ik

        self.assertAlmostEqualsTensors(gt_grad_g, grad_g, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_i, grad_x_i, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_j, grad_x_j, places=3)
        self.assertAlmostEqualsTensors(gt_grad_x_k, grad_x_k, places=3)


if __name__ == "__main__":
    unittest.main()
