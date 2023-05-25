import torch
import torch.nn as nn


class Grad(nn.Module):
    def __init__(self):
        super().__init__()

        self.I = nn.Parameter(torch.eye(3), requires_grad=False)
        self.K = nn.Parameter(torch.tensor([[[0, 0, 0], [0, 0, 1], [0, -1, 0]], [[0, 0, -1], [0, 0, 0], [
                              1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]], dtype=torch.float32), requires_grad=False)

    def jacobian_atan2(self, y, x):
        diff_x = -y / (x ** 2 + y ** 2)
        diff_y = x / (x ** 2 + y ** 2)

        return diff_y, diff_x

    def jacobian_dot(self, x, y):
        return y.clone(), x.clone()

    def jacobian_norm(self, x):
        return x / x.norm(dim=1)[:, None]

    def jacobian_cross_norm(self, x, y):
        diff_cross_x = (self.K[None]*y[:, None, None, :]).sum(dim=3)
        diff_cross_y = -(self.K[None]*x[:, None, None, :]).sum(dim=3)

        diff_norm = self.jacobian_norm(torch.cross(x, y))
        diff_x = torch.bmm(diff_norm.unsqueeze(1), diff_cross_x).squeeze(1)
        diff_y = torch.bmm(diff_norm.unsqueeze(1), diff_cross_y).squeeze(1)

        return diff_x, diff_y

    def jacobian_m(self, u):
        diff_m = self.I[None, :, :, None]*u[:, None, None, :]
        return diff_m

    def jacobian_mu(self, m, u):
        diff_m = self.I[None, :, :, None]*u[:, None, None, :]
        diff_u = m.clone()
        return diff_m, diff_u

    def jacobian_angle_vector(self, u, v):
        diff_atan2_y, diff_atan2_x = self.jacobian_atan2(
            torch.cross(u, v).norm(dim=1), (u*v).sum(dim=1))
        diff_cross_norm_u, diff_cross_norm_v = self.jacobian_cross_norm(u, v)
        diff_dot_u, diff_dot_v = self.jacobian_dot(u, v)

        diff_u = diff_atan2_y[:, None] * diff_cross_norm_u + \
            diff_atan2_x[:, None] * diff_dot_u
        diff_v = diff_atan2_y[:, None] * diff_cross_norm_v + \
            diff_atan2_x[:, None] * diff_dot_v

        return diff_u, diff_v

    def grad_distance(self, rho, x_ij, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm(g, rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)

        diff_u = self.jacobian_norm(u)

        diff_g_u = self.jacobian_m(
            torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))

        diff_g = torch.einsum("bi,bijk->bjk", diff_u, diff_g_u)

        diff_x = torch.bmm(diff_u.unsqueeze(1), rho_prime).squeeze(1)

        diff_x_i = -diff_x
        diff_x_j = diff_x

        return diff_g, diff_x_i, diff_x_j

    def grad_distance_sym(self, rho, x_ij, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm((g + torch.transpose(g, 1, 2)), rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)

        diff_u = self.jacobian_norm(u)

        diff_g_u = self.jacobian_m(
            torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))

        diff_g_demi = torch.einsum("bi,bijk->bjk", diff_u, diff_g_u)

        diff_g = diff_g_demi + torch.transpose(diff_g_demi, 1, 2)

        diff_x = torch.bmm(diff_u.unsqueeze(1), rho_prime).squeeze(1)

        diff_x_i = -diff_x
        diff_x_j = diff_x

        return diff_g, diff_x_i, diff_x_j

    def grad_area(self, rho, x_ij, x_ik, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm(g, rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)
        v = torch.bmm(rho_prime, x_ik.unsqueeze(2)).squeeze(2)

        diff_u, diff_v = self.jacobian_cross_norm(u, v)
        diff_g_u = self.jacobian_m(
            torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))
        diff_g_v = self.jacobian_m(
            torch.bmm(rho, x_ik.unsqueeze(2)).squeeze(2))

        diff_g = 0.5 * (torch.einsum("bi,bijk->bjk", diff_u, diff_g_u) +
                        torch.einsum("bi,bijk->bjk", diff_v, diff_g_v))
        diff_vect = rho_prime

        diff_x_i = -0.5 * (torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1) +
                           torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1))
        diff_x_j = 0.5 * torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1)
        diff_x_k = 0.5 * torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1)

        return diff_g, diff_x_i, diff_x_j, diff_x_k

    def grad_area_sym(self, rho, x_ij, x_ik, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm((g + torch.transpose(g, 1, 2)), rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)
        v = torch.bmm(rho_prime, x_ik.unsqueeze(2)).squeeze(2)
        diff_u, diff_v = self.jacobian_cross_norm(u, v)
        diff_g_u = self.jacobian_m(
            torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))
        diff_g_v = self.jacobian_m(
            torch.bmm(rho, x_ik.unsqueeze(2)).squeeze(2))

        diff_g_demi = torch.einsum(
            "bi,bijk->bjk", diff_u, diff_g_u)+torch.einsum("bi,bijk->bjk", diff_v, diff_g_v)
        diff_g = 0.5 * (diff_g_demi + torch.transpose(diff_g_demi, 1, 2))

        diff_vect = rho_prime

        diff_x_i = -0.5 * (torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1) +
                           torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1))
        diff_x_j = 0.5 * torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1)
        diff_x_k = 0.5 * torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1)

        return diff_g, diff_x_i, diff_x_j, diff_x_k

    def grad_angle(self, rho, x_ij, x_ik, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm(g, rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)
        v = torch.bmm(rho_prime, x_ik.unsqueeze(2)).squeeze(2)
        diff_u, diff_v = self.jacobian_angle_vector(u, v)
        diff_g_u = self.jacobian_m(
            torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))
        diff_g_v = self.jacobian_m(
            torch.bmm(rho, x_ik.unsqueeze(2)).squeeze(2))

        diff_g = (torch.einsum("bi,bijk->bjk", diff_u, diff_g_u) +
                  torch.einsum("bi,bijk->bjk", diff_v, diff_g_v))
        diff_vect = rho_prime

        diff_x_i = -(torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1) +
                     torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1))
        diff_x_j = torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1)
        diff_x_k = torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1)

        return diff_g, diff_x_i, diff_x_j, diff_x_k

    def grad_angle_sym(self, rho, x_ij, x_ik, g=None):
        if g is None:
            rho_prime = rho
        else:
            rho_prime = torch.bmm((g + torch.transpose(g, 1, 2)), rho)

        u = torch.bmm(rho_prime, x_ij.unsqueeze(2)).squeeze(2)
        v = torch.bmm(rho_prime, x_ik.unsqueeze(2)).squeeze(2)
        diff_u, diff_v = self.jacobian_angle_vector(u, v)
        diff_g_u = self.jacobian_m(
            torch.bmm(rho, x_ij.unsqueeze(2)).squeeze(2))
        diff_g_v = self.jacobian_m(
            torch.bmm(rho, x_ik.unsqueeze(2)).squeeze(2))

        diff_g_demi = (torch.einsum("bi,bijk->bjk", diff_u, diff_g_u) +
                       torch.einsum("bi,bijk->bjk", diff_v, diff_g_v))
        diff_g = diff_g_demi + torch.transpose(diff_g_demi, 1, 2)

        diff_vect = rho_prime

        diff_x_i = -(torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1) +
                     torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1))
        diff_x_j = torch.bmm(diff_u.unsqueeze(1), diff_vect).squeeze(1)
        diff_x_k = torch.bmm(diff_v.unsqueeze(1), diff_vect).squeeze(1)

        return diff_g, diff_x_i, diff_x_j, diff_x_k
