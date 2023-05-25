import torch
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io import write
from ase.spacegroup import crystal
from ase.geometry import cell_to_cellpar

import os

def save_snapshot(batch, model, filename, n=(6, 2), figsize=(30, 30),noise_pos=0.05):
    x_thild = (batch.pos + noise_pos * torch.randn_like(batch.pos)) % 1.0
    batch.x_thild = x_thild
    eye = torch.eye(3, device=batch.pos.device).unsqueeze(0).repeat(batch.cell.shape[0], 1, 1)

    batch.rho_tilde = eye.clone().detach()
    x_prime, x_traj, rho_prime = model.forward(eye, x_thild, batch.z, batch.num_atoms)
    batch.rho_prime = rho_prime.clone().detach()
    batch.x_prime = x_prime.clone().detach()
    
    _, ax = plt.subplots(n[0], n[1] * 3, figsize=figsize)

    batch_cpu = batch.clone()
    batch_cpu = batch_cpu.cpu().detach()

    for i in range(n[0]):
        for j in range(n[1]):
            idx = j + i * n[1]
            mask = batch_cpu.batch == idx

            path, _ = os.path.splitext(os.path.abspath(filename))
            path = os.path.join(path, f"{idx}")
            os.makedirs(path, exist_ok=True)

            atoms = crystal(
                batch_cpu.z[mask].numpy(),
                basis=batch_cpu.x_thild[mask].numpy(),
                cell=batch_cpu.rho_tilde[idx].numpy(),
            )
            write(os.path.join(path, f"noisy.cif"), atoms)
            plot_atoms(atoms, ax[i][j * 3 + 0], radii=0.06)
            ax[i][j * 3 + 0].set_axis_off()

            atoms = crystal(
                batch_cpu.z[mask].numpy(),
                basis=batch_cpu.x_prime[mask].numpy(),
                cellpar=cell_to_cellpar(batch_cpu.rho_prime[idx].numpy()),
            )
            write(os.path.join(path, f"denoised.cif"), atoms)
            plot_atoms(atoms, ax[i][j * 3 + 1], radii=0.3)
            ax[i][j * 3 + 1].set_axis_off()

            atoms = crystal(
                batch_cpu.z[mask].numpy(),
                basis=batch_cpu.pos[mask].numpy(),
                cell=batch_cpu.cell[idx].numpy(),
            )
            write(os.path.join(path, f"original.cif"), atoms)
            plot_atoms(atoms, ax[i][j * 3 + 2], radii=0.3)
            ax[i][j * 3 + 2].set_axis_off()
    plt.savefig(filename)
    plt.close()
