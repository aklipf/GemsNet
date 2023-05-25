import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.io import write
from ase.spacegroup import crystal

import os
import json
import math
import random
import datetime

from src.utils.scaler import LatticeScaler
from src.utils.data import MP20, Carbon24, Perov5
from src.utils.hparams import Hparams
from src.utils.snapshot import save_snapshot
from src.utils.metrics import get_metrics
from src.model.crystal import EGNNDenoiser, GemsNetDenoiser
from src.loss import OptimalTrajLoss, LatticeParametersLoss


def get_dataloader(path: str, dataset: str, batch_size: int):
    assert dataset in ["mp-20", "carbon-24", "perov-5"]

    dataset_path = os.path.join(path, dataset)
    if dataset == "mp-20":
        train_set = MP20(dataset_path, "train")
        valid_set = MP20(dataset_path, "val")
        test_set = MP20(dataset_path, "test")
    elif dataset == "carbon-24":
        train_set = Carbon24(dataset_path, "train")
        valid_set = Carbon24(dataset_path, "val")
        test_set = Carbon24(dataset_path, "test")
    elif dataset == "perov-5":
        train_set = Perov5(dataset_path, "train")
        valid_set = Perov5(dataset_path, "val")
        test_set = Perov5(dataset_path, "test")

    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return loader_train, loader_valid, loader_test


def build_model(hparams: Hparams) -> nn.Module:
    assert hparams.model in ["gemsnet", "egnn"]

    if hparams.model == "gemsnet":
        return GemsNetDenoiser(
            hparams.features,
            knn=hparams.knn,
            num_blocks=hparams.layers,
            vector_fields=hparams.vector_fields,
        )
    elif hparams.model == "egnn":
        return EGNNDenoiser(
            features=hparams.features,
            knn=hparams.knn,
            vector_fields=hparams.vector_fields,
            layers=hparams.layers,
            limit_actions=0.5,
            scale_hidden_dim=256,
            scale_reduce_rho="mean",
        )


if __name__ == "__main__":
    import argparse

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description="train denoising model")
    parser.add_argument("--hparams", "-H", default=None, help="json file")
    parser.add_argument("--logs", "-l", default="./runs/denoising")
    parser.add_argument("--dataset", "-D", default="mp-20")
    parser.add_argument("--dataset-path", "-dp", default="./data")
    parser.add_argument("--device", "-d", default="cuda")

    args = parser.parse_args()

    noise_pos = 0.05

    # run name
    tday = datetime.datetime.now()
    run_name = tday.strftime(
        f"training_%Y_%m_%d_%H_%M_%S_{args.dataset}_{random.randint(0,1000):<03d}"
    )
    print("run name:", run_name)

    log_dir = os.path.join(args.logs, run_name)

    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=3)

    # basic setup
    device = args.device

    hparams = Hparams()
    if args.hparams is not None:
        hparams.from_json(args.hparams)

    with open(os.path.join(log_dir, "hparams.json"), "w") as fp:
        json.dump(hparams.dict(), fp, indent=4)

    print("hparams:")
    print(json.dumps(hparams.dict(), indent=4))

    loader_train, loader_valid, loader_test = get_dataloader(
        args.dataset_path, args.dataset, hparams.batch_size
    )

    scaler = LatticeScaler().to(device)
    scaler.fit(loader_train)

    model = build_model(hparams).to(device)

    loss_pos_fn = OptimalTrajLoss(center=True, euclidian=True, distance="l1").to(device)
    loss_lattice_fn = LatticeParametersLoss(lattice_scaler=scaler).to(device)

    opt = optim.Adam(model.parameters(), lr=hparams.lr, betas=(hparams.beta1, 0.999))

    logs = {"batch": [], "loss": [], "loss_pos": [], "loss_lat": []}

    best_val = float("inf")

    batch_idx = 0
    snapshot_idx = 0
    for epoch in tqdm.tqdm(range(hparams.epochs), leave=True, position=0):
        losses, losses_pos, losses_lat = [], [], []

        it = tqdm.tqdm(loader_train, leave=False, position=1)

        for batch in it:
            batch = batch.to(device)

            opt.zero_grad()

            opti_traj = noise_pos * torch.randn_like(batch.pos)
            opti_traj -= scatter_mean(
                opti_traj, batch.batch, dim=0, dim_size=batch.cell.shape[0]
            )[batch.batch]

            if hparams.train_pos:
                x_thild = (batch.pos + opti_traj) % 1.0
            else:
                x_thild = batch.pos

            eye = (
                torch.eye(3, device=device)
                .unsqueeze(0)
                .repeat(batch.cell.shape[0], 1, 1)
            )
            x_prime, x_traj, rho_prime = model.forward(
                eye, x_thild, batch.z, batch.num_atoms
            )

            loss_pos = loss_pos_fn(
                batch.cell, batch.pos, x_thild, x_traj, batch.num_atoms
            )
            loss_lat = loss_lattice_fn(rho_prime, batch.cell)
            if hparams.train_pos:
                loss = loss_pos + loss_lat
            else:
                loss = loss_lat
            loss.backward()

            metrics = get_metrics(
                batch.cell, rho_prime, batch.pos, x_prime, batch.num_atoms
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clipping)
            opt.step()

            # loss_zero = loss_fn(batch.cell, batch.pos, x_thild, batch.num_atoms).item()
            losses.append(loss.item())
            losses_pos.append(loss_pos.item())
            losses_lat.append(loss_lat.item())

            # it.set_description(f"loss: {loss.item():.3f}/{loss_zero:.3f}")
            it.set_description(
                f"loss: {loss.item():.3f} atomic pos={loss_pos.item():.3f} lattice={loss_lat.item():.3f}"
            )

            batch_idx += 1

        losses = torch.tensor(losses).mean().item()
        losses_pos = torch.tensor(losses_pos).mean().item()
        losses_lat = torch.tensor(losses_lat).mean().item()

        writer.add_scalar("train/loss", losses, batch_idx)
        writer.add_scalar("train/loss_pos", losses_pos, batch_idx)
        writer.add_scalar("train/loss_lattice", losses_lat, batch_idx)

        logs["batch"].append(batch_idx)
        logs["loss"].append(losses)
        logs["loss_pos"].append(losses_pos)
        logs["loss_lat"].append(losses_lat)

        pd.DataFrame(logs).set_index("batch").to_csv(os.path.join(log_dir, "loss.csv"))

        with torch.no_grad():
            valid_losses = []
            valid_losses_pos = []
            valid_losses_lat = []

            valid_rho = []
            valid_rho_prime = []
            valid_x = []
            valid_x_prime = []
            valid_num_atoms = []

            for batch in tqdm.tqdm(
                loader_valid, leave=False, position=1, desc="validation"
            ):
                batch = batch.to(device)

                opti_traj = noise_pos * torch.randn_like(batch.pos)
                opti_traj -= scatter_mean(
                    opti_traj, batch.batch, dim=0, dim_size=batch.cell.shape[0]
                )[batch.batch]

                if hparams.train_pos:
                    x_thild = (batch.pos + opti_traj) % 1.0
                else:
                    x_thild = batch.pos

                eye = (
                    torch.eye(3, device=device)
                    .unsqueeze(0)
                    .repeat(batch.cell.shape[0], 1, 1)
                )
                x_prime, x_traj, rho_prime = model.forward(
                    eye, x_thild, batch.z, batch.num_atoms
                )

                loss_pos = loss_pos_fn(
                    batch.cell, batch.pos, x_thild, x_traj, batch.num_atoms
                )
                loss_lat = loss_lattice_fn(rho_prime, batch.cell)
                if hparams.train_pos:
                    loss = loss_pos + loss_lat
                else:
                    loss = loss_lat

                valid_rho.append(batch.cell)
                valid_rho_prime.append(rho_prime)
                valid_x.append(batch.pos)
                valid_x_prime.append(x_prime)
                valid_num_atoms.append(batch.num_atoms)

                valid_losses.append(loss.item())
                valid_losses_pos.append(loss_pos.item())
                valid_losses_lat.append(loss_lat.item())

            losses = torch.tensor(losses).mean().item()
            losses_pos = torch.tensor(losses_pos).mean().item()
            losses_lat = torch.tensor(losses_lat).mean().item()

            if losses < best_val:
                best_val = losses
                torch.save(model.state_dict(), os.path.join(log_dir, "best.pt"))

                snapshot_path = os.path.join(log_dir, f"snapshot")
                os.makedirs(snapshot_path,exist_ok=True)
                save_snapshot(batch, model, os.path.join(snapshot_path, f"{snapshot_idx}.png"))
                snapshot_idx+=1

            writer.add_scalar("valid/loss", losses, batch_idx)
            writer.add_scalar("valid/loss_pos", losses_pos, batch_idx)
            writer.add_scalar("valid/loss_lattice", losses_lat, batch_idx)

            valid_rho = torch.cat(valid_rho, dim=0)
            valid_rho_prime = torch.cat(valid_rho_prime, dim=0)
            valid_x = torch.cat(valid_x, dim=0)
            valid_x_prime = torch.cat(valid_x_prime, dim=0)
            valid_num_atoms = torch.cat(valid_num_atoms, dim=0)

            metrics = get_metrics(
                valid_rho, valid_rho_prime, valid_x, valid_x_prime, valid_num_atoms
            )

            writer.add_scalar("valid/mae_pos", metrics["mae_pos"], batch_idx)
            writer.add_scalar("valid/mae_lengths", metrics["mae_lengths"], batch_idx)
            writer.add_scalar("valid/mae_angles", metrics["mae_angles"], batch_idx)

            # save_snapshot(batch, model, os.path.join(log_dir, "snapshot.png"),noise_pos=noise_pos)

    with torch.no_grad():
        test_losses = []
        test_losses_pos = []
        test_losses_lat = []

        test_rho = []
        test_rho_prime = []
        test_x = []
        test_x_prime = []
        test_num_atoms = []

        for batch in tqdm.tqdm(loader_test, leave=False, position=1, desc="testing"):
            batch = batch.to(device)

            opti_traj = noise_pos * torch.randn_like(batch.pos)
            opti_traj -= scatter_mean(
                opti_traj, batch.batch, dim=0, dim_size=batch.cell.shape[0]
            )[batch.batch]

            if hparams.train_pos:
                x_thild = (batch.pos + opti_traj) % 1.0
            else:
                x_thild = batch.pos

            eye = (
                torch.eye(3, device=device)
                .unsqueeze(0)
                .repeat(batch.cell.shape[0], 1, 1)
            )
            x_prime, x_traj, rho_prime = model.forward(
                eye, x_thild, batch.z, batch.num_atoms
            )

            loss_pos = loss_pos_fn(
                batch.cell, batch.pos, x_thild, x_traj, batch.num_atoms
            )
            loss_lat = loss_lattice_fn(rho_prime, batch.cell)
            if hparams.train_pos:
                loss = loss_pos + loss_lat
            else:
                loss = loss_lat

            test_rho.append(batch.cell)
            test_rho_prime.append(rho_prime)
            test_x.append(batch.pos)
            test_x_prime.append(x_prime)
            test_num_atoms.append(batch.num_atoms)

            test_losses.append(loss.item())
            test_losses_pos.append(loss_pos.item())
            test_losses_lat.append(loss_lat.item())

        losses = torch.tensor(losses).mean().item()
        losses_pos = torch.tensor(losses_pos).mean().item()
        losses_lat = torch.tensor(losses_lat).mean().item()

        test_rho = torch.cat(test_rho, dim=0)
        test_rho_prime = torch.cat(test_rho_prime, dim=0)
        test_x = torch.cat(test_x, dim=0)
        test_x_prime = torch.cat(test_x_prime, dim=0)
        test_num_atoms = torch.cat(test_num_atoms, dim=0)

        metrics = {
            "loss": losses,
            "loss_pos": losses_pos,
            "loss_lattice": losses_lat,
            **get_metrics(
                test_rho, test_rho_prime, test_x, test_x_prime, test_num_atoms
            ),
        }

        with open(os.path.join(log_dir, "metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)

        print("\nmetrics:")
        print(json.dumps(metrics, indent=4))

        writer.add_scalar("test/loss", losses, batch_idx)
        writer.add_scalar("test/loss_pos", losses_pos, batch_idx)
        writer.add_scalar("test/loss_lattice", losses_lat, batch_idx)

        writer.add_scalar("test/mae_pos", metrics["mae_pos"], batch_idx)
        writer.add_scalar("test/mae_lengths", metrics["mae_lengths"], batch_idx)
        writer.add_scalar("test/mae_angles", metrics["mae_angles"], batch_idx)

        writer.add_hparams(hparams.dict(), metrics)

        # save_snapshot(batch, model, os.path.join(log_dir, "snapshot.png"),noise_pos=noise_pos)

    writer.close()
