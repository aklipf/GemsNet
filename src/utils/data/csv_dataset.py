from typing import Iterator
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

import h5py
import torch
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import neighbor_list
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import multiprocessing as mp
import warnings
import os
import json


def process_cif(args):
    (cif, warning_queue) = args

    with warnings.catch_warnings(record=True) as ws:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        struct = Structure.from_str(cif, fmt="cif")

        if warning_queue is not None:
            for w in ws:
                warning_queue.put((hash(str(w.message)), w))

    lengths = np.array(struct.lattice.abc, dtype=np.float32)
    angles = np.array(struct.lattice.angles, dtype=np.float32)

    atoms = AseAtomsAdaptor.get_atoms(struct)

    atoms.set_scaled_positions(atoms.get_scaled_positions(wrap=True))

    assert (0 <= atoms.get_scaled_positions()).all() and (
        atoms.get_scaled_positions() < 1
    ).all()

    cell = atoms.cell.array.astype(np.float32)
    z = np.array(struct.atomic_numbers, dtype=np.long)
    pos = struct.frac_coords.astype(np.float32)

    data = {"cell": cell, "lengths": lengths, "angles": angles, "z": z, "pos": pos}

    return data


class CSVDataset(InMemoryDataset, metaclass=ABCMeta):
    def __init__(
        self,
        root: str,
        transform=None,
        pre_filter=None,
        warn: bool = False,
        multithread: bool = True,
        verbose: bool = True,
    ):
        self.warn = warn
        self.multithread = multithread
        self.verbose = verbose

        super().__init__(root, transform, pre_filter=pre_filter)

        self.load()

        if self.verbose:
            print(f"{len(self)} structures loaded!")

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def load_hdf5(self, hdf5_file: str) -> None:
        f = h5py.File(hdf5_file, "r")

        self.material_id = f["material_id"][:]
        self.batch = f["batch"][:]
        self.num_atoms = f["num_atoms"][:]
        self.ptr = f["ptr"][:]
        self.cell = f["cell"][:]
        self.lengths = f["lengths"][:]
        self.angles = f["angles"][:]
        self.pos = f["pos"][:]
        self.z = f["z"][:]

        f.close()

        # basic tensor shape checking
        assert self.material_id.ndim == 1 and self.material_id.dtype == np.int32
        assert self.batch.ndim == 1 and self.batch.dtype == np.int64
        assert self.num_atoms.ndim == 1 and self.num_atoms.dtype == np.int64
        assert self.ptr.ndim == 1 and self.ptr.dtype == np.int64

        assert self.cell.ndim == 3 and self.cell.dtype == np.float32
        assert self.lengths.ndim == 2 and self.lengths.dtype == np.float32
        assert self.angles.ndim == 2 and self.angles.dtype == np.float32

        assert self.pos.ndim == 2 and self.pos.dtype == np.float32
        assert self.z.ndim == 1 and self.z.dtype == np.int64

        # checking size
        n_struct = self.num_atoms.shape[0]
        n_atoms = np.sum(self.num_atoms)

        assert self.material_id.shape == (n_struct,)
        assert self.batch.shape == (n_atoms,)
        assert self.ptr.shape == (n_struct + 1,) and self.ptr[-1] == n_atoms
        assert self.cell.shape == (n_struct, 3, 3)
        assert self.lengths.shape == (n_struct, 3)
        assert self.angles.shape == (n_struct, 3)
        assert self.pos.shape == (n_atoms, 3)
        assert self.z.shape == (n_atoms,)

        self.idx_filtered = torch.arange(self.num_atoms.shape[0], dtype=torch.long)
        if self.pre_filter is not None:
            mask = torch.tensor(
                [
                    self.pre_filter(self.get(idx))
                    for idx in range(self.num_atoms.shape[0])
                ],
                dtype=torch.bool,
            )
            self.idx_filtered = torch.arange(self.num_atoms.shape[0], dtype=torch.long)[
                mask
            ]

    def process_csv(
        self,
        csv_file: str,
        hdf5_file: str,
        loading_description: str = "loading dataset",
    ) -> None:
        df = pd.read_csv(csv_file)

        if self.warn:
            m = mp.Manager()
            warning_queue = m.Queue()
        else:
            warning_queue = None

        iterator = [(row["cif"], warning_queue) for _, row in df.iterrows()]

        if self.multithread:
            if self.verbose:
                results = process_map(
                    process_cif,
                    iterator,
                    desc=loading_description,
                    chunksize=8,
                )
            else:
                with mp.Pool(mp.cpu_count()) as p:
                    results = p.map(process_cif, iterator)
        else:
            results = []

            if self.verbose:
                iterator = tqdm(iterator, desc=loading_description, total=len(df))

            for args in iterator:
                results.append(process_cif(args))

        if self.warn:
            warnings_type = {}
            while not warning_queue.empty():
                key, warning = warning_queue.get()
                if key not in warnings_type:
                    warnings_type[key] = warning

            for w in warnings_type.values():
                warnings.warn_explicit(
                    w.message, category=w.category, filename=w.filename, lineno=w.lineno
                )

        material_id = np.arange(len(results), dtype=np.int32)

        cell = np.stack([struct["cell"] for struct in results], axis=0).astype(
            np.float32
        )
        lengths = np.stack([struct["lengths"] for struct in results], axis=0).astype(
            np.float32
        )
        angles = np.stack([struct["angles"] for struct in results], axis=0).astype(
            np.float32
        )

        batch = np.concatenate(
            [
                np.full_like(struct["z"], fill_value=idx, dtype=np.int64)
                for idx, struct in enumerate(results)
            ],
            axis=0,
        )
        num_atoms = np.array(
            [struct["z"].shape[0] for struct in results], dtype=np.int64
        )
        z = np.concatenate([struct["z"] for struct in results], axis=0).astype(np.int64)
        pos = np.concatenate([struct["pos"] for struct in results], axis=0).astype(
            np.float32
        )

        ptr = np.pad(np.cumsum(num_atoms, axis=0), (1, 0))

        print(f"saving to {hdf5_file}")
        f = h5py.File(hdf5_file, "w")
        f.create_dataset("material_id", material_id.shape, dtype=material_id.dtype)[
            :
        ] = material_id
        f.create_dataset("batch", batch.shape, dtype=batch.dtype)[:] = batch
        f.create_dataset("num_atoms", num_atoms.shape, dtype=num_atoms.dtype)[
            :
        ] = num_atoms
        f.create_dataset("ptr", ptr.shape, dtype=ptr.dtype)[:] = ptr

        f.create_dataset("cell", cell.shape, dtype=cell.dtype)[:, :, :] = cell
        f.create_dataset("lengths", lengths.shape, dtype=lengths.dtype)[:, :] = lengths
        f.create_dataset("angles", angles.shape, dtype=angles.dtype)[:, :] = angles

        f.create_dataset("pos", pos.shape, dtype=pos.dtype)[:] = pos
        f.create_dataset("z", z.shape, dtype=z.dtype)[:] = z

        f.close()

    def len(self) -> int:
        return self.idx_filtered.shape[0]

    def get(self, idx: int) -> Data:
        idx = self.idx_filtered[idx]

        material_id = torch.tensor(self.material_id[idx])
        num_atoms = torch.tensor(self.num_atoms[idx])
        cell = torch.from_numpy(self.cell[idx]).unsqueeze(0)
        z = torch.from_numpy(
            self.z[self.ptr[idx] : self.ptr[idx] + self.num_atoms[idx]]
        )
        pos = torch.from_numpy(
            self.pos[self.ptr[idx] : self.ptr[idx] + self.num_atoms[idx]]
        )

        return Data(
            material_id=material_id,
            z=z,
            pos=pos,
            cell=cell,
            num_atoms=num_atoms,
        )
