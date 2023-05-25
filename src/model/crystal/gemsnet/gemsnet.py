"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor

from .data_utils import get_pbc_distances, radius_graph_pbc

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.interaction_block import (
    InteractionBlockTripletsOnly,
)
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer
from .utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)
from .layers.grad.vector_fields import make_vector_fields

from src.utils.geometry import Geometry
from crystallographic_graph import sparse_meshgrid


class GemsNetT(torch.nn.Module):
    """
    GemsNet

    Parameters
    ----------
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        aggregate: bool
            Whether to aggregated node outputs
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        latent_dim: int,
        knn:int=32,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 128,
        emb_size_edge: int = 128,
        emb_size_trip: int = 32,  # 64
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        cutoff: float = 6.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        vector_fields: dict = {
            "type": "grad",
            "edges": [],
            "triplets": ["n_ij", "n_ik", "angle"],
            "normalize": True,
        },
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        scale_file: Optional[str] = None,
    ):
        super().__init__()
        assert num_blocks > 0
        assert knn > 0
        self.num_blocks = num_blocks

        self.knn = knn

        self.cutoff = cutoff
        # assert self.cutoff <= 6 or otf_graph

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf_out = Dense(
            num_spherical,
            emb_size_cbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        self.vector_fields = make_vector_fields(vector_fields)

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.atom_latent_emb = nn.Linear(emb_size_atom + latent_dim, emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        out_blocks = []
        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i+1}",
                )
            )

        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    nHidden=num_atom,
                    num_targets=1,
                    num_vector_fields=self.vector_fields.triplets_dim,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=True,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def forward(
        self,
        cell: torch.FloatTensor,
        x: torch.FloatTensor,
        z: torch.LongTensor,
        num_atoms: torch.LongTensor,
    ):
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """

        geometry = Geometry(
            cell,
            num_atoms,
            x,
            knn=self.knn,
            triplets=False,
            symetric=True,
            compute_reverse_idx=True,
        )

        batch = geometry.batch
        idx_s = geometry.edges.src
        idx_t = geometry.edges.dst
        D_st = geometry.edges_r_ij
        V_st = -geometry.edges_v_ij / geometry.edges_r_ij[:, None]
        id_swap = geometry.edges.reverse_idx

        num_edges = scatter_add(
            torch.ones_like(idx_s), idx_s, dim=0, dim_size=x.shape[0]
        )
        i_triplets, j_triplets = sparse_meshgrid(num_edges)

        mask = i_triplets != j_triplets
        id3_ba = i_triplets[mask]
        id3_ca = j_triplets[mask]

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, sbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(z)
        # Merge z and atom embedding

        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, sbf3)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        cbf_out = self.mlp_cbf_out(sbf3)

        E_t, F_st, S_st = self.out_blocks[0](
            h, m, rbf_out, cbf_out, idx_t, id3_ba, id3_ca
        )
        # (nAtoms, num_targets), (nEdges, num_targets)

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F, S = self.out_blocks[i + 1](
                h, m, rbf_out, cbf_out, idx_t, id3_ba, id3_ca
            )
            # (nAtoms, num_targets), (nEdges, num_targets)
            E_t += E
            F_st += F
            S_st += S

        nMolecules = torch.max(batch) + 1

        # ========================== ENERGY ==========================
        # always use mean aggregation
        E_t = scatter(
            E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
        )  # (nMolecules, num_targets)
        # if predict forces, there should be only 1 energy

        # ========================== FORCES ==========================
        # map forces in edge directions
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        # (nEdges, num_targets, 3)
        F_t = scatter(
            F_st_vec,
            idx_t,
            dim=0,
            dim_size=num_atoms.sum(),
            reduce="add",
        )  # (nAtoms, num_targets, 3)
        F_t = F_t.squeeze(1)  # (nAtoms, 3)

        F_t_in = torch.bmm(cell.inverse()[batch].detach(), F_t.unsqueeze(2)).squeeze(2)

        # ========================== STRESS ==========================
        batch_triplets = batch[idx_s[id3_ba]]

        # e_ij = x[idx_s[id3_ba]] - x[idx_t[id3_ba]] + offset[id3_ba]
        # e_ik = x[idx_s[id3_ca]] - x[idx_t[id3_ca]] + offset[id3_ca]
        e_ij = geometry.edges_e_ij[id3_ba]
        e_ik = geometry.edges_e_ij[id3_ca]

        """
        v_ij = torch.bmm(cell[batch_triplets], -e_ij.unsqueeze(2)).squeeze(2)
        v_ik = torch.bmm(cell[batch_triplets], -e_ik.unsqueeze(2)).squeeze(2)

        print(v_ij / D_st[id3_ba, None])
        print(V_st[id3_ba])

        print((V_st[id3_ba] - v_ij / D_st[id3_ba, None]).abs().max())
        print((V_st[id3_ca] - v_ik / D_st[id3_ca, None]).abs().max())
        """

        vector_fields = self.vector_fields(cell, batch_triplets, e_ij, e_ik)
        filter_nan = ~(
            (vector_fields != vector_fields).view(vector_fields.shape[0], -1).any(dim=1)
        )

        batch_triplets = batch_triplets[filter_nan]
        fields = (S_st[filter_nan, :, None, None] * vector_fields[filter_nan]).sum(
            dim=1
        )
        I = torch.eye(3, 3, device=cell.device)[None]
        S_t = I + scatter(
            fields, batch_triplets, dim=0, dim_size=cell.shape[0], reduce="mean"
        )  # 1st order approx of matrix exp
        cell_prime = torch.bmm(S_t, cell)

        return (
            (x + F_t_in) % 1.0,
            F_t_in,
            h,
            cell_prime,
            S_t,
        )  # (nMolecules, num_targets), (nAtoms, 3)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
