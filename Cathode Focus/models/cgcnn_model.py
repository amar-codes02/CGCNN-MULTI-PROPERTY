"""
CGCNN model + crystal-graph feature engineering.

This file is a direct, faithful port of the training notebook
(Final_jarvis_EN.ipynb) sections:
  - 5. Crystal Graph Feature Engineering
  - 7. CGCNN Model
  - 11. Predicting New Materials from a CIF File

Keeping this code IDENTICAL to the notebook is important: the checkpoint
(cgcnn_model.pt) was trained with this exact graph construction and this
exact architecture, so any change here (feature order, radius default,
gaussian basis, etc.) would silently break predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Structure

# ---------------------------------------------------------------------------
# Defaults (must match the values used during training in the notebook)
# ---------------------------------------------------------------------------
MAX_NUM_NBR = 12
RADIUS = 8.0
GAUSSIAN_DMIN, GAUSSIAN_DMAX, GAUSSIAN_STEP, GAUSSIAN_VAR = 0, 8, 0.2, 0.2

MODEL_TARGETS = ["band_gap", "formation_energy", "bulk_modulus", "shear_modulus"]
TARGET_UNITS = {
    "band_gap": "eV",
    "formation_energy": "eV/atom",
    "bulk_modulus": "GPa",
    "shear_modulus": "GPa",
}


def classify_band_gap(bg: float) -> str:
    """Same classification rule used in the notebook's EDA section."""
    if bg == 0:
        return "Metal"
    elif bg < 0.5:
        return "Semimetal"
    elif bg < 3.0:
        return "Semiconductor"
    else:
        return "Insulator"


# ---------------------------------------------------------------------------
# Crystal graph feature engineering (section 5 of the notebook)
# ---------------------------------------------------------------------------
def atom_features(structure, max_z=100):
    """One-hot encoding of the atomic number (Z) for each site in the structure."""
    feats = []
    for site in structure:
        z = site.specie.Z
        oh = np.zeros(max_z, dtype=np.float32)
        oh[z - 1] = 1.0
        feats.append(oh)
    return np.array(feats, dtype=np.float32)


def gaussian_expand(distances, dmin=GAUSSIAN_DMIN, dmax=GAUSSIAN_DMAX,
                     step=GAUSSIAN_STEP, var=GAUSSIAN_VAR):
    """Expand scalar distances into a Gaussian radial basis function (RBF) vector."""
    filt = np.arange(dmin, dmax + step, step)
    return np.exp(-((distances[..., None] - filt[None, :]) ** 2) / var ** 2).astype(np.float32)


def build_graph(structure, max_num_nbr=MAX_NUM_NBR, radius=RADIUS):
    """Build a CGCNN graph representation (atom_fea, nbr_fea, nbr_fea_idx) from
    a pymatgen Structure. Identical to the notebook's build_graph()."""
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_dist = [], []
    for nbrs in all_nbrs:
        if len(nbrs) < max_num_nbr:
            idx = [n[2] for n in nbrs] + [0] * (max_num_nbr - len(nbrs))
            dist = [n[1] for n in nbrs] + [radius + 1] * (max_num_nbr - len(nbrs))
        else:
            idx = [n[2] for n in nbrs[:max_num_nbr]]
            dist = [n[1] for n in nbrs[:max_num_nbr]]
        nbr_fea_idx.append(idx)
        nbr_dist.append(dist)

    nbr_fea_idx = np.array(nbr_fea_idx)
    nbr_dist = np.array(nbr_dist)
    nbr_fea = gaussian_expand(nbr_dist)
    atom_fea = atom_features(structure)

    return (torch.tensor(atom_fea),
            torch.tensor(nbr_fea),
            torch.tensor(nbr_fea_idx, dtype=torch.long))


def build_graph_for_viz(structure, max_num_nbr=MAX_NUM_NBR, radius=RADIUS):
    """Same neighbor search as build_graph(), but also returns the actual
    Cartesian coordinates of every neighbor (including periodic images), so
    the graph can be drawn geometrically correctly in 3D. Only used for the
    'Graph Model Visualization' tab -- does not affect predictions."""
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    site_coords = np.array([site.coords for site in structure])
    site_elems = [site.specie.symbol for site in structure]

    edges = []  # (src_idx, dst_coords, dst_elem, distance)
    for i, nbrs in enumerate(all_nbrs):
        take = nbrs[:max_num_nbr] if len(nbrs) > max_num_nbr else nbrs
        for n in take:
            edges.append((i, np.array(n.coords), n.specie.symbol, n.nn_distance))

    return site_coords, site_elems, edges


# ---------------------------------------------------------------------------
# CGCNN architecture (section 7 of the notebook)
# ---------------------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, 2 * self.atom_fea_len)
        ).view(N, M, 2 * self.atom_fea_len)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        return self.softplus2(atom_in_fea + nbr_sumed)


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64,
                 n_conv=3, h_fea_len=128, n_h=1, n_outputs=2, non_negative_idx=None):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.fcs = nn.ModuleList(
            [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
        self.softpluses = nn.ModuleList(
            [nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, n_outputs)
        self.non_negative_idx = non_negative_idx or []
        self.out_softplus = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = torch.stack(
            [atom_fea[idx].mean(dim=0) for idx in crystal_atom_idx])
        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        for fc, sp in zip(self.fcs, self.softpluses):
            crys_fea = sp(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.non_negative_idx:
            out = out.clone()
            out[:, self.non_negative_idx] = self.out_softplus(out[:, self.non_negative_idx])
        return out


# ---------------------------------------------------------------------------
# Checkpoint loading + single-CIF prediction (section 11 of the notebook)
# ---------------------------------------------------------------------------
def load_trained_model(checkpoint_path, map_device=torch.device("cpu")):
    """Load the CGCNN model + target normalization statistics from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=map_device)

    loaded_model = CrystalGraphConvNet(
        orig_atom_fea_len=ckpt["orig_atom_fea_len"],
        nbr_fea_len=ckpt["nbr_fea_len"],
        atom_fea_len=ckpt["atom_fea_len"],
        n_conv=ckpt["n_conv"],
        h_fea_len=ckpt["h_fea_len"],
        n_h=ckpt["n_h"],
        n_outputs=ckpt["n_outputs"],
        non_negative_idx=ckpt.get("non_negative_idx", []),
    ).to(map_device)
    loaded_model.load_state_dict(ckpt["model_state"])
    loaded_model.eval()

    t_mean = ckpt["target_mean"].to(map_device)
    t_std = ckpt["target_std"].to(map_device)
    meta = {
        "max_num_nbr": ckpt.get("max_num_nbr", MAX_NUM_NBR),
        "radius": ckpt.get("radius", RADIUS),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "target_names": ckpt.get("target_names", MODEL_TARGETS),
    }
    return loaded_model, t_mean, t_std, meta


def predict_from_cif(cif_path_or_structure, model_, t_mean, t_std,
                      target_names=MODEL_TARGETS, max_num_nbr=MAX_NUM_NBR,
                      radius=RADIUS, map_device=None):
    """Predict band_gap, formation_energy, bulk_modulus, shear_modulus for a CIF file
    (path) or an already-parsed pymatgen Structure."""
    if map_device is None:
        try:
            map_device = next(model_.parameters()).device
        except Exception:
            map_device = torch.device("cpu")

    if isinstance(cif_path_or_structure, Structure):
        structure = cif_path_or_structure
    else:
        structure = Structure.from_file(cif_path_or_structure)

    atom_fea, nbr_fea, nbr_fea_idx = build_graph(structure, max_num_nbr=max_num_nbr, radius=radius)
    
    atom_fea = atom_fea.to(map_device)
    nbr_fea = nbr_fea.to(map_device)
    nbr_fea_idx = nbr_fea_idx.to(map_device)
    crystal_atom_idx = [torch.arange(atom_fea.shape[0]).to(map_device)]
    
    t_mean = t_mean.to(map_device)
    t_std = t_std.to(map_device)

    model_.eval()
    with torch.no_grad():
        pred_norm = model_(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        pred = pred_norm * t_std + t_mean
        # Enforce non-negative band_gap (eV) and mechanical moduli (GPa)
        pred[:, 0] = torch.clamp(pred[:, 0], min=0.0)
        if pred.shape[1] > 2:
            pred[:, 2] = torch.clamp(pred[:, 2], min=0.0)
            pred[:, 3] = torch.clamp(pred[:, 3], min=0.0)

    pred = pred.cpu().numpy().ravel()

    result = {
        "formula": structure.composition.reduced_formula,
        "n_atoms": len(structure),
    }
    for i, name in enumerate(target_names):
        result[f"{name}_pred"] = float(pred[i])

    if "bulk_modulus_pred" in result and "shear_modulus_pred" in result:
        bm = result["bulk_modulus_pred"]
        sm = result["shear_modulus_pred"]
        result["pugh_ratio_pred"] = float(bm / sm) if sm > 0 else 0.0

    return result, structure


