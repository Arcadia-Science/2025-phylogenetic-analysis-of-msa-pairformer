from pathlib import Path

import biotite.structure.io as structio
import numpy as np
import torch
from biotite.structure.atoms import AtomArray
from numpy.typing import NDArray


def load_structure(path: Path | str) -> AtomArray:
    path = Path(path)
    structure = structio.load_structure(path)
    assert isinstance(structure, AtomArray)
    return structure


def calculate_long_range_p_at_l(
    pred_scores: torch.Tensor,
    ground_truth: torch.Tensor,
    min_seq_sep: int = 24,
) -> float:
    L = pred_scores.size(0)
    indices = torch.arange(L)
    seq_sep = indices.unsqueeze(0) - indices.unsqueeze(1)

    mask = (seq_sep >= min_seq_sep) & (seq_sep >= 0)

    flat_pred_scores = pred_scores[mask]
    flat_ground_truth = ground_truth[mask]

    top_pred_indices = torch.argsort(flat_pred_scores, descending=True)[:L]
    correct = flat_ground_truth[top_pred_indices]
    p_at_l = (correct.sum() / correct.size(0)).item()

    return p_at_l


def get_virtual_beta_carbon(
    structure: AtomArray,
    gly_mask: NDArray,
    ca_mask: NDArray,
    n_mask: NDArray,
    c_mask: NDArray,
) -> NDArray:
    gly_ca_coord = structure[gly_mask & ca_mask].coord  # type: ignore
    gly_n_coord = structure[gly_mask & n_mask].coord  # type: ignore
    gly_c_coord = structure[gly_mask & c_mask].coord  # type: ignore

    gly_n_vec = gly_n_coord - gly_ca_coord  # type: ignore
    gly_c_vec = gly_c_coord - gly_ca_coord  # type: ignore

    gly_cb_vec = _rotate_around_axis(gly_n_vec, gly_c_vec, -2 * np.pi / 3)  # type: ignore
    return gly_ca_coord + gly_cb_vec  # type: ignore


def _rotate_around_axis(
    v: NDArray[np.float64], axis: NDArray[np.float64], angle_rad: float
) -> NDArray[np.float64]:
    """Vectorised Rodrigues rotation.

    `v` and `axis` can be shape (3,) or (N, 3); broadcasting rules apply.
    """
    k = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # (k @ v) for batched vectors
    kv_dot = np.sum(k * v, axis=-1, keepdims=True)

    return v * cos_a + np.cross(k, v) * sin_a + k * kv_dot * (1.0 - cos_a)


def split_structure_by_atom_type(
    structure: AtomArray,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the structure by atom type into tensors."""
    ca_mask = structure.atom_name == "CA"
    cb_mask = structure.atom_name == "CB"
    n_mask = structure.atom_name == "N"
    c_mask = structure.atom_name == "C"

    ca_atoms = structure[ca_mask]
    n_atoms = structure[n_mask]
    c_atoms = structure[c_mask]

    # Beta carbons are more complicated, because glycine doesn't have them. For each
    # glycine atom, we'll use a virtual carbon atom within the tetrahedral where the
    # lone proton sidechain is, at a carbon-carbon bond length away.
    cb_coords = np.zeros_like(ca_atoms.coord)

    ca_res_ids = ca_atoms.res_id
    res_id_to_idx = {res_id: idx for idx, res_id in enumerate(ca_res_ids)}

    gly_mask = structure.res_name == "GLY"
    gly_res_ids = np.unique(structure[gly_mask].res_id)
    gly_cb_coords = get_virtual_beta_carbon(structure, gly_mask, ca_mask, n_mask, c_mask)

    non_gly_res_ids = structure[~gly_mask & cb_mask].res_id
    non_gly_cb_coords = structure[cb_mask].coord

    gly_indices = [res_id_to_idx[res_id] for res_id in gly_res_ids]
    non_gly_indices = [res_id_to_idx[res_id] for res_id in non_gly_res_ids]

    cb_coords[gly_indices] = gly_cb_coords
    cb_coords[non_gly_indices] = non_gly_cb_coords

    return (
        torch.tensor(ca_atoms.coord).to(**kwargs),
        torch.tensor(cb_coords).to(**kwargs),
        torch.tensor(n_atoms.coord).to(**kwargs),
        torch.tensor(c_atoms.coord).to(**kwargs),
    )


if __name__ == "__main__":
    path = Path("data/uniclust30/pdbs/A0A0F0F1E5.pdb")
    split_structure_by_atom_type(load_structure(path))
