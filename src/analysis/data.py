from pathlib import Path
from typing import Any

import torch

from MSA_Pairformer.dataset import MSA, aa2tok_d, prepare_msa_masks


def write_processed_msa(msa: MSA, output: Path) -> None:
    """Write an MSA processed by Pairformer to file.

    This functionality is missing from the Pairformer codebase. It's specifically needed
    because the MSAs in OpenProteinSet are filtered by the `MSA_Pairformer.dataset.MSA`
    object, and it is this filtered MSA that we are interested in performing downstream
    calculations on, like calculating a tree.
    """

    # We keep all insertions, since these are meaningful for tree-building and other
    # potential downstream applications.
    unfiltered_seqs, unfiltered_ids = msa.parse_a3m_file(
        keep_insertions=True,
        to_upper=False,
        remove_lowercase_cols=False,
    )

    filtered_seqs = []
    filtered_ids = []
    for idx in msa.select_diverse_ids:
        filtered_seqs.append(unfiltered_seqs[idx])
        filtered_ids.append(unfiltered_ids[idx])






def get_model_input_data(msa: MSA, device: torch.device) -> dict[str, torch.Tensor]:
    msa_tokenized_t = msa.diverse_tokenized_msa
    msa_onehot_t = (
        torch.nn.functional.one_hot(msa_tokenized_t, num_classes=len(aa2tok_d))
        .unsqueeze(0)
        .float()
        .to(device)
    )
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(
        msa.diverse_tokenized_msa.unsqueeze(0)
    )
    mask, msa_mask, full_mask, pairwise_mask = (
        mask.to(device),
        msa_mask.to(device),
        full_mask.to(device),
        pairwise_mask.to(device),
    )

    return dict(
        msa=msa_onehot_t.to(torch.bfloat16),
        mask=mask,
        msa_mask=msa_mask,
        full_mask=full_mask,
        pairwise_mask=pairwise_mask,
    )


def get_sequence_weight_data(model_results: dict[str, Any]) -> torch.Tensor:
    return torch.concat(tuple(model_results["seq_weights_list_d"].values()), dim=0)


def to_cpu_dict(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu_dict(v) for k, v in obj.items()}
    else:
        return obj
