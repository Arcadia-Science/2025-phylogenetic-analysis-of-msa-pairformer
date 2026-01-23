from typing import Any

import torch
from MSA_Pairformer.dataset import MSA, aa2tok_d, prepare_msa_masks


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
    """Returns the per-layer sequence weight data.

    Returns:
        Tensor: Shape (N, 22), where N is the number of sequences (including query).
    """
    return torch.concat(tuple(model_results["seq_weights_list_d"].values()), dim=0).T


def to_cpu_dict(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu_dict(v) for k, v in obj.items()}
    else:
        return obj
