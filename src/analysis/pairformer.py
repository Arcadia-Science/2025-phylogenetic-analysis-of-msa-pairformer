from typing import Any

import torch
from torch.amp.autocast_mode import autocast

from analysis.data import get_model_input_data, to_cpu_dict
from analysis.modal_infrastructure import image, modal_run_settings, volume
from MSA_Pairformer.dataset import MSA
from MSA_Pairformer.model import MSAPairformer


@modal_run_settings(
    app_name="msa-pairformer-inference",
    base_image=image,
    volumes={"/data": volume},
    gpu="T4",
)
def run_inference(
    msa: MSA,
    return_seq_weights: bool = True,
    query_only: bool = True,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSAPairformer.from_pretrained(device=device)
    with torch.no_grad(), autocast(dtype=torch.bfloat16, device_type=device):
        return to_cpu_dict(
            model(
                return_seq_weights=return_seq_weights,
                query_only=query_only,
                **get_model_input_data(msa, torch.device(device)),
            )
        )
