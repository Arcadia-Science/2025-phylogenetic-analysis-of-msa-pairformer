from typing import Any

import torch
from torch.amp.autocast_mode import autocast

from analysis.data import get_model_input_data, get_sequence_weight_data, to_cpu_dict
from analysis.modal_infrastructure import image, runnable_on_modal, volume
from analysis.utils import progress
from MSA_Pairformer.dataset import MSA
from MSA_Pairformer.model import MSAPairformer


@runnable_on_modal(
    app_name="msa-pairformer-inference",
    base_image=image,
    volumes={"/data": volume},
    gpu="H100",
)
def run_inference(
    msa: MSA,
    return_seq_weights: bool = True,
    query_only: bool = True,
) -> dict[str, Any]:
    """Run MSA Pairformer inference on a single MSA.

    This function will execute on local hardware if the environmental variable
    `USE_MODAL=0` or is unset, and will execute remotely on Modal if `USE_MODAL=1`
    (requires modal credentials).

    This function is designed for one-off inferences and is not scalable for batch
    processing multiple MSAs. This is because:

        (1) The function instantiates the model internally rather than accepting it as a
            parameter because serializing/deserializing large models between local and
            Modal remote is costly.
        (2) Results are returned immediately for a single MSA rather than processing
            batches because serialization overhead when transferring results from remote
            to local makes large batch operations inefficient.

    For local-only execution, the above considerations don't matter, but the function's
    value is that it runs identically in both local and remote contexts. In other words,
    function is implemented in a way that provides a consistent interface regardless of
    the execution environment (Modal or local).

    Args:
        msa: MSA object containing the alignment data
        return_seq_weights: Whether to return sequence attention weights
        query_only: Whether to only compute features for the query sequence

    Returns:
        dict[str, Any]:
            Dictionary containing model outputs (embeddings, attention weights, etc.) on
            CPU.
    """
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


@runnable_on_modal(
    app_name="msa-pairformer-sequence-weights",
    base_image=image,
    volumes={"/data": volume},
    timeout=36000,
    enable_output=False,
    gpu="H100",
)
def calculate_sequence_weights(msas: dict[str, MSA]) -> dict[str, torch.Tensor]:
    """Calculate sequence bias weights for multiple MSAs.

    This function will execute on local hardware if the environmental variable
    `USE_MODAL=0` or is unset, and will execute remotely on Modal if `USE_MODAL=1`
    (requires modal credentials).

    This function is designed for processing many MSAs for the narrow use case of
    extracting and returning sequence weights (a small amount of data per MSA) rather
    than full model outputs.

    Implemented to provide a consistent interface regardless of the
    execution environment (Modal or local).

    Currently, each forward pass contains a single MSA. Performance could be optimized
    by batching.

    Args:
        msas: Dictionary mapping MSA identifiers to MSA objects

    Returns:
        dict[str, torch.Tensor]:
            Dictionary mapping MSA identifiers to their sequence attention weights on
            CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSAPairformer.from_pretrained(device=device)

    seq_weights_dict = {}
    with torch.no_grad(), autocast(dtype=torch.bfloat16, device_type=device):
        for msa_id, msa in progress(msas.items(), desc="Running forward passes through the model"):
            model_output = model(
                return_seq_weights=True,
                query_only=True,
                **get_model_input_data(msa, torch.device(device)),
            )

            seq_weights_dict[msa_id] = get_sequence_weight_data(model_output).cpu()

    return seq_weights_dict


@runnable_on_modal(
    app_name="msa-pairformer-cb-contacts",
    base_image=image,
    volumes={"/data": volume},
    timeout=36000,
    enable_output=False,
    gpu="H100",
)
def calculate_cb_contacts(msas: dict[str, MSA]) -> dict[str, torch.Tensor]:
    """Calculate beta-carbon contacts for multiple MSAs.

    This function will execute on local hardware if the environmental variable
    `USE_MODAL=0` or is unset, and will execute remotely on Modal if `USE_MODAL=1`
    (requires modal credentials).

    This function is designed for processing many MSAs for the narrow use case of
    extracting and returning beta carbon contact prediction (a small amount of data per
    MSA) rather than full model outputs.

    Implemented to provide a consistent interface regardless of the
    execution environment (Modal or local).

    Currently, each forward pass contains a single MSA. Performance could be optimized
    by batching.

    Args:
        msas: Dictionary mapping MSA identifiers to MSA objects

    Returns:
        dict[str, torch.Tensor]:
            Dictionary mapping MSA identifiers to their sequence attention weights on
            CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSAPairformer.from_pretrained(device=device)

    cb_contacts_dict = {}
    with torch.no_grad(), autocast(dtype=torch.bfloat16, device_type=device):
        for msa_id, msa in progress(msas.items(), desc="Running forward passes through the model"):
            cb_contacts = model.predict_cb_contacts(
                **get_model_input_data(msa, torch.device(device)),
            )["predicted_cb_contacts"]

            cb_contacts_dict[msa_id] = cb_contacts.cpu()

    return cb_contacts_dict
