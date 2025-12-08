import types
from collections.abc import Generator
from typing import Any

import torch
from einops import rearrange
from torch.amp.autocast_mode import autocast

from analysis.data import get_model_input_data, get_sequence_weight_data, to_cpu_dict
from analysis.modal_infrastructure import image, runnable_on_modal, volume
from analysis.utils import progress
from MSA_Pairformer.core import exists
from MSA_Pairformer.custom_typing import Bool, Float
from MSA_Pairformer.dataset import MSA
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.outer_product import OuterProduct, PresoftmaxDifferentialOuterProductMean


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
    shuffled: bool = False,
    shuffled_layers: list[int] | None = None,
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

    if shuffled:
        apply_shuffling_patch(model, shuffled_layers)

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
def calculate_sequence_weights(
    msas: dict[str, MSA], query_biasing: bool = True
) -> dict[str, torch.Tensor]:
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
        msas:
            Dictionary mapping MSA identifiers to MSA objects.
        query_biasing:
            If False, uniform sequence weighting will be used (no query-biased outer product).

    Returns:
        dict[str, torch.Tensor]:
            Dictionary mapping MSA identifiers to their sequence attention weights on
            CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSAPairformer.from_pretrained(device=device)

    if not query_biasing:
        model.turn_off_query_biasing()

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
def calculate_cb_contacts(
    msas: dict[str, MSA], query_biasing: bool = True
) -> dict[str, torch.Tensor]:
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
        msas:
            Dictionary mapping MSA identifiers to MSA objects.
        query_biasing:
            If False, uniform sequence weighting will be used (no query-biased outer product).

    Returns:
        dict[str, torch.Tensor]:
            Dictionary mapping MSA identifiers to their sequence attention weights on
            CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSAPairformer.from_pretrained(device=device)

    if not query_biasing:
        model.turn_off_query_biasing()

    cb_contacts_dict = {}
    with torch.no_grad(), autocast(dtype=torch.bfloat16, device_type=device):
        for msa_id, msa in progress(msas.items(), desc="Running forward passes through the model"):
            cb_contacts = model.predict_cb_contacts(
                **get_model_input_data(msa, torch.device(device)),
            )["predicted_cb_contacts"]

            cb_contacts_dict[msa_id] = cb_contacts.cpu()

    return cb_contacts_dict


def patched_opm_forward(
    self: PresoftmaxDifferentialOuterProductMean,
    msa: Float["b s n d"],
    mask: Bool["b n"] | None = None,
    msa_mask: Bool["b s"] | None = None,
    seq_weights: Float["b s"] | None = None,
    full_mask: Bool["b s n"] | None = None,
    pairwise_mask: Bool["b n n"] | None = None,
) -> Float["b n n dp"]:
    """Patched version of PresoftmaxDifferentialOuterProductMean.forward

    This patch introduces a permutation functionality that will permute the calculated
    sequence weights consistently across layers.

    Except for the section marked with inline comment "SHUFFLE PATCH", the code matches
    git commit 2adb1ff5654f24004285b1006447e212445e6b03.

    We patch at runtime to avoid diverging from the MSA Pairformer HEAD state.
    """
    # Default to full mask if not provided
    if not exists(full_mask):
        full_mask = msa.new_ones(msa.shape[:-1]).to(bool).to(msa.device)  # [b, s, n]
    if not exists(msa_mask):
        msa_mask = msa.new_ones(msa.shape[:-2]).to(bool).to(msa.device)  # [b, s]
    if not exists(mask):
        mask = msa.new_ones((msa.shape[0], msa.shape[2])).to(bool).to(msa.device)
    # Normalize MSA representation
    norm_msa = self.norm(msa)
    # Unsupervised sequence weight learning
    if self.seq_attn:
        # Compute Q, K (Both q1/q2 and k1/k2 are computed from the same projection)
        q = self.q_proj(norm_msa[:, 0])  # [b n (2d)]
        k = self.k_proj(norm_msa)  # [b s n (2d)]
        q_type = q.dtype
        # Split last dimension in half to create q1/q2 and k1/k2
        q = rearrange(q, "... n (two d) -> ... two n d", two=2)
        k = rearrange(k, "... s n (two d) -> ... two s n d", two=2)
        # Normalize q and k
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Compute attention scores
        seq_weights = (
            torch.einsum("... t n d, ... t s n d-> ... t s n", q, k) * self.scaling
        )  # [b 2 s n]
        # Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).to(q_type)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).to(q_type)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        # Average pooling
        norm_factor = (full_mask.sum(dim=-1) + self.eps).unsqueeze(1).expand(-1, 2, -1)  # [b, 2, s]
        seq_weights = seq_weights.masked_fill(~full_mask.unsqueeze(1).expand(-1, 2, -1, -1), 0)
        seq_weights = seq_weights.sum(dim=-1) / norm_factor  # [b 2 s]
        # Compute differential ``
        seq_weights = seq_weights[:, 0, :] - (lambda_full * seq_weights[:, 1, :])  # [b s]
        seq_weights = seq_weights.masked_fill(~msa_mask, -1e9)
        seq_weights = seq_weights.softmax(dim=-1)
        del lambda_1, lambda_2, lambda_full, q, k
    else:
        # Default to uniform sequence weights
        if not exists(seq_weights):
            seq_weights = msa_mask / msa_mask.sum(dim=-1, keepdim=True)  # [b s]

    # ==================================================================
    # SHUFFLE PATCH {{{
    # ==================================================================
    if (
        getattr(self, "should_shuffle", True)
        and hasattr(self, "permuted_indices")
        and self.permuted_indices is not None
    ):
        if self.permuted_indices.device != seq_weights.device:
            self.permuted_indices = self.permuted_indices.to(seq_weights.device)

        seq_weights = seq_weights[:, self.permuted_indices]
    # ==================================================================
    # }}}
    # ==================================================================

    # Create left and right hidden representations and apply mask to padded positions
    expanded_full_mask = full_mask.unsqueeze(-1)
    a = self.to_left_hidden(norm_msa) * expanded_full_mask  # [b s n c]
    b = self.to_right_hidden(norm_msa) * expanded_full_mask  # [b s n c]
    del norm_msa, expanded_full_mask
    # Transpose for efficient matrix multiplication
    a = a.transpose(-2, -3)  # [b n s c]
    b = b.transpose(-2, -3)  # [b n s c]
    # Scale a and b by the square root of the sequence weights
    scaled_seq_weights = (seq_weights + self.eps).sqrt()
    a = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, a)  # [b n s c]
    b = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, b)  # [b n s c]
    if self.chunk_size is not None:
        outer = self._chunk(a, b, self.chunk_size)
    else:
        outer = self._opm(a, b)
    # Mask invalid pairwise positions
    if not exists(pairwise_mask):
        pairwise_mask = to_pairwise_mask(mask)
    outer = torch.einsum("... i j d, ... i j -> ... i j d", outer, pairwise_mask)
    if not self.return_seq_weights:
        del seq_weights
        seq_weights = None
    return outer, seq_weights


def _get_outer_product_instances(
    layers_iterable, layer_indices: list[int] | None = None
) -> Generator[PresoftmaxDifferentialOuterProductMean, None, None]:
    for idx, layer in enumerate(layers_iterable):
        if layer_indices is not None and idx not in layer_indices:
            continue

        for module in layer:
            if not (
                isinstance(module, OuterProduct)
                and isinstance(module.opm, PresoftmaxDifferentialOuterProductMean)
            ):
                continue

            yield module.opm


def apply_shuffling_patch(model: MSAPairformer, layer_indices: list[int] | None = None) -> None:
    patch_count = 0
    layers = model.core_stack.layers
    for opm_instance in _get_outer_product_instances(layers, layer_indices):
        opm_instance.forward = types.MethodType(patched_opm_forward, opm_instance)
        opm_instance.should_shuffle = True
        opm_instance.permuted_indices = None
        patch_count += 1

    model.core_stack._orig_forward = model.core_stack.forward

    def core_stack_forward(self, *args, **kwargs):
        if len(args) > 0:
            msa = args[0]
        else:
            msa = kwargs.get("msa")
        msa_depth = msa.shape[1]

        permuted_indices = torch.randperm(msa_depth).to(msa.device)
        print(permuted_indices)

        for opm_instance in _get_outer_product_instances(self.layers, layer_indices):
            opm_instance.permuted_indices = permuted_indices

        return self._orig_forward(*args, **kwargs)

    model.core_stack.forward = types.MethodType(core_stack_forward, model.core_stack)

    print(f"Successfully patched {patch_count} layers.")
