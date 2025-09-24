from pathlib import Path
from typing import Any

import modal
import torch

app = modal.App("msa-pairformer-inference")

MSA_PATH = "data/response_regulators/PF00072.small_1NXS.a3m"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "torch>=2.0.0",
            "einx",
            "einops",
            "numpy",
            "biopython",
            "huggingface-hub",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "pandas",
            "msa-pairformer",
        ]
    )
    .add_local_python_source("analysis")
    .add_local_file(
        Path(__file__).parent.parent.parent / MSA_PATH,
        f"/root/{MSA_PATH}",
    )
)


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    else:
        return obj


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
)
def run_inference(
    msa_file_path: str,
    diverse_select_method: str = "none",
    return_seq_weights: bool = True,
    query_only: bool = True,
    device: str = "cuda",
    weights_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run MSA PairFormer inference on an MSA file.

    Args:
        msa_file_path: Path to MSA file (A3M format)
        diverse_select_method: Method for sequence selection ("hhfilter", "greedy", "none")
        return_seq_weights: Whether to return sequence attention weights
        query_only: Whether to return only query sequence representations
        device: Device to run inference on
        weights_dir: Directory to cache model weights

    Returns:
        Dictionary containing inference results and MSA metadata
    """
    import sys

    sys.path.append("/root/MSA_Pairformer")

    from MSA_Pairformer.dataset import MSA
    from MSA_Pairformer.model import MSAPairformer

    device = "cuda"

    # Load model
    model = MSAPairformer.from_pretrained(device=device, weights_dir=weights_dir)
    model.eval()

    # Load and process MSA
    print(msa_file_path)
    print(Path(msa_file_path).exists())
    msa = MSA(
        msa_file_path=msa_file_path,
        diverse_select_method=diverse_select_method,
    )

    # Use the get_model_input_data function from src/analysis/data.py
    from analysis.data import get_model_input_data

    input_data = get_model_input_data(msa, device)

    # Run inference
    with (
        torch.no_grad(),
        torch.amp.autocast(dtype=torch.bfloat16, device_type=device),
    ):
        results = model(
            return_seq_weights=return_seq_weights,
            query_only=query_only,
            **input_data,
        )

    return to_cpu(results)


@app.local_entrypoint()
def main(msa_file_path: str):
    """Example usage for local testing."""
    print(f"Running inference on {msa_file_path}")

    results = run_inference.remote(
        msa_file_path=msa_file_path,
        return_seq_weights=True,
        query_only=True,
    )

    print(results)
