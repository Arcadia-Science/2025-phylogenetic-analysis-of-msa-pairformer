# Reproducing MSA difficulty scores

Due to strict environment requirements and dependency conflicts (reported issue [here](https://github.com/tschuelia/PyPythia/issues/22)), Pythia 2.0 is not compatible with the runtime environment used for this publication. Instead, we performed these calculations in a standalone environment and have git-tracked the resulting difficulty scores, which are loaded from file when the publication is executed.

This document provides full instructions for setting up the Pythia-specific environment and reproducing the MSA difficulty scores.

## Prerequisites

### Install raxml-ng

Pythia requires raxml-ng for its calculations. You can either:

1. *Download from GitHub**: Get the latest release from [raxml-ng releases](https://github.com/amkozlov/raxml-ng/releases)

1. **Install via conda** (in your base environment):
   ```bash
   conda install -c bioconda raxml-ng
   ```

Take note of the path to the raxml-ng binary. If installed via conda in your base environment, it will typically be at `~/miniconda3/bin/raxml-ng` or similar, which you can determine with `which raxml-ng`.

## Setting up the Pythia environment

Create a dedicated conda environment for Pythia:

```bash
conda create -n pythia
conda activate pythia
conda install "python=3.12"
conda install pythiaphylopredictor -c conda-forge
```

## Running the difficulty calculation

With the `pythia` environment activated, run the calculation script from the `msa_difficulty/` directory:

```bash
python run_pythia.py ../data/uniclust30/msas \
    --raxml-path ~/miniconda3/bin/raxml-ng \
    --output ../data/msa_difficulty/msa_difficulty.csv
```

Adjust `--raxml-path` to match your raxml-ng installation location.

The script saves progress periodically, so it can be interrupted and resumed.
