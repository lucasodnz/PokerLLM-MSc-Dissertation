# Excluded assets

The following categories were intentionally excluded from the public release package:

- full datasets and benchmark distributions such as datasets/ and canonical_datasets/ under the original workspace
- trained model weights and adapters such as models/, canonical_models/, and any *.safetensors, *.bin, *.pt, or *.pth payloads
- optimizer state and training checkpoints such as checkpoint-* directories and trainer state files
- raw prediction dumps and bulky evaluation outputs such as canonical_predictions/ and outputs/raw/
- archived or obsolete scripts retained only in the original workspace, including archive/ and archived_noncanonical/ trees
- local caches and virtual environments such as __pycache__/, .cache/, wandb/, .venv/, and temporary logs
- files containing absolute local paths that would not be portable in a public repository

The package also avoids copying the original trained checkpoints and data files that are required for full reproduction of the dissertation results.
