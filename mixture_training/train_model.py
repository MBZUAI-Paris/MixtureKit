"""
Kept for existing launch commands; the training script is now MixtureKit.train:

    mixturekit train mixture_training/config_training.yaml
    accelerate launch --config_file mixture_training/deepspeed_config.yaml \
        mixture_training/train_model.py [mixture_training/config_training.yaml]
"""

import os
import sys

# The MixtureKit of this clone, even when another version is installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MixtureKit.train import main  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_training.yaml")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
