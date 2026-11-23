"""Helpers shared by the evaluation scripts."""
import os

import torch

from models import get_generator


def get_device():
    """Return the CUDA device when one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(model_name, ckpt):
    """Return the path of a generator checkpoint.

    ``ckpt`` is either a file name inside ``epochs/<model_name>/`` (where train.py
    saves checkpoints) or a path to a ``.pth`` file.
    """
    candidate = os.path.join("epochs", model_name, ckpt)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(ckpt):
        return ckpt
    raise FileNotFoundError(
        "Checkpoint not found: tried '%s' and '%s'" % (candidate, ckpt)
    )


def load_generator(model_name, upscale_factor, ckpt, device):
    """Build the named generator, load the weights in ``ckpt`` and move it to ``device``."""
    model = get_generator(model_name)(1, upscale_factor)
    state_dict = torch.load(
        resolve_checkpoint(model_name, ckpt), map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    return model.to(device)
