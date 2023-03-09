from __future__ import annotations
import torch
import pycoriander
import typing

__all__ = [
    "mutual_information_kraskov"
]


def _cleanup() -> None:
    """
    Cleanup correlation estimator data.
    """
def mutual_information_kraskov(X: torch.Tensor, Y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computes the mutual information of the Torch tensors X and Y using the Kraskov estimator.
    """
