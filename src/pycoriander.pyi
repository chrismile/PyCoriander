from __future__ import annotations
import torch
import pycoriander
import typing

__all__ = [
    "pearson_correlation",
    "spearman_rank_correlation",
    "kendall_rank_correlation",
    "mutual_information_binned",
    "mutual_information_kraskov"
]


def _cleanup() -> None:
    """
    Cleanup correlation estimator data.
    """
def pearson_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Pearson correlation coefficient of the Torch tensors X and Y.
    """
def spearman_rank_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman rank correlation coefficient of the Torch tensors X and Y.
    """
def kendall_rank_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kendall rank correlation coefficient of the Torch tensors X and Y.
    """
def mutual_information_binned(
        X: torch.Tensor, Y: torch.Tensor, num_bins: int,
        X_min: float, X_max: float, Y_min: float, Y_max: float) -> torch.Tensor:
    """
    Computes the mutual information of the Torch tensors X and Y using a binning estimator.
    """
def mutual_information_kraskov(X: torch.Tensor, Y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computes the mutual information of the Torch tensors X and Y using the Kraskov estimator.
    """
