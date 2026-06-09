"""
Generic utility functions used across the liquidity stress-testing model.

This file should contain only reusable helpers, not banking-specific model logic.
"""

from typing import Any

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """
    Create a reproducible random number generator.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    np.random.Generator
        Reproducible random number generator.
    """
    return np.random.default_rng(seed)


def draw_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int | tuple[int, ...] | None = None,
) -> Any:
    """
    Draw values from a uniform distribution.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    low : float
        Lower bound.
    high : float
        Upper bound.
    size : int, tuple, or None
        Output size.

    Returns
    -------
    Any
        Uniform random draw.
    """
    return rng.uniform(low, high, size=size)


def normalize_to_sum(values: np.ndarray, target_sum: float) -> np.ndarray:
    """
    Rescale a non-negative vector so that it sums to target_sum.

    Parameters
    ----------
    values : np.ndarray
        Input vector.
    target_sum : float
        Desired total sum.

    Returns
    -------
    np.ndarray
        Rescaled vector.
    """
    values = np.asarray(values, dtype=float)

    if np.any(values < 0):
        raise ValueError("Cannot normalize negative values.")

    total = values.sum()

    if total <= 0:
        raise ValueError("Cannot normalize a vector with non-positive sum.")

    return values / total * target_sum


def safe_clip(value: float, low: float, high: float) -> float:
    """
    Clip a scalar value into a closed interval.

    Parameters
    ----------
    value : float
        Input value.
    low : float
        Lower bound.
    high : float
        Upper bound.

    Returns
    -------
    float
        Clipped value.
    """
    return float(np.minimum(np.maximum(value, low), high))