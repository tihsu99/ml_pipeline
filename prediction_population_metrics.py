"""Population distances and sample-based coverage diagnostics (NumPy only)."""
from __future__ import annotations

import numpy as np


def probability_weights(values, weights=None):
    values = np.asarray(values, dtype=float)
    weights = np.ones(values.size) if weights is None else np.asarray(weights, dtype=float)
    if values.ndim != 1 or not values.size or weights.shape != values.shape:
        raise ValueError("Values and weights must be nonempty one-dimensional arrays of equal length")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("Values and weights must be finite")
    if np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("Probability distances require nonnegative weights with positive finite sum")
    return values, weights / weights.sum()


def wasserstein1(target, prediction, target_weight=None, prediction_weight=None, period=None):
    """Exact empirical W1; circular geodesic cost when period is specified.

    Integrate the empirical CDF difference over all support intervals. On a
    circle, subtract its interval-length-weighted median before integration:
    https://arxiv.org/abs/0906.5499. No histogram, tail clipping or event matching.
    """
    target, tw = probability_weights(target, target_weight)
    prediction, pw = probability_weights(prediction, prediction_weight)
    if period is not None:
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be positive and finite")
        target, prediction = target % period, prediction % period
    positions = np.concatenate((target, prediction))
    increments = np.concatenate((tw, -pw))
    if period is not None:
        positions = np.concatenate(([0.0], positions, [period]))
        increments = np.concatenate(([0.0], increments, [0.0]))
    order = np.argsort(positions, kind="stable")
    widths = np.diff(positions[order])
    cdf_difference = np.cumsum(increments[order])[:-1]
    center = 0.0
    if period is not None:
        by_level = np.argsort(cdf_difference)
        median_index = np.searchsorted(np.cumsum(widths[by_level]), period / 2)
        center = cdf_difference[by_level[median_index]]
    return float(np.sum(widths * np.abs(cdf_difference - center)))


def tarp_coverage(samples, truth, references, alpha, weights=None, period=None):
    """One-dimensional TARP with explicit random references and empirical coverage.

    samples: (events, draws), truth/references/weights: (events,).
    References must be generated independently of the truth of each event.
    Circular distance is used when period is given. The caller owns reference
    generation and repeated-reference aggregation. This is not a p-value.
    Definition: https://proceedings.mlr.press/v202/lemos23a.html
    """
    truth, weights = probability_weights(truth, weights)
    samples = np.asarray(samples, dtype=float)
    references = np.asarray(references, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != truth.size or samples.shape[1] < 2:
        raise ValueError("TARP requires at least two posterior draws for each event")
    if references.shape != truth.shape or not np.all(np.isfinite(references)) or not np.all(np.isfinite(samples)):
        raise ValueError("TARP samples and event-specific references must be finite and correctly shaped")
    if alpha.ndim != 1 or not np.all(np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > 1)) or np.any(np.diff(alpha) <= 0):
        raise ValueError("alpha must be an increasing one-dimensional grid in [0, 1]")
    sample_distance = samples - references[:, None]
    truth_distance = truth - references
    if period is not None:
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be positive and finite")
        sample_distance = (sample_distance + period / 2) % period - period / 2
        truth_distance = (truth_distance + period / 2) % period - period / 2
    ranks = np.mean(np.abs(sample_distance) < np.abs(truth_distance[:, None]), axis=1)
    # Strict threshold matches histogram upper edges; alpha=1 includes rank=1.
    coverage = np.array([weights[ranks < level].sum() if level < 1 else 1.0 for level in alpha])
    return coverage


def coverage_area(alpha, coverage):
    """Integral of |coverage-alpha| for the piecewise-linear plotted curve."""
    alpha, difference = np.asarray(alpha), np.asarray(coverage) - np.asarray(alpha)
    left, right = difference[:-1], difference[1:]
    widths = np.diff(alpha)
    area = widths * (np.abs(left) + np.abs(right)) / 2
    crossed = left * right < 0
    area[crossed] = widths[crossed] * (left[crossed] ** 2 + right[crossed] ** 2) / (2 * (np.abs(left[crossed]) + np.abs(right[crossed])))
    return float(area.sum())


def target_normalization(target, min_std=1e-12):
    """Fit a single standardization on target alone; return kept dimensions."""
    target = np.asarray(target, dtype=float)
    if target.ndim != 2 or len(target) < 2 or not np.isfinite(target).all():
        raise ValueError("Target must be a finite [event, variable] matrix with at least two rows")
    if not np.isfinite(min_std) or min_std < 0:
        raise ValueError("min_target_std must be finite and nonnegative")
    mean, std = target.mean(axis=0), target.std(axis=0)
    keep = std > min_std
    if not np.any(keep):
        raise ValueError("All joint variables have negligible target variance")
    return mean, std, keep


def empirical_copula(values):
    """Independent average-tie ranks: (rank - 0.5) / N for every column."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Copula input must be a nonempty finite matrix")
    result = np.empty_like(values)
    for column in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, column], return_inverse=True, return_counts=True)
        # Midpoints of tie blocks equal average ranks minus one half.
        result[:, column] = (np.cumsum(counts) - counts / 2)[inverse] / len(values)
    return result


def random_projections(dimensions, count=1000, seed=42):
    """Normalized isotropic Gaussian vectors are uniform on the unit sphere."""
    if dimensions < 1 or count < 1:
        raise ValueError("Projection dimension and count must be positive")
    directions = np.random.default_rng(seed).normal(size=(count, dimensions))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def sliced_wasserstein(target, prediction, directions, batch_size=16):
    """Mean exact projected W1 for equal-count, uniformly weighted populations.

    Batch projections to avoid allocating events x all-projections matrices.
    Unlike a per-event residual this is invariant to row permutations.
    """
    target, prediction, directions = map(np.asarray, (target, prediction, directions))
    if (target.ndim != 2 or target.shape != prediction.shape or not len(target)
            or directions.ndim != 2 or directions.shape[1] != target.shape[1]
            or not len(directions) or batch_size < 1
            or not all(np.isfinite(array).all() for array in (target, prediction, directions))):
        raise ValueError("SWD requires equal-size finite matrices and compatible projection vectors")
    total = 0.0
    for start in range(0, len(directions), batch_size):
        vectors = directions[start:start+batch_size].T
        target_projection = np.sort(target @ vectors, axis=0)
        prediction_projection = np.sort(prediction @ vectors, axis=0)
        total += np.mean(np.abs(target_projection-prediction_projection), axis=0).sum()
    return float(total / len(directions))
