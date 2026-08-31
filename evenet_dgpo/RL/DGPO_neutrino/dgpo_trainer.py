"""
Standalone DGPO fine-tuning loop for neutrino diffusion (Step 5 of the neutrino RL plan).

Uses the same ``EveNetModel`` backbone and Parquet → Ray → ``iter_torch_batches`` path as
``evenet/train.py``, but replaces Lightning with a plain PyTorch optimizer step and the DGPO
objective from ``dgpo_utils.py``.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from evenet.control.global_config import global_config
from evenet.shared import make_process_fn, prepare_datasets
from evenet.utilities.diffusion_sampler import (
    DDIMSampler,
    get_logsnr_alpha_sigma,
)

from RL.DGPO_neutrino.dgpo_utils import (
    _dgpo_cfg_get,
    build_dgpo_loss,
    compute_per_event_advantage,
    predict_x0_normalized_from_velocity_diffusion,
    repeat_batch_for_candidates,
)
from RL.DGPO_neutrino.model_utils import (
    apply_component_freezes,
    freeze_reference_model,
    load_evenet_model_for_dgpo,
    make_ema,
    make_ema_rollout,
    make_reference_model,
    parse_dgpo_resume_from_checkpoint,
    save_lightning_compatible_checkpoint,
)
from RL.DGPO_neutrino.latent_constraint.dgpo_constraint import (
    LatentSWDState,
    broadcast_latent_swd_state,
    sync_projection_constraint_C_across_ranks,
    compute_latent_swd_constraint,
    init_latent_swd_state,
    _validate_dgpo_constraint_resume,
)
from RL.DGPO_neutrino.projection_cpo import (
    ProjectionConstraintConfig,
    assign_params_,
    assign_params_from_theta_old_delta_,
    compute_cpo_adamw_final_update,
    compute_projection_lambda,
    compute_projection_lambda_from_violation,
    flatten_adam_preconditioned_direction,
    flatten_param_delta,
    flatten_param_grads,
    projection_stratified_t_grid,
    resolve_projection_constraint_config,
    snapshot_params,
    trainable_params_all_finite,
)
from RL.DGPO_neutrino.rewards import (
    CalibrationMagnitudeReward,
    ComponentNormalizedTruthDistanceReward,
    RewardAggregator,
    cartesian_to_log_pt_eta_phi,
    get_event_valid_mask,
    log_pt_eta_phi_to_cartesian,
)
from RL.DGPO_neutrino.domains.ztautau import build_feature_space_scales

_log = logging.getLogger(__name__)

# Highest W&B ``step=`` committed so far (rank-0 logging only).  Training logs use ``step=global_step``.
# Epoch-end panels (val, train_dist) reuse the last training step of the epoch
# but chart x-axis is ``epoch`` via ``wandb.define_metric``.
_wandb_committed_step: int = -1

_GRAD_CLIP_NORM = 1.0
# Projection-constraint payload key in DGPO checkpoints. The legacy key (from the
# removed discriminator-Wasserstein era) is still read on resume for older runs.
_DGPO_CONSTRAINT_CKPT_KEY = "dgpo_projection_constraint_state"
_DGPO_CONSTRAINT_CKPT_KEY_LEGACY = "dgpo_discriminator_wasserstein_state"
ProjectionConstraintState = LatentSWDState
_RL_DISABLED_MSG = (
    "RL pipeline disabled. Use evenet/train.py for original foundation training."
)


def _assert_rl_enabled() -> None:
    """Exit cleanly when ``rl.enabled`` is false (foundation training stays in evenet/train.py)."""
    rl = getattr(global_config, "rl", None)
    if rl is None or not bool(getattr(rl, "enabled", False)):
        raise SystemExit(_RL_DISABLED_MSG)



def _dgpo_rollout_ema_decay(global_step: int) -> float:
    """Effective decay for rollout EMA update: ``min(max, ramp * step)`` (Flow GRPO ``ema_ref`` style)."""
    dg = global_config.dgpo
    decay_max = float(dg.get("ema_rollout_decay_max", 0.3))
    decay_ramp = float(dg.get("ema_rollout_decay_ramp", 0.001))
    return min(decay_max, decay_ramp * float(global_step))


class _DGPODDPForward(nn.Module):
    """Routes DDP ``forward`` to ``EveNetModel.predict_diffusion_vector`` (neutrino mode)."""

    def __init__(self, eve_net: nn.Module) -> None:
        super().__init__()
        self.eve_net = eve_net

    def forward(
        self,
        noise_x: Tensor,
        cond_x: dict[str, Any],
        time: Tensor,
        noise_mask: Tensor,
    ) -> Tensor:
        return self.eve_net.predict_diffusion_vector(
            noise_x=noise_x,
            cond_x=cond_x,
            time=time,
            mode="neutrino",
            noise_mask=noise_mask,
        )


def _unwrap_core_evenet(model: nn.Module) -> nn.Module:
    """Unwrap ``DDP(_DGPODDPForward(eve))`` to the underlying ``EveNetModel``."""
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "eve_net") and isinstance(getattr(m, "eve_net"), nn.Module):
        return m.eve_net
    return m


def _next_batch_synced(
    iterator: Any,
    *,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, Any] | None, bool]:
    """Pull the next batch from a per-rank Ray DataIterator with cross-rank termination sync.

    Each rank fetches its own batch from its own shard. To keep DDP collectives in lock-step,
    we all-reduce a "has-more" flag with ``MIN``: the loop terminates as soon as **any** rank
    runs out of data. This may drop a few batches from longer shards but prevents NCCL hangs.

    Returns ``(batch, all_have)``.  ``batch`` is ``None`` when the local shard is exhausted.
    """
    try:
        batch = next(iterator)
        local_has = 1
    except StopIteration:
        batch = None
        local_has = 0

    if world_size > 1:
        flag = torch.tensor([local_has], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_have = bool(flag.item() > 0)
    else:
        all_have = local_has > 0
    return batch, all_have


def _truth_generation_cartesian() -> bool:
    tg = global_config.options.Training.Components.TruthGeneration
    return bool(getattr(tg, "cartesian", False))


def _histogram_jsd(truth_counts: np.ndarray, pred_counts: np.ndarray) -> float:
    """Jensen-Shannon distance between two count histograms, or ``nan`` when undefined."""
    truth = np.asarray(truth_counts, dtype=np.float64)
    pred = np.asarray(pred_counts, dtype=np.float64)
    truth_sum = float(truth.sum())
    pred_sum = float(pred.sum())
    if not np.isfinite(truth_sum) or not np.isfinite(pred_sum):
        return float("nan")
    if truth_sum <= 0.0 or pred_sum <= 0.0:
        return float("nan")
    return float(jensenshannon(truth / truth_sum, pred / pred_sum))


def _array_histogram_jsd(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
    num_bins: int = 40,
) -> float:
    """Histogram JSD from raw truth/pred arrays using explicit or data-driven bin edges."""
    truth = np.asarray(truth_values, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    truth = truth[np.isfinite(truth)]
    pred = pred[np.isfinite(pred)]
    if truth.size == 0 or pred.size == 0:
        return float("nan")

    edges = None if bin_edges is None else np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if edges is None or edges.size < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        merged = np.concatenate((truth, pred), axis=0)
        lo, hi = [float(x) for x in np.nanpercentile(merged, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi):
            return float("nan")
        if hi <= lo:
            center = float(np.nanmean(merged))
            span = max(abs(center) * 0.1, 1.0)
            lo, hi = center - span, center + span
        pad = max(0.05 * (hi - lo), 1e-6)
        edges = np.linspace(lo - pad, hi + pad, max(2, int(num_bins)) + 1)

    truth_counts, _ = np.histogram(truth, bins=edges)
    pred_counts, _ = np.histogram(pred, bins=edges)
    return _histogram_jsd(truth_counts, pred_counts)


def _truth_pred_scalar_metrics(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
) -> dict[str, float]:
    """Scalar summaries for truth-vs-pred arrays used by 2D monitoring panels."""
    truth = np.asarray(truth_values, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    n = min(truth.size, pred.size)
    if n == 0:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "pearson_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }
    truth = truth[:n]
    pred = pred[:n]
    keep = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[keep]
    pred = pred[keep]
    if truth.size == 0:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "pearson_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }

    delta = pred - truth
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    bias = float(np.mean(delta))
    if truth.size >= 2:
        truth_mean = float(np.mean(truth))
        pred_mean = float(np.mean(pred))
        truth_centered = truth - truth_mean
        pred_centered = pred - pred_mean
        denom = float(np.sqrt(np.sum(truth_centered * truth_centered) * np.sum(pred_centered * pred_centered)))
        pearson_r = float(np.sum(truth_centered * pred_centered) / denom) if denom > 0.0 else float("nan")
        truth_var = float(np.sum(truth_centered * truth_centered))
        slope = float(np.sum(truth_centered * pred_centered) / truth_var) if truth_var > 0.0 else float("nan")
        intercept = pred_mean - slope * truth_mean if math.isfinite(slope) else float("nan")
    else:
        pearson_r = float("nan")
        slope = float("nan")
        intercept = float("nan")
    return {
        "count": float(truth.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "pearson_r": pearson_r,
        "slope": slope,
        "intercept": intercept,
    }


@torch.no_grad()
def _kin_hist_candidate_indices_per_event(
    rewards_kb: Tensor,
    candidates_kb: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
) -> Tensor:
    """Per-event candidate index ``(B,)`` for ``train_dist/*`` and ``val_neutrino/*`` histograms.

    Uses the scalar ``rewards_kb`` argmax (component-normalized truth-distance reward).
    """
    return rewards_kb.argmax(dim=0)


@torch.no_grad()
def compute_reward_mean_gap(rewards_kb: Tensor, valid_b: Tensor) -> float:
    """Mean over valid events of (mean reward above median − mean reward below median) along ``K``."""
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0 or rewards_kb.shape[0] < 2:
        return float("nan")
    r = rewards_kb[:, vb]
    med = r.median(dim=0).values.unsqueeze(0)
    good_m = r > med
    bad_m = r < med
    good_den = good_m.sum(dim=0).clamp(min=1).to(r.dtype)
    bad_den = bad_m.sum(dim=0).clamp(min=1).to(r.dtype)
    good_mean = (r * good_m.to(r.dtype)).sum(dim=0) / good_den
    bad_mean = (r * bad_m.to(r.dtype)).sum(dim=0) / bad_den
    return float((good_mean - bad_mean).mean().cpu())


@torch.no_grad()
def compute_reward_advantage_pos_neg_gap(
    rewards_kb: Tensor,
    advantages_kb: Tensor,
    valid_b: Tensor,
) -> float:
    """Mean reward where advantage > 0 minus mean reward where advantage < 0 (valid events only)."""
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        return float("nan")
    r = rewards_kb[:, vb]
    a = advantages_kb[:, vb]
    pos = a > 0
    neg = a < 0
    if not pos.any() or not neg.any():
        return float("nan")
    pos_m = r[pos].mean()
    neg_m = r[neg].mean()
    return float((pos_m - neg_m).cpu())


def _grad_norm_pre_clip_and_clip_active(
    model: nn.Module,
    max_norm: float,
) -> tuple[float, float]:
    """Return (total L2 grad norm before clipping, 1.0 if norm exceeded ``max_norm`` else 0.0).

    ``torch.nn.utils.clip_grad_norm_`` returns the norm **before** scaling; clipping applies when
    that norm exceeds ``max_norm``.
    """
    gn = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm))
    )
    active = 1.0 if gn > float(max_norm) + 1e-12 else 0.0
    return gn, active


def _dgpo_nonfinite_fraction(t: Tensor) -> float:
    """Fraction of elements in ``t`` that are not finite."""
    if t.numel() == 0:
        return 0.0
    return float((~torch.isfinite(t)).float().mean().detach().cpu())


def _dgpo_zero_nonfinite(t: Tensor) -> Tensor:
    """Replace non-finite entries with zero."""
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _dgpo_sanitize_rollout_rewards(
    rewards: Tensor,
    reward_breakdown: dict[str, Tensor],
    *,
    global_step: int,
) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
    """Zero non-finite rewards and emit diagnostic fractions."""
    diag: dict[str, float] = {}
    rew_frac = _dgpo_nonfinite_fraction(rewards)
    if rew_frac > 0.0:
        diag["train/reward_nonfinite_fraction"] = rew_frac
        for name, rb in reward_breakdown.items():
            src_frac = _dgpo_nonfinite_fraction(rb)
            if src_frac > 0.0:
                diag[f"train/reward_nonfinite_fraction/{name}"] = src_frac
                _log.warning(
                    "[DGPO] non-finite reward source %s (frac=%.4g) at global_step=%s.",
                    name,
                    src_frac,
                    global_step,
                )
                reward_breakdown[name] = _dgpo_zero_nonfinite(rb)
        _log.warning(
            "[DGPO] non-finite total rewards (frac=%.4g) at global_step=%s; zeroing.",
            rew_frac,
            global_step,
        )
        rewards = _dgpo_zero_nonfinite(rewards)
    return rewards, reward_breakdown, diag


def _dgpo_assert_train_step_invariants(
    L_ref: Tensor,
    advantages: Tensor,
    rewards: Tensor,
) -> None:
    """Cheap per-step guards for DGPO training."""
    assert not L_ref.requires_grad, "[DGPO CHECK] L_ref must not require grad."
    assert not advantages.requires_grad, "[DGPO CHECK] advantages must not require grad."
    if not torch.isfinite(rewards).all():
        bad = int((~torch.isfinite(rewards)).sum().item())
        raise AssertionError(
            f"[DGPO CHECK] rewards must be finite after sanitization ({bad} bad entries)."
        )
    if not torch.isfinite(L_ref).all():
        bad = int((~torch.isfinite(L_ref)).sum().item())
        raise AssertionError(
            f"[DGPO CHECK] L_ref must be finite ({bad} bad entries); check model weights / DDIM."
        )
    if not torch.isfinite(advantages).all():
        bad = int((~torch.isfinite(advantages)).sum().item())
        raise AssertionError(
            f"[DGPO CHECK] advantages must be finite ({bad} bad entries)."
        )


_REWARD_DIST_OVERLAY_BINS = 40
_REL_PT_DIST_BINS = 50


def _reward_dist_overlaid_figure(
    best: np.ndarray,
    worst: np.ndarray,
    med: np.ndarray,
) -> Any:
    """Three overlapped 1D histograms (density), EveNet validation style, as ``wandb.Image``."""
    import wandb

    stacked = np.concatenate([best, worst, med])
    lo = float(np.min(stacked))
    hi = float(np.max(stacked))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    elif hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    else:
        span = hi - lo
        pad = max(1e-6 * span, 1e-9)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REWARD_DIST_OVERLAY_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    labels = ("best (max per event)", "worst (min per event)", "median along K")
    for arr, c, lab in (
        (best, colors[0], labels[0]),
        (worst, colors[1], labels[1]),
        (med, colors[2], labels[2]),
    ):
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.42,
            label=lab,
            color=c,
            histtype="stepfilled",
        )
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.set_title("Per-event reward (best / worst / median among K)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _finite_1d_numpy(arr: np.ndarray) -> np.ndarray:
    """Return finite 1D float values for histogram plotting."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def _rel_pt_distribution_figure(
    all_rel_pt: np.ndarray,
    best_rel_pt: np.ndarray,
) -> Any:
    """Overlaid density plot for ``pT_pred / pT_truth - 1`` diagnostics."""
    import wandb

    all_rel_pt = _finite_1d_numpy(all_rel_pt)
    best_rel_pt = _finite_1d_numpy(best_rel_pt)
    stacked = np.concatenate([all_rel_pt, best_rel_pt])
    if stacked.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = [float(x) for x in np.nanpercentile(stacked, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max(0.05 * (hi - lo), 1e-3)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REL_PT_DIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arr, color, label in (
        (all_rel_pt, "#1f77b4", "all K candidates"),
        (best_rel_pt, "#d62728", "reward-best candidate"),
    ):
        if arr.size == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.65,
            label=f"{label}: mean={arr.mean():+.3f}, mean abs={np.abs(arr).mean():.3f}",
            color=color,
            histtype="step",
            linewidth=2.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
    ax.set_ylabel("Normalized density")
    ax.set_title("Relative pT residual distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _single_rel_pt_distribution_figure(
    rel_pt: np.ndarray,
    *,
    title: str,
    label: str,
) -> Any:
    """Single-series density plot for reference/rollout relative-pT bias diagnostics."""
    import wandb

    rel_pt = _finite_1d_numpy(rel_pt)
    if rel_pt.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = [float(x) for x in np.nanpercentile(rel_pt, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(rel_pt)) if rel_pt.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max(0.05 * (hi - lo), 1e-3)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REL_PT_DIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if rel_pt.size > 0:
        ax.hist(
            rel_pt,
            bins=bins,
            density=True,
            alpha=0.75,
            label=f"{label}: mean={rel_pt.mean():+.3f}, mean abs={np.abs(rel_pt).mean():.3f}",
            color="#9467bd",
            histtype="step",
            linewidth=2.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _pt_delta_vs_truth_pt_figure(
    truth_pt: np.ndarray,
    delta_pt: np.ndarray,
    *,
    title: str,
) -> Any:
    """Profile plot of mean ``pT_pred - pT_truth`` in truth-pT bins."""
    import wandb

    truth_pt = np.asarray(truth_pt, dtype=np.float64).reshape(-1)
    delta_pt = np.asarray(delta_pt, dtype=np.float64).reshape(-1)
    if truth_pt.shape != delta_pt.shape:
        n = min(truth_pt.size, delta_pt.size)
        truth_pt = truth_pt[:n]
        delta_pt = delta_pt[:n]
    keep = np.isfinite(truth_pt) & np.isfinite(delta_pt) & (truth_pt >= 0.0)
    truth_pt = truth_pt[keep]
    delta_pt = delta_pt[keep]

    bin_edges = _diagnostic_bin_edges("pt")
    num_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means = np.full(num_bins, np.nan, dtype=np.float64)
    errors = np.full(num_bins, np.nan, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    if truth_pt.size > 0:
        bin_idx = np.digitize(truth_pt, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        for i in range(num_bins):
            vals = delta_pt[valid & (bin_idx == i)]
            counts[i] = int(vals.size)
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                errors[i] = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    has_points = np.isfinite(means)
    if np.any(has_points):
        ax.errorbar(
            centers[has_points],
            means[has_points],
            yerr=errors[has_points],
            fmt="o-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label=r"mean $(p_T^{pred} - p_T^{truth})$",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    ax_count.bar(
        centers,
        counts,
        width=float(bin_edges[1] - bin_edges[0]) * 0.85,
        alpha=0.12,
        color="gray",
        label="entries",
    )
    ax_count.set_ylabel("Entries")
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _binned_delta_profile(
    truth_value: np.ndarray,
    delta_value: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return truth-value bin centers, mean residual, standard error, and counts."""
    truth_value = np.asarray(truth_value, dtype=np.float64).reshape(-1)
    delta_value = np.asarray(delta_value, dtype=np.float64).reshape(-1)
    if truth_value.shape != delta_value.shape:
        n = min(truth_value.size, delta_value.size)
        truth_value = truth_value[:n]
        delta_value = delta_value[:n]
    keep = np.isfinite(truth_value) & np.isfinite(delta_value)
    truth_value = truth_value[keep]
    delta_value = delta_value[keep]

    if bin_edges is None:
        bin_edges = _diagnostic_bin_edges("pt")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    num_bins = len(bin_edges) - 1
    means = np.full(num_bins, np.nan, dtype=np.float64)
    errors = np.full(num_bins, np.nan, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    if truth_value.size > 0:
        bin_idx = np.digitize(truth_value, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        for i in range(num_bins):
            vals = delta_value[valid & (bin_idx == i)]
            counts[i] = int(vals.size)
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                errors[i] = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return centers, means, errors, counts


def _pt_delta_selection_profiles_figure(
    truth_pt_all: np.ndarray,
    delta_pt_all: np.ndarray,
    truth_pt_best: np.ndarray,
    delta_pt_best: np.ndarray,
    truth_pt_oracle: np.ndarray | None = None,
    delta_pt_oracle: np.ndarray | None = None,
    *,
    title: str,
) -> Any:
    """Profile plot comparing rollout-all, reward-best, and optional pT-oracle pT delta."""
    import wandb

    centers, mean_all, err_all, _ = _binned_delta_profile(truth_pt_all, delta_pt_all)
    _, mean_best, err_best, counts = _binned_delta_profile(truth_pt_best, delta_pt_best)
    mean_oracle = err_oracle = None
    if truth_pt_oracle is not None and delta_pt_oracle is not None:
        _, mean_oracle, err_oracle, _ = _binned_delta_profile(
            truth_pt_oracle, delta_pt_oracle
        )
    best_gap = mean_best - mean_all
    oracle_gap = mean_oracle - mean_all if mean_oracle is not None else None

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    has_all = np.isfinite(mean_all)
    has_best = np.isfinite(mean_best)
    if np.any(has_all):
        ax.errorbar(
            centers[has_all],
            mean_all[has_all],
            yerr=err_all[has_all],
            fmt="o-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label="all rollout candidates",
        )
    if np.any(has_best):
        ax.errorbar(
            centers[has_best],
            mean_best[has_best],
            yerr=err_best[has_best],
            fmt="s-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label="reward-best candidates",
        )
    if mean_oracle is not None and err_oracle is not None:
        has_oracle = np.isfinite(mean_oracle)
        if np.any(has_oracle):
            ax.errorbar(
                centers[has_oracle],
                mean_oracle[has_oracle],
                yerr=err_oracle[has_oracle],
                fmt="^-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                label="pT-oracle-best candidates",
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    has_gap = np.isfinite(best_gap)
    if np.any(has_gap):
        ax_gap.plot(
            centers[has_gap],
            best_gap[has_gap],
            "o-",
            linewidth=1.8,
            markersize=4,
            color="#d62728",
            label="reward-best - all",
        )
    if oracle_gap is not None:
        has_oracle_gap = np.isfinite(oracle_gap)
        if np.any(has_oracle_gap):
            ax_gap.plot(
                centers[has_oracle_gap],
                oracle_gap[has_oracle_gap],
                "^-",
                linewidth=1.8,
                markersize=4,
                color="#2ca02c",
                label="pT-oracle - all",
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax_gap.set_ylabel(r"$\Delta p_T$ gap [GeV]")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _profile_bin_edges(profile_name: str, truth_arrays: list[np.ndarray]) -> np.ndarray:
    """Bin edges for residual profile plots keyed by the profiled truth variable."""
    fixed_edges = _diagnostic_bin_edges(profile_name)
    if fixed_edges is not None:
        return fixed_edges

    finite_parts = [
        np.asarray(arr, dtype=np.float64).reshape(-1)
        for arr in truth_arrays
        if isinstance(arr, np.ndarray) and arr.size > 0
    ]
    if not finite_parts:
        lo, hi = -100.0, 100.0
    else:
        values = np.concatenate(finite_parts, axis=0)
        values = values[np.isfinite(values)]
        if values.size == 0:
            lo, hi = -100.0, 100.0
        else:
            lo, hi = [float(x) for x in np.nanpercentile(values, [0.5, 99.5])]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                center = float(np.nanmean(values)) if values.size > 0 else 0.0
                lo, hi = center - 100.0, center + 100.0
            hi = max(abs(lo), abs(hi), 1.0)
            lo = -hi
    pad = max(0.05 * (hi - lo), 1e-3)
    return np.linspace(lo - pad, hi + pad, _VAL_KIN_NUM_BINS + 1)


def _profile_axis_labels(profile_name: str) -> tuple[str, str, str]:
    """Return x-label, y-label, and display name for a residual profile variable."""
    display = {
        "pt": "pT",
        "eta": "eta",
        "phi": "phi",
        "px": "px",
        "py": "py",
        "pz": "pz",
    }.get(profile_name, profile_name)
    if profile_name in {"pt", "px", "py", "pz"}:
        return f"Truth {display} [GeV]", f"Mean delta {display} [GeV]", display
    if profile_name == "phi":
        return "Truth phi [rad]", "Mean wrapped delta phi [rad]", display
    return f"Truth {display}", f"Mean delta {display}", display


def _invisible_feature_names() -> tuple[str, ...]:
    """Invisible feature names from ``event_info.yaml`` (fallback: reward_config)."""
    event_info = getattr(global_config, "event_info", None)
    raw = getattr(event_info, "invisible_feature_names", None)
    if raw:
        return tuple(str(name) for name in raw)
    reward_names = _reward_feature_names()
    return tuple(reward_names) if reward_names is not None else ()


def _invisible_periodic_feature_indices() -> tuple[int, ...]:
    """Periodic invisible-feature indices from ``event_info.yaml`` uniform/inv-CDF metadata."""
    event_info = getattr(global_config, "event_info", None)
    raw = getattr(event_info, "invisible_inv_cdf_index", None)
    if raw is None:
        return ()
    return tuple(int(index) for index in raw)


def _generation_monitor_feature_names(*, cartesian: bool) -> tuple[str, ...]:
    """Feature names for generation-style monitoring plots."""
    if cartesian:
        return ("px", "py", "pz")
    return _invisible_feature_names()


def _variance_regularization_config(dg_cfg: Any | None = None) -> Any | None:
    """Optional anti-shrink regularization block under ``dgpo.variance_regularization``."""
    cfg = dg_cfg if dg_cfg is not None else getattr(global_config, "dgpo", None)
    return _dgpo_cfg_get(cfg, "variance_regularization", None)


def _variance_regularization_enabled(dg_cfg: Any | None = None) -> bool:
    """Whether batch-level std matching regularization is enabled."""
    block = _variance_regularization_config(dg_cfg)
    return bool(_dgpo_cfg_get(block, "enabled", False))


def _variance_regularization_weight(dg_cfg: Any | None = None) -> float:
    """Scalar weight for the anti-shrink regularizer."""
    block = _variance_regularization_config(dg_cfg)
    return max(0.0, float(_dgpo_cfg_get(block, "weight", 0.0)))


def _variance_regularization_feature_names(dg_cfg: Any | None = None) -> tuple[str, ...]:
    """Target features for anti-shrink regularization.

    Default: inspect ``event_info.yaml`` via ``event_info.invisible_feature_names`` and keep
    the angular-like entries we actually care about. This makes ``eta/phi`` and ``theta/phi``
    layouts both work without hard-coding one schema.
    """
    block = _variance_regularization_config(dg_cfg)
    raw = _dgpo_cfg_get(block, "features", None)
    if raw:
        return tuple(str(name) for name in raw)
    event_features = _invisible_feature_names()
    preferred = tuple(
        str(name) for name in event_features
        if str(name) in {"eta", "theta", "phi"}
    )
    return preferred if preferred else event_features


def _named_invisible_feature_tensors(
    kin: Tensor,
    *,
    cartesian: bool,
    feature_names: tuple[str, ...],
) -> dict[str, Tensor]:
    """Expose invisible kinematics as named tensors in the current feature space."""
    if int(kin.shape[-1]) <= 0:
        return {}
    if cartesian:
        if int(kin.shape[-1]) < 3:
            return {}
        log_pt, eta, phi = cartesian_to_log_pt_eta_phi(
            kin[..., 0],
            kin[..., 1],
            kin[..., 2],
        )
        return {
            "log_pt": log_pt,
            "pt": torch.expm1(log_pt),
            "eta": eta,
            "phi": phi,
            "px": kin[..., 0],
            "py": kin[..., 1],
            "pz": kin[..., 2],
        }
    out: dict[str, Tensor] = {}
    max_features = min(len(feature_names), int(kin.shape[-1]))
    for index, name in enumerate(feature_names[:max_features]):
        out[str(name)] = kin[..., index]
    return out


def _masked_batch_std(values: Tensor, mask: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Population std over valid batch slots; ``mask`` is 0/1 with the same broadcast shape."""
    weights = mask.to(device=values.device, dtype=values.dtype)
    count = weights.sum().clamp(min=1.0)
    mean = (values * weights).sum() / count
    var = ((values - mean).pow(2) * weights).sum() / count
    return torch.sqrt(var.clamp(min=0.0) + float(eps))


def _variance_matching_penalty(
    pred_phys: Tensor,
    truth_phys: Tensor,
    valid_mask: Tensor,
    *,
    cartesian: bool,
    feature_names: tuple[str, ...],
    selected_features: tuple[str, ...],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Small anti-shrink penalty using relative shrinkage vs truth std."""
    zero = pred_phys.new_zeros(())
    diag: dict[str, Tensor] = {
        "train/regularization/variance/active": pred_phys.new_tensor(0.0, dtype=torch.float64),
        "train/regularization/variance/active_features": pred_phys.new_tensor(0.0, dtype=torch.float64),
        "train/regularization/variance/raw": zero.detach(),
    }
    if not selected_features:
        return zero, diag

    pred_named = _named_invisible_feature_tensors(
        pred_phys,
        cartesian=cartesian,
        feature_names=feature_names,
    )
    truth_named = _named_invisible_feature_tensors(
        truth_phys,
        cartesian=cartesian,
        feature_names=feature_names,
    )
    mask = valid_mask.squeeze(-1) if int(valid_mask.dim()) == int(pred_phys.dim()) else valid_mask
    penalties: list[Tensor] = []
    active_features = 0
    valid_count = float(mask.sum().detach().cpu())

    for feature_name in selected_features:
        prefix = f"train/regularization/variance/{feature_name}"
        pred_feature = pred_named.get(feature_name)
        truth_feature = truth_named.get(feature_name)
        diag[f"{prefix}/active"] = pred_phys.new_tensor(0.0, dtype=torch.float64)
        diag[f"{prefix}/count"] = pred_phys.new_tensor(valid_count, dtype=torch.float64)
        if pred_feature is None or truth_feature is None or valid_count < 2.0:
            diag[f"{prefix}/std_truth"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/std_pred"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/std_delta_ratio"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/std_gap"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/penalty"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            continue
        std_truth = _masked_batch_std(truth_feature.detach(), mask)
        std_pred = _masked_batch_std(pred_feature, mask)
        std_scale = std_truth.detach().clamp(min=1.0e-8)
        std_delta_ratio = (std_pred - std_truth) / std_scale
        std_gap = torch.relu(-std_delta_ratio)
        penalty_feature = std_gap.pow(2)
        penalties.append(penalty_feature)
        active_features += 1
        diag[f"{prefix}/active"] = pred_phys.new_tensor(1.0, dtype=torch.float64)
        diag[f"{prefix}/std_truth"] = std_truth.detach().to(dtype=torch.float64)
        diag[f"{prefix}/std_pred"] = std_pred.detach().to(dtype=torch.float64)
        diag[f"{prefix}/std_delta_ratio"] = std_delta_ratio.detach().to(dtype=torch.float64)
        diag[f"{prefix}/std_gap"] = std_gap.detach().to(dtype=torch.float64)
        diag[f"{prefix}/penalty"] = penalty_feature.detach().to(dtype=torch.float64)

    if penalties:
        raw = torch.stack(penalties).mean()
        diag["train/regularization/variance/active"] = pred_phys.new_tensor(1.0, dtype=torch.float64)
        diag["train/regularization/variance/active_features"] = pred_phys.new_tensor(
            float(active_features), dtype=torch.float64
        )
        diag["train/regularization/variance/raw"] = raw.detach().to(dtype=torch.float64)
        return raw, diag
    return zero, diag


def _generation_special_bin_edges(feature_name: str) -> np.ndarray | None:
    """Mirror EveNet ``Generation-Binning`` lookup for ``neutrino-{feature}``."""
    metrics_cfg = getattr(global_config.options, "Metrics", None)
    if metrics_cfg is None:
        return None
    bins_cfg = metrics_cfg.get("Generation-Binning", {})
    raw = bins_cfg.get(f"neutrino-{feature_name}")
    if raw is None or len(raw) != 3:
        return None
    nbins, lo, hi = raw
    try:
        return np.linspace(float(lo), float(hi), int(nbins))
    except (TypeError, ValueError):
        return None


def _available_truth_pred_features(
    arrays: Mapping[str, np.ndarray],
    feature_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Feature names that have non-empty truth/pred arrays for 2D truth-vs-pred plots."""
    available: list[str] = []
    for feature_name in feature_names:
        truth = np.asarray(
            arrays.get(f"{feature_name}_truth", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        pred = np.asarray(
            arrays.get(f"{feature_name}_pred", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if min(truth.size, pred.size) == 0:
            continue
        if not np.isfinite(truth).any() or not np.isfinite(pred).any():
            continue
        available.append(str(feature_name))
    return tuple(available)


def _supports_legacy_invisible_kinematics(*, cartesian: bool, feature_dim: int | None = None) -> bool:
    """Whether legacy ``(log_pt, eta, phi)`` / Cartesian diagnostics are valid."""
    if cartesian:
        return feature_dim is None or int(feature_dim) >= 3
    feature_names = _invisible_feature_names()
    if len(feature_names) < 3:
        return False
    if tuple(feature_names[:3]) != ("log_pt", "eta", "phi"):
        return False
    return feature_dim is None or int(feature_dim) >= 3


def _validation_winrate_enabled(*, compute_winrate: bool, cartesian: bool, feature_dim: int | None = None) -> bool:
    """Validation win-rate is available whenever the extra reference rollout is enabled."""
    del cartesian, feature_dim
    return bool(compute_winrate)


def _validation_profile_feature_names(*, cartesian: bool) -> tuple[str, ...]:
    """Validation residual-profile features derived from ``event_info.yaml``."""
    if _supports_legacy_invisible_kinematics(cartesian=cartesian):
        return ("pt", "eta")
    feature_names = _invisible_feature_names()
    return feature_names if feature_names else ("feature_0",)


def _delta_selection_profiles_figure(
    truth_all: np.ndarray,
    delta_all: np.ndarray,
    truth_best: np.ndarray,
    delta_best: np.ndarray,
    truth_oracle: np.ndarray | None = None,
    delta_oracle: np.ndarray | None = None,
    *,
    profile_name: str,
    title: str,
) -> Any:
    """Profile plot comparing rollout-all, reward-best, and optional variable-oracle residuals."""
    import wandb

    x_label, y_label, display = _profile_axis_labels(profile_name)
    bin_edges = _profile_bin_edges(
        profile_name,
        [truth_all, truth_best] + ([] if truth_oracle is None else [truth_oracle]),
    )
    centers, mean_all, err_all, _ = _binned_delta_profile(
        truth_all, delta_all, bin_edges=bin_edges
    )
    _, mean_best, err_best, counts = _binned_delta_profile(
        truth_best, delta_best, bin_edges=bin_edges
    )
    mean_oracle = err_oracle = None
    if truth_oracle is not None and delta_oracle is not None:
        _, mean_oracle, err_oracle, _ = _binned_delta_profile(
            truth_oracle, delta_oracle, bin_edges=bin_edges
        )
    best_gap = mean_best - mean_all
    oracle_gap = mean_oracle - mean_all if mean_oracle is not None else None

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for means, errs, fmt, label, color in (
        (mean_all, err_all, "o-", "all rollout candidates", "#1f77b4"),
        (mean_best, err_best, "s-", "reward-best candidates", "#d62728"),
    ):
        keep = np.isfinite(means)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                means[keep],
                yerr=errs[keep],
                fmt=fmt,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )
    if mean_oracle is not None and err_oracle is not None:
        keep = np.isfinite(mean_oracle)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                mean_oracle[keep],
                yerr=err_oracle[keep],
                fmt="^-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color="#2ca02c",
                label=f"{display}-oracle-best candidates",
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    gap_series = (
        (best_gap, "o-", "reward-best - all", "#d62728"),
        (oracle_gap, "^-", f"{display}-oracle - all", "#2ca02c"),
    )
    for gap, fmt, label, color in gap_series:
        if gap is None:
            continue
        keep = np.isfinite(gap)
        if np.any(keep):
            ax_gap.plot(
                centers[keep],
                gap[keep],
                fmt,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(x_label)
    ax_gap.set_ylabel("Selection gap")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


_DIAG_PROFILE_NAMES = ("pt", "eta", "phi", "px", "py", "pz")


def _diag_profile_raw_key(profile_name: str, suffix: str) -> str:
    """Internal metric key for raw arrays used by accumulated W&B profile images."""
    return f"_diag_{profile_name}_profile_{suffix}"


def _diag_profile_log_key(profile_name: str, *, accumulated: bool = False) -> str:
    """Public W&B image key for a binned residual profile."""
    suffix = "_accumulated" if accumulated else ""
    return (
        f"diagnostics/reward_hacking/profile/"
        f"{profile_name}_delta_vs_truth_{profile_name}{suffix}"
    )


def _diag_profile_title(profile_name: str, *, accumulated_batches: int | None = None) -> str:
    """Human-readable title for a binned residual profile."""
    _x_label, _y_label, display = _profile_axis_labels(profile_name)
    title = f"Reward selection {display} bias vs truth {display}"
    if accumulated_batches is not None:
        title += f" ({accumulated_batches} train batches)"
    return title


def _align_truth_tensor_to_delta(truth: Tensor, delta: Tensor) -> Tensor:
    """Expand cached truth tensors from ``(1, B, S)`` to the candidate shape when needed."""
    if truth.shape == delta.shape:
        return truth
    if truth.dim() == delta.dim() and truth.shape[0] == 1 and truth.shape[1:] == delta.shape[1:]:
        return truth.expand_as(delta)
    return truth


def _finite_profile_numpy(truth: Tensor, delta: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired truth and residual arrays for plotting."""
    mask = torch.isfinite(truth) & torch.isfinite(delta)
    return (
        truth[mask].detach().float().cpu().numpy(),
        delta[mask].detach().float().cpu().numpy(),
    )


def _pt_delta_prefix_vs_full_figure(
    truth_pt_reward_prefix: np.ndarray,
    delta_pt_reward_prefix: np.ndarray,
    truth_pt_reward_full: np.ndarray,
    delta_pt_reward_full: np.ndarray,
    truth_pt_oracle_prefix: np.ndarray,
    delta_pt_oracle_prefix: np.ndarray,
    truth_pt_oracle_full: np.ndarray,
    delta_pt_oracle_full: np.ndarray,
    *,
    prefix_k: int,
    full_k: int,
    title: str,
) -> Any:
    """Compare first-prefix-K vs full-K selection for reward-best and pT-oracle."""
    import wandb

    centers, reward_prefix, reward_prefix_err, counts = _binned_delta_profile(
        truth_pt_reward_prefix, delta_pt_reward_prefix
    )
    _, reward_full, reward_full_err, _ = _binned_delta_profile(
        truth_pt_reward_full, delta_pt_reward_full
    )
    _, oracle_prefix, oracle_prefix_err, _ = _binned_delta_profile(
        truth_pt_oracle_prefix, delta_pt_oracle_prefix
    )
    _, oracle_full, oracle_full_err, _ = _binned_delta_profile(
        truth_pt_oracle_full, delta_pt_oracle_full
    )

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    series = (
        (reward_prefix, reward_prefix_err, "o-", f"reward-best first {prefix_k}", "#ff7f0e"),
        (reward_full, reward_full_err, "s-", f"reward-best full {full_k}", "#d62728"),
        (oracle_prefix, oracle_prefix_err, "^-", f"pT-oracle first {prefix_k}", "#2ca02c"),
        (oracle_full, oracle_full_err, "v-", f"pT-oracle full {full_k}", "#1f77b4"),
    )
    for means, errs, fmt, label, color in series:
        keep = np.isfinite(means)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                means[keep],
                yerr=errs[keep],
                fmt=fmt,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    reward_gain = reward_full - reward_prefix
    oracle_gain = oracle_full - oracle_prefix
    for gain, fmt, label, color in (
        (reward_gain, "o-", f"reward full {full_k} - first {prefix_k}", "#d62728"),
        (oracle_gain, "^-", f"oracle full {full_k} - first {prefix_k}", "#2ca02c"),
    ):
        keep = np.isfinite(gain)
        if np.any(keep):
            ax_gap.plot(
                centers[keep],
                gain[keep],
                fmt,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax_gap.set_ylabel(r"Full - prefix [GeV]")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _projection_metric_finite(out: dict[str, Any], key: str) -> float | None:
    """Return a finite float from a metrics dict, or ``None``."""
    val = out.get(key)
    if val is None:
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _projection_labeled_bar_figure(
    *,
    title: str,
    series: list[tuple[str, float | None, str]],
    ylabel: str,
    reference_lines: list[tuple[str, float, str, str]] | None = None,
) -> Any | None:
    """Horizontal bar chart for projection estimator comparison (``wandb.Image``)."""
    import wandb

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for label, val, color in series:
        if val is None or not math.isfinite(float(val)):
            continue
        labels.append(label)
        values.append(float(val))
        colors.append(color)
    if not labels:
        return None

    fig_h = max(3.2, 0.55 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos, labels=labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.35)
    if reference_lines:
        for ref_label, ref_val, ref_color, ref_style in reference_lines:
            if not math.isfinite(float(ref_val)):
                continue
            ax.axvline(
                float(ref_val),
                color=ref_color,
                linewidth=1.2,
                linestyle=ref_style,
                alpha=0.9,
                label=ref_label,
            )
    if reference_lines:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _projection_violation_compare_figure(out: dict[str, Any]) -> Any | None:
    """Compare violation margins ``v = C - epsilon`` (lambda driver vs baselines)."""
    return _projection_labeled_bar_figure(
        title="Projection violation (v = C - epsilon)",
        ylabel="Violation margin v",
        series=[
            (
                "v_selected (lambda)",
                _projection_metric_finite(out, "projection/v_selected"),
                "#C44E52",
            ),
            (
                "v_linear (ref)",
                _projection_metric_finite(out, "projection/v_linear"),
                "#4C72B0",
            ),
        ],
        reference_lines=[("feasible (v=0)", 0.0, "black", ":")],
    )


def _build_projection_wandb_panel_metrics(
    out: dict[str, Any],
    plot_names: set[str],
) -> dict[str, Any]:
    """Optional W&B media panels for projection estimator comparison."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return {}
    active = out.get("projection/active")
    if active is None or not math.isfinite(float(active)) or float(active) != 1.0:
        return {}
    panels: dict[str, Any] = {}
    if "projection_violation_compare" in plot_names:
        try:
            img = _projection_violation_compare_figure(out)
            if img is not None:
                panels["projection/panel/violation_compare"] = img
        except Exception as exc:
            _log.warning("[DGPO] projection violation panel failed: %s", exc)
    return panels


@torch.no_grad()
def _build_reference_bias_metrics(
    candidates: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
    log_distribution: bool = False,
    diagnostic_plot_names: set[str] | None = None,
) -> dict[str, Any]:
    """Diagnostics for the rollout/reference policy's raw kinematic bias vs truth."""
    out: dict[str, Any] = {}
    K, B, N_nu, F = candidates.shape
    S = min(2, N_nu)
    if S == 0:
        return out

    if "x_invisible_mask" in batch:
        mask = batch["x_invisible_mask"].to(device=candidates.device, dtype=candidates.dtype)
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() != 2:
            return out
        slot_valid = mask[:, :S] > 0
    else:
        slot_valid = torch.ones(B, S, device=candidates.device, dtype=torch.bool)

    event_valid = get_event_valid_mask(batch, B, candidates.device, candidates.dtype) > 0
    valid_kbs = (slot_valid & event_valid.unsqueeze(-1)).unsqueeze(0).expand(K, B, S)

    if cartesian:
        truth = batch.get("x_invisible_cartesian")
        if not isinstance(truth, Tensor) or truth.dim() != 3 or truth.shape[0] != B:
            return out
        truth_xyz = truth[:, :S, :3].to(device=candidates.device, dtype=candidates.dtype)
        cand_xyz = candidates[:, :, :S, :3]
        truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
            truth_xyz[..., 0],
            truth_xyz[..., 1],
            truth_xyz[..., 2],
        )
        cand_log_pt, cand_eta, cand_phi = cartesian_to_log_pt_eta_phi(
            cand_xyz[..., 0],
            cand_xyz[..., 1],
            cand_xyz[..., 2],
        )
    else:
        truth = batch.get("x_invisible")
        if (
            not isinstance(truth, Tensor)
            or truth.dim() != 3
            or truth.shape[0] != B
            or F < 3
        ):
            return out
        truth_kin = truth[:, :S, :3].to(device=candidates.device, dtype=candidates.dtype)
        cand_kin = candidates[:, :, :S, :3]
        truth_log_pt, truth_eta, truth_phi = truth_kin.unbind(dim=-1)
        cand_log_pt, cand_eta, cand_phi = cand_kin.unbind(dim=-1)

    truth_pt = torch.expm1(truth_log_pt.clamp(-10.0, 10.0)).unsqueeze(0)
    cand_pt = torch.expm1(cand_log_pt.clamp(-10.0, 10.0))
    residuals = {
        "pt": cand_pt - truth_pt,
        "rel_pt": (cand_pt - truth_pt) / truth_pt.clamp(min=1e-6),
        "eta": cand_eta - truth_eta.unsqueeze(0),
        "phi": torch.atan2(
            torch.sin(cand_phi - truth_phi.unsqueeze(0)),
            torch.cos(cand_phi - truth_phi.unsqueeze(0)),
        ),
    }

    finite_residuals: dict[str, Tensor] = {}
    for name, tensor in residuals.items():
        values = tensor[valid_kbs]
        values = values[torch.isfinite(values)]
        finite_residuals[name] = values
        if name == "rel_pt":
            mean_key = f"diagnostics/reference_bias/all/{name}/mean"
            abs_mean_key = f"diagnostics/reference_bias/all/{name}/abs_mean"
        else:
            mean_key = f"diagnostics/reference_bias/all/{name}/delta_mean"
            abs_mean_key = f"diagnostics/reference_bias/all/{name}/delta_abs_mean"
        if values.numel() > 0:
            out[mean_key] = float(values.mean().detach().cpu())
            out[abs_mean_key] = float(values.abs().mean().detach().cpu())
        else:
            out[mean_key] = float("nan")
            out[abs_mean_key] = float("nan")

    plot_names = diagnostic_plot_names or set()
    if log_distribution and "rel_pt_dist" in plot_names:
        rel_pt = finite_residuals.get("rel_pt")
        if rel_pt is not None:
            try:
                import wandb  # noqa: F401

                out["diagnostics/reference_bias/dist/rel_pt"] = (
                    _single_rel_pt_distribution_figure(
                        rel_pt.detach().float().cpu().numpy(),
                        title="Reference / frozen-rollout relative pT bias",
                        label="rollout candidates",
                    )
                )
            except Exception:
                pass
        delta_pt = residuals["pt"][valid_kbs]
        truth_pt_rep = truth_pt.expand(K, B, S)[valid_kbs]
        profile_mask = torch.isfinite(delta_pt) & torch.isfinite(truth_pt_rep)
        delta_pt = delta_pt[profile_mask]
        truth_pt_rep = truth_pt_rep[profile_mask]
        try:
            import wandb  # noqa: F401

            out["diagnostics/reference_bias/profile/pt_delta_vs_truth_pt"] = (
                _pt_delta_vs_truth_pt_figure(
                    truth_pt_rep.detach().float().cpu().numpy(),
                    delta_pt.detach().float().cpu().numpy(),
                    title="Reference / frozen-rollout pT bias vs truth pT",
                )
            )
        except Exception:
            pass
    return out



@torch.no_grad()
def build_reward_distribution_histograms(
    rewards: Tensor,
    valid_b: Tensor,
) -> dict[str, Any]:
    """Panel ``reward/dist``: overlapped 1D histograms for best / worst / median as ``wandb.Image``.

    Logs a **single** media key each time so the W&B Images panel shows one series with a
    **step slider** (same pattern as ``wandb.Image`` validation plots in ``evenet/``).
    """
    try:
        import wandb  # noqa: F401 — require package; figure built in _reward_dist_overlaid_figure
    except ImportError:
        return {}
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        return {}
    rv = rewards[:, vb]
    best = rv.max(dim=0).values.detach().float().cpu().numpy()
    worst = rv.min(dim=0).values.detach().float().cpu().numpy()
    med = rv.median(dim=0).values.detach().float().cpu().numpy()
    return {"reward/dist/overlap": _reward_dist_overlaid_figure(best, worst, med)}


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a Ray/Lightning-style batch dict to ``device``."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _save_trainable_weights(model: torch.nn.Module) -> dict[str, Tensor]:
    """Buffer for EMA rollout swap (only parameters that participate in EMA shadow)."""
    core = _unwrap_core_evenet(model)
    return {n: p.data.clone() for n, p in core.named_parameters() if p.requires_grad}


def _restore_trainable_weights(model: torch.nn.Module, buf: dict[str, Tensor]) -> None:
    core = _unwrap_core_evenet(model)
    for n, p in core.named_parameters():
        if n in buf:
            p.data.copy_(buf[n])


def _scales_from_normalization(normalization_dict: dict[str, Any] | None) -> dict[str, float]:
    """Build the 6 per-component scales from ``normalization.pt['invisible_cartesian_std']``.

    Reads shape-``(3,)`` std in order ``[px, py, pz]`` (computed sample-wise at
    preprocessing over both ``nu1`` and ``nu2`` slots) and applies the same px/py/pz
    std to both neutrinos. Raises if the key is absent — re-run preprocessing first.
    """
    if normalization_dict is None or "invisible_cartesian_std" not in normalization_dict:
        raise ValueError(
            "component_normalized_truth_distance requires 'invisible_cartesian_std' in "
            "normalization.pt (shape (3,) [px, py, pz]). Re-run preprocessing so the "
            "Cartesian std is saved."
        )
    std_t = normalization_dict["invisible_cartesian_std"]["Source"]
    std = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std) < 3:
        raise ValueError(
            f"invisible_cartesian_std must have 3 entries [px, py, pz], got {len(std)}"
        )
    return {
        "nu1_px": float(std[0]), "nu1_py": float(std[1]), "nu1_pz": float(std[2]),
        "nu2_px": float(std[0]), "nu2_py": float(std[1]), "nu2_pz": float(std[2]),
    }


def _reward_feature_names() -> tuple[str, ...] | None:
    rc = getattr(global_config, "reward_config", None)
    if rc is None:
        return None
    raw = getattr(rc, "feature_names", None)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def _reward_component_axis_pairs(component_names: tuple[str, ...] | list[str]) -> dict[str, tuple[str, str]]:
    """Pair ``nu1_*`` and ``nu2_*`` components by feature name."""
    names = set(str(name) for name in component_names)
    pairs: dict[str, tuple[str, str]] = {}
    for name in sorted(names):
        if not name.startswith("nu1_"):
            continue
        feature_name = name[4:]
        other = f"nu2_{feature_name}"
        if other in names:
            pairs[feature_name] = (name, other)
    return pairs


def build_reward_aggregator(
    model: torch.nn.Module,
    device: torch.device,
    normalization_dict: dict[str, Any] | None = None,
) -> RewardAggregator:
    """Construct the configured DGPO reward from ``reward_config``."""
    rc = global_config.reward_config
    reward_type = str(getattr(rc, "type", "component_normalized_truth_distance")).strip().lower()
    cn = getattr(rc, "component_normalized", None)
    eps = float(getattr(cn, "eps", 1e-8)) if cn is not None else 1e-8
    primary_weight = float(getattr(rc, "weight", 1.0))
    component_weight = float(getattr(cn, "weight", 0.0)) if cn is not None else 0.0
    feature_names = _reward_feature_names()
    agg = RewardAggregator()
    if reward_type in {"calibration_magnitude", "physics_consistency", "ztautau_calibration_magnitude"}:
        _log.info(
            "[DGPO/reward] using calibration_magnitude reward for feature_names=%s.",
            feature_names,
        )
        agg.add(
            CalibrationMagnitudeReward(feature_names=feature_names),
            primary_weight,
        )

    use_component_reward = reward_type == "component_normalized_truth_distance" or component_weight > 0.0
    if use_component_reward:
        if feature_names is not None:
            scales = build_feature_space_scales(
                normalization_dict,
                feature_names=feature_names,
            )
            _log.info(
                "[DGPO/reward] feature-space scales from normalization.pt invisible_std for %s: %s",
                feature_names,
                {k: round(v, 4) for k, v in scales.items()},
            )
        else:
            scales = _scales_from_normalization(normalization_dict)
            _log.info(
                "[DGPO/reward] component_normalized scales from normalization.pt "
                "invisible_cartesian_std [px, py, pz]: %s",
                {k: round(v, 4) for k, v in scales.items()},
            )
        agg.add(
            ComponentNormalizedTruthDistanceReward(
                scales,
                cartesian=_truth_generation_cartesian(),
                eps=eps,
                feature_names=feature_names,
            ),
            primary_weight if reward_type == "component_normalized_truth_distance" else component_weight,
        )

    if not agg.sources:
        raise ValueError(f"unsupported reward_config.type={reward_type!r}")
    return agg


@torch.no_grad()
def generate_neutrino_candidates(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sampler: DDIMSampler,
    *,
    K: int,
    num_ddim_steps: int,
    device: torch.device,
    parallel_chains: int = 1,
    tqdm_k_chains: bool = False,
    use_tqdm_ddim: bool = False,
    chain_progress_desc: str = "DGPO DDIM chains",
) -> Tensor:
    """DDIM rollouts in physical invisible space, shape ``(K, B, N_nu, F)``.

    ``data_shape`` for the DDIM prior must match what ``predict_diffusion_vector``
    receives as ``noise_x``:  ``(B, N_nu, invisible_input_dim)``.  When
    ``TruthGeneration.cartesian: true`` that is 3 (px, py, pz), not the full
    7-D ``x_invisible`` from the parquet.
    """
    if "x_invisible" not in batch:
        raise KeyError("batch missing x_invisible for DDIM data_shape.")
    B, N_nu = batch["x_invisible"].shape[:2]
    inv_dim = int(getattr(model, "invisible_input_dim", batch["x_invisible"].shape[-1]))
    data_shape = (B, N_nu, inv_dim)
    candidates: list[Tensor] = []
    parallel_chains = max(1, min(int(parallel_chains), int(K)))
    noise_mask = batch["x_invisible_mask"].unsqueeze(-1)
    k_iter: Any = range(0, K, parallel_chains)
    if tqdm_k_chains:
        try:
            from tqdm.auto import tqdm

            k_iter = tqdm(
                range(0, K, parallel_chains),
                desc=chain_progress_desc,
                leave=False,
                unit="group",
            )
        except ImportError:
            k_iter = range(0, K, parallel_chains)

    inner_name = f"{chain_progress_desc} steps"
    for chain_start in k_iter:
        chain_count = min(parallel_chains, K - chain_start)
        if chain_count == 1:
            batch_group = batch
            noise_mask_group = noise_mask
            data_shape_group = data_shape
        else:
            batch_group = repeat_batch_for_candidates(batch, chain_count)
            noise_mask_group = repeat_batch_for_candidates(
                {"x_invisible_mask": noise_mask},
                chain_count,
            )["x_invisible_mask"]
            data_shape_group = (chain_count * B, N_nu, inv_dim)
        pred_partial = partial(
            model.predict_diffusion_vector,
            mode="neutrino",
            cond_x=batch_group,
            noise_mask=noise_mask_group,
        )
        gen = sampler.sample(
            data_shape=data_shape_group,
            pred_fn=pred_partial,
            num_steps=num_ddim_steps,
            normalize_fn=model.invisible_normalizer,
            remove_padding=True,
            noise_mask=noise_mask_group,
            use_tqdm=use_tqdm_ddim,
            process_name=inner_name,
        )
        if chain_count == 1:
            candidates.append(gen)
        else:
            candidates.extend(gen.reshape(chain_count, B, N_nu, gen.shape[-1]).unbind(dim=0))
    return torch.stack(candidates, dim=0)


def _normalize_candidates_for_policy(
    model: torch.nn.Module,
    c_phys: Tensor,
    inv_mask: Tensor,
) -> Tensor:
    """Map denormalized DDIM output to the normalized space for ``predict_diffusion_vector``.

    Matches training / DDIM: raw invisible is padded to ``sequential_input_dim``, then
    ``invisible_normalizer`` runs on that width. ``predict_diffusion_vector`` (neutrino) then
    applies ``F.pad(..., invisible_padding)`` itself, so ``noise_x`` must be only the first
    ``invisible_input_dim`` channels (same width as ``DDIMSampler`` uses from ``x_invisible``),
    not the full padded-normalized tensor — otherwise features become ``sequential + padding``
    and ``torch.cat`` with jets fails (e.g. 7 vs 11).
    """
    # c_phys: (R, N_nu, F_phys)
    pad = int(getattr(model, "invisible_padding", 0))
    m = inv_mask.unsqueeze(-1).to(dtype=c_phys.dtype)
    x = c_phys
    if pad > 0:
        x = F.pad(x, (0, pad))
    full_norm = model.invisible_normalizer(x=x, mask=m)
    inv_in = int(getattr(model, "invisible_input_dim", full_norm.shape[-1]))
    return full_norm[..., :inv_in]


def per_row_velocity_mse(
    pred_v: Tensor,
    target_v: Tensor,
    noise_mask_bn11: Tensor,
    invisible_padding: int,
) -> Tensor:
    """Masked mean squared error per row (one scalar per event×candidate row)."""
    m = noise_mask_bn11.expand_as(pred_v).to(dtype=pred_v.dtype)
    if invisible_padding > 0:
        m = m.clone()
        m[:, :, -invisible_padding:] = 0.0
    sq = (pred_v - target_v).pow(2) * m
    den = m.sum(dim=(1, 2)).clamp(min=1e-8)
    return sq.sum(dim=(1, 2)) / den


def _diag_scalar_float(diag: dict[str, Tensor], key: str) -> float:
    """Single scalar tensor from diagnostics, or NaN if absent / non-finite."""
    t = diag.get(key)
    if t is None:
        return float("nan")
    v = float(t.detach().float().cpu())
    return v if math.isfinite(v) else float("nan")


def _parameter_panel_from_diag(diag_last: dict[str, Tensor]) -> dict[str, float]:
    """Map loss diagnostics to W&B ``parameter/*`` keys (extend with more tensors here later)."""
    out: dict[str, float] = {}
    mapping = (
        ("w_e_mean", "parameter/w_e/mean"),
        ("w_e_std", "parameter/w_e/std"),
        ("w_e_min", "parameter/w_e/min"),
        ("w_e_max", "parameter/w_e/max"),
        ("kl_weight_mean", "parameter/kl_weight/mean"),
        ("kl_weight_min", "parameter/kl_weight/min"),
        ("kl_weight_max", "parameter/kl_weight/max"),
    )
    for src, dst in mapping:
        t = diag_last.get(src)
        if t is None:
            continue
        v = float(t.detach().cpu())
        if math.isfinite(v):
            out[dst] = v
    return out


def _mean_diag_dict(diags: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Elementwise mean of detached loss diagnostics across training sub-steps (accumulate mode).

    Keys are unioned across sub-steps so optional diagnostic panels survive
    when some substeps omit them—the missing slice is averaged as NaN (then typically filtered by
    the logger).
    """
    if not diags:
        return {}
    all_keys: set[str] = set()
    for d in diags:
        all_keys.update(d.keys())
    out: dict[str, Tensor] = {}
    for k in sorted(all_keys):  # stable ordering helps debugging parity across ranks.
        first: Tensor | None = None
        for d in diags:
            t = d.get(k)
            if t is None:
                continue
            t = t.detach()
            first = t
            break
        if first is None:
            continue
        filled: list[Tensor] = []
        for d in diags:
            t = d.get(k)
            if t is None:
                filled.append(torch.tensor(float("nan"), device=first.device, dtype=first.dtype))
            else:
                filled.append(t.detach())
        out[k] = torch.stack(filled, dim=0).mean(dim=0)
    return out


def _finite_mean_float(values: Tensor) -> float:
    """Mean over finite tensor entries, or NaN if none are finite."""
    flat = values.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().detach().cpu())


@torch.no_grad()
def _build_reward_source_metrics(
    rewards: Tensor,
    valid_b: Tensor,
    reward_agg: RewardAggregator,
    reward_breakdown: dict[str, Tensor] | None,
) -> dict[str, float]:
    """Generic per-source reward decomposition for spotting competing reward terms."""
    out: dict[str, float] = {}
    if not reward_breakdown:
        return out

    weight_by_name: dict[str, float] = {}
    for src, weight in reward_agg.sources:
        weight_by_name[src.name] = weight_by_name.get(src.name, 0.0) + float(weight)

    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        for name in reward_breakdown:
            for suffix in (
                "mean",
                "weighted_mean",
                "selected_by_total_mean",
                "selected_by_total_weighted_mean",
                "source_best_of_k",
                "source_last_place",
                "selection_gap",
            ):
                out[f"reward/sources/{name}/{suffix}"] = float("nan")
        return out

    total_v = rewards[:, vb]
    total_best_k = total_v.argmax(dim=0)
    cols = torch.arange(int(total_best_k.numel()), device=rewards.device, dtype=torch.long)

    for name, tensor in reward_breakdown.items():
        if tensor.dim() != 2:
            continue
        rv = tensor[:, vb]
        weight = float(weight_by_name.get(name, 1.0))
        selected = rv[total_best_k, cols]
        mean = _finite_mean_float(rv)
        selected_mean = _finite_mean_float(selected)
        out[f"reward/sources/{name}/mean"] = mean
        out[f"reward/sources/{name}/weighted_mean"] = weight * mean
        out[f"reward/sources/{name}/selected_by_total_mean"] = selected_mean
        out[f"reward/sources/{name}/selected_by_total_weighted_mean"] = weight * selected_mean
        out[f"reward/sources/{name}/source_best_of_k"] = _finite_mean_float(
            rv.max(dim=0).values
        )
        out[f"reward/sources/{name}/source_last_place"] = _finite_mean_float(
            rv.min(dim=0).values
        )
        out[f"reward/sources/{name}/selection_gap"] = selected_mean - mean
    return out


@torch.no_grad()
def _build_reward_extra_metrics(
    rewards: Tensor,
    valid_b: Tensor,
    reward_agg: RewardAggregator,
    reward_breakdown: dict[str, Tensor] | None = None,
    *,
    log_distribution: bool = False,
    collect_profile_accum: bool = False,
    diagnostic_plot_names: set[str] | None = None,
) -> dict[str, Any]:
    """Light extras for W&B: ``reward/raw/{mean,std}`` and per-component contributions.

    Per-component contributions are emitted only when the active reward is
    :class:`ComponentNormalizedTruthDistanceReward`.
    They are reported under the independent ``components/`` panel as **negative** squared errors
    (i.e. per-component reward contributions, ``-err_c``) so the sign convention
    matches ``reward/raw/*`` (``<= 0``, larger = better, increasing = improving).
    By construction ``sum_c components/{c}/mean == reward/raw/mean`` for the
    component-normalized reward.

    Reward-hacking checks keep compact scalar breakdowns for the competing reward
    components: axis reward means, raw residual means, raw absolute residual means,
    and the relative-pT distribution panel.
    """
    out: dict[str, Any] = {}
    vb = valid_b.reshape(-1) > 0
    plot_names = diagnostic_plot_names or set()

    def _log_mean_abs(prefix: str, values: Tensor) -> Tensor:
        finite = values.reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() > 0:
            out[f"{prefix}/delta_mean"] = float(finite.mean().detach().cpu())
            out[f"{prefix}/delta_abs_mean"] = float(finite.abs().mean().detach().cpu())
        else:
            out[f"{prefix}/delta_mean"] = float("nan")
            out[f"{prefix}/delta_abs_mean"] = float("nan")
        return finite

    def _log_mean(prefix: str, values: Tensor) -> None:
        finite = values.reshape(-1)
        finite = finite[torch.isfinite(finite)]
        out[prefix] = float(finite.mean().detach().cpu()) if finite.numel() > 0 else float("nan")

    if vb.sum() > 0:
        r = rewards[:, vb]
        out["reward/raw/mean"] = float(r.mean().detach().cpu())
        out["reward/raw/std"] = float(r.std(unbiased=False).detach().cpu()) if r.numel() > 1 else 0.0
    else:
        out["reward/raw/mean"] = float("nan")
        out["reward/raw/std"] = float("nan")

    out.update(_build_reward_source_metrics(rewards, valid_b, reward_agg, reward_breakdown))

    for src, _w in reward_agg.sources:
        if vb.sum() > 0:
            rewards_v = rewards[:, vb]
            best_k = rewards_v.argmax(dim=0)
            bv = int(best_k.numel())
            cols = torch.arange(bv, device=rewards.device, dtype=torch.long)
            topology = src.last_topology_metrics()
            if topology is not None:
                for name, values in topology.items():
                    all_values = values[:, vb]
                    best_values = all_values[best_k, cols]
                    _log_mean(f"diagnostics/ztautau_back_to_back/all/{name}", all_values)
                    _log_mean(f"diagnostics/ztautau_back_to_back/best/{name}", best_values)
        if isinstance(src, ComponentNormalizedTruthDistanceReward):
            comps = src.last_component_errors()
            if comps is None:
                continue
            for cname, ctensor in comps.items():
                if vb.sum() == 0:
                    out[f"components/{cname}/mean"] = float("nan")
                    continue
                cv = ctensor[:, vb]
                # Negate so the sign matches ``reward/raw/*`` (per-component reward, not error).
                out[f"components/{cname}/mean"] = float((-cv).mean().detach().cpu())
            if vb.sum() > 0:
                axis_pairs = _reward_component_axis_pairs(tuple(comps.keys()))
                for axis, (a, b) in axis_pairs.items():
                    axis_reward = -(comps[a] + comps[b])[:, vb]
                    out[f"diagnostics/reward_hacking/all/{axis}/reward_mean"] = float(
                        axis_reward.mean().detach().cpu()
                    )
                    out[f"diagnostics/reward_hacking/best/{axis}/reward_mean"] = float(
                        axis_reward[best_k, cols].mean().detach().cpu()
                    )
                profile_tensors: dict[str, tuple[Tensor, Tensor]] = {}
                deltas = src.last_component_deltas()
                truths = src.last_component_truths()
                if deltas is not None:
                    for axis, (a, b) in axis_pairs.items():
                        all_delta = torch.stack((deltas[a], deltas[b]), dim=-1)[:, vb]
                        best_delta = all_delta[best_k, cols, :]
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/all/{axis}",
                            all_delta,
                        )
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/best/{axis}",
                            best_delta,
                        )
                    if truths is not None:
                        for axis, (a, b) in axis_pairs.items():
                            truth_axis = torch.stack((truths[a], truths[b]), dim=-1)
                            delta_axis = torch.stack((deltas[a], deltas[b]), dim=-1)
                            profile_tensors[axis] = (truth_axis, delta_axis)
                kin_deltas = src.last_kinematic_deltas()
                if kin_deltas is not None:
                    rel_pt = kin_deltas.get("rel_pt")
                    feature_metric_names = tuple(
                        key
                        for key in kin_deltas.keys()
                        if not key.startswith("truth_") and key != "rel_pt"
                    )
                    for name in feature_metric_names:
                        tensor = kin_deltas.get(name)
                        all_delta = tensor[:, vb]
                        best_delta = all_delta[best_k, cols, :]
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/all/{name}",
                            all_delta,
                        )
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/best/{name}",
                            best_delta,
                        )
                        truth_tensor = kin_deltas.get(f"truth_{name}")
                        if truth_tensor is not None:
                            profile_tensors[name] = (truth_tensor, tensor)
                    if rel_pt is not None:
                        all_rel_pt = rel_pt[:, vb].reshape(-1)
                        all_rel_pt = all_rel_pt[torch.isfinite(all_rel_pt)]
                        t_sel = rel_pt[:, vb, :]
                        best_rel_pt = t_sel[best_k, cols, :].reshape(-1)
                        best_rel_pt = best_rel_pt[torch.isfinite(best_rel_pt)]
                        if all_rel_pt.numel() > 0:
                            out["diagnostics/reward_hacking/all/rel_pt/mean"] = float(
                                all_rel_pt.mean().detach().cpu()
                            )
                            out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = float(
                                all_rel_pt.abs().mean().detach().cpu()
                            )
                        else:
                            out["diagnostics/reward_hacking/all/rel_pt/mean"] = float("nan")
                            out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = float("nan")
                        if best_rel_pt.numel() > 0:
                            out["diagnostics/reward_hacking/best/rel_pt/mean"] = float(
                                best_rel_pt.mean().detach().cpu()
                            )
                            out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = float(
                                best_rel_pt.abs().mean().detach().cpu()
                            )
                        else:
                            out["diagnostics/reward_hacking/best/rel_pt/mean"] = float("nan")
                            out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = float("nan")
                    if log_distribution or collect_profile_accum:
                        try:
                            import wandb  # noqa: F401

                            if log_distribution and "rel_pt_dist" in plot_names:
                                out["diagnostics/reward_hacking/dist/rel_pt"] = (
                                    _rel_pt_distribution_figure(
                                        all_rel_pt.detach().float().cpu().numpy(),
                                        best_rel_pt.detach().float().cpu().numpy(),
                                    )
                                )
                            pt_delta = kin_deltas.get("pt")
                            truth_pt = kin_deltas.get("truth_pt")
                            if pt_delta is not None and truth_pt is not None:
                                pt_all = pt_delta[:, vb]
                                truth_all = truth_pt[:, vb, :]
                                if truth_all.shape[0] == 1 and pt_all.shape[0] > 1:
                                    truth_all = truth_all.expand_as(pt_all)
                                best_pt_delta = pt_all[best_k, cols, :]
                                best_truth_pt = truth_all[best_k, cols, :]
                                pt_oracle_k = pt_all.abs().sum(dim=-1).argmin(dim=0)
                                oracle_pt_delta = pt_all[pt_oracle_k, cols, :]
                                oracle_truth_pt = truth_all[pt_oracle_k, cols, :]
                                _log_mean_abs(
                                    "diagnostics/reward_hacking/pt_oracle/pt",
                                    oracle_pt_delta,
                                )
                                prefix_k = min(10, int(pt_all.shape[0]))
                                pt_prefix = pt_all[:prefix_k]
                                truth_prefix = truth_all[:prefix_k]
                                rewards_prefix = rewards_v[:prefix_k]
                                prefix_reward_k = rewards_prefix.argmax(dim=0)
                                prefix_cols = torch.arange(
                                    int(prefix_reward_k.numel()),
                                    device=rewards.device,
                                    dtype=torch.long,
                                )
                                prefix_reward_delta = pt_prefix[
                                    prefix_reward_k, prefix_cols, :
                                ]
                                prefix_reward_truth = truth_prefix[
                                    prefix_reward_k, prefix_cols, :
                                ]
                                prefix_oracle_k = pt_prefix.abs().sum(dim=-1).argmin(dim=0)
                                prefix_oracle_delta = pt_prefix[
                                    prefix_oracle_k, prefix_cols, :
                                ]
                                prefix_oracle_truth = truth_prefix[
                                    prefix_oracle_k, prefix_cols, :
                                ]
                                prefix_reward_mask = (
                                    torch.isfinite(prefix_reward_truth)
                                    & torch.isfinite(prefix_reward_delta)
                                )
                                prefix_oracle_mask = (
                                    torch.isfinite(prefix_oracle_truth)
                                    & torch.isfinite(prefix_oracle_delta)
                                )
                                all_mask = torch.isfinite(truth_all) & torch.isfinite(pt_all)
                                best_mask = torch.isfinite(best_truth_pt) & torch.isfinite(best_pt_delta)
                                oracle_mask = (
                                    torch.isfinite(oracle_truth_pt)
                                    & torch.isfinite(oracle_pt_delta)
                                )
                                if collect_profile_accum:
                                    out[_diag_profile_raw_key("pt", "truth_all")] = (
                                        truth_all[all_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "delta_all")] = (
                                        pt_all[all_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "truth_best")] = (
                                        best_truth_pt[best_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "delta_best")] = (
                                        best_pt_delta[best_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "truth_oracle")] = (
                                        oracle_truth_pt[oracle_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "delta_oracle")] = (
                                        oracle_pt_delta[oracle_mask].detach().float().cpu().numpy()
                                    )
                                if (
                                    log_distribution
                                    and "pt_first10_vs_fullK" in plot_names
                                    and int(pt_all.shape[0]) > prefix_k
                                ):
                                    out[
                                        "diagnostics/reward_hacking/profile/pt_delta_first10_vs_fullK"
                                    ] = _pt_delta_prefix_vs_full_figure(
                                        prefix_reward_truth[prefix_reward_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_reward_delta[prefix_reward_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        best_truth_pt[best_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        best_pt_delta[best_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_oracle_truth[prefix_oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_oracle_delta[prefix_oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        oracle_truth_pt[oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        oracle_pt_delta[oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_k=prefix_k,
                                        full_k=int(pt_all.shape[0]),
                                        title=(
                                            "pT response: first 10 candidates vs full "
                                            f"{int(pt_all.shape[0])}"
                                        ),
                                    )
                                if log_distribution and "pt_profile" in plot_names:
                                    out[
                                        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt"
                                    ] = _pt_delta_selection_profiles_figure(
                                        truth_all.detach().float().cpu().numpy(),
                                        pt_all.detach().float().cpu().numpy(),
                                        best_truth_pt.detach().float().cpu().numpy(),
                                        best_pt_delta.detach().float().cpu().numpy(),
                                        oracle_truth_pt.detach().float().cpu().numpy(),
                                        oracle_pt_delta.detach().float().cpu().numpy(),
                                        title="Reward selection pT bias vs truth pT",
                                    )
                            profile_names = tuple(
                                name for name in profile_tensors.keys() if name != "pt"
                            )
                            for profile_name in profile_names:
                                if not log_distribution or f"{profile_name}_profile" not in plot_names:
                                    continue
                                tensors = profile_tensors.get(profile_name)
                                if tensors is None:
                                    continue
                                truth_tensor, delta_tensor = tensors
                                truth_tensor = _align_truth_tensor_to_delta(
                                    truth_tensor, delta_tensor
                                )
                                profile_delta_all = delta_tensor[:, vb]
                                profile_truth_all = truth_tensor[:, vb]
                                profile_best_delta = profile_delta_all[best_k, cols, :]
                                profile_best_truth = profile_truth_all[best_k, cols, :]
                                profile_oracle_k = profile_delta_all.abs().sum(dim=-1).argmin(dim=0)
                                profile_oracle_delta = profile_delta_all[
                                    profile_oracle_k, cols, :
                                ]
                                profile_oracle_truth = profile_truth_all[
                                    profile_oracle_k, cols, :
                                ]
                                _log_mean_abs(
                                    f"diagnostics/reward_hacking/{profile_name}_oracle/{profile_name}",
                                    profile_oracle_delta,
                                )
                                truth_all_np, delta_all_np = _finite_profile_numpy(
                                    profile_truth_all, profile_delta_all
                                )
                                truth_best_np, delta_best_np = _finite_profile_numpy(
                                    profile_best_truth, profile_best_delta
                                )
                                truth_oracle_np, delta_oracle_np = _finite_profile_numpy(
                                    profile_oracle_truth, profile_oracle_delta
                                )
                                raw_items = {
                                    "truth_all": truth_all_np,
                                    "delta_all": delta_all_np,
                                    "truth_best": truth_best_np,
                                    "delta_best": delta_best_np,
                                    "truth_oracle": truth_oracle_np,
                                    "delta_oracle": delta_oracle_np,
                                }
                                for suffix, arr in raw_items.items():
                                    out[_diag_profile_raw_key(profile_name, suffix)] = arr
                                out[_diag_profile_log_key(profile_name)] = (
                                    _delta_selection_profiles_figure(
                                        truth_all_np,
                                        delta_all_np,
                                        truth_best_np,
                                        delta_best_np,
                                        truth_oracle_np,
                                        delta_oracle_np,
                                        profile_name=profile_name,
                                        title=_diag_profile_title(profile_name),
                                    )
                                )
                        except Exception:
                            pass
            else:
                nan_f = float("nan")
                axis_pairs = _reward_component_axis_pairs(tuple(comps.keys()))
                kin_deltas = src.last_kinematic_deltas() or {}
                feature_metric_names = tuple(
                    key for key in kin_deltas.keys() if not key.startswith("truth_") and key != "rel_pt"
                )
                for scope in ("all", "best"):
                    for axis in axis_pairs:
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/reward_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/delta_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/delta_abs_mean"] = nan_f
                    for name in feature_metric_names:
                        out[f"diagnostics/reward_hacking/{scope}/{name}/delta_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{name}/delta_abs_mean"] = nan_f
                if kin_deltas.get("rel_pt") is not None:
                    out["diagnostics/reward_hacking/all/rel_pt/mean"] = nan_f
                    out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = nan_f
                    out["diagnostics/reward_hacking/best/rel_pt/mean"] = nan_f
                    out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = nan_f
            break

    return out


def _append_projection_constraint_panel_metrics(out: dict[str, float]) -> None:
    """Populate ``projection/constraint/*`` for the dedicated W&B projection panel."""
    pure = out.get("projection/active")
    if pure is None or not math.isfinite(float(pure)) or float(pure) != 1.0:
        return
    prefix = "latent_constraint/"
    for key, val in list(out.items()):
        if isinstance(key, str) and key.startswith(prefix):
            out[f"projection/constraint/{key[len(prefix):]}"] = val


def _append_swd_panel_metrics(out: dict[str, float]) -> None:
    """Populate ``swd/*`` for the dedicated W&B latent-SWD monitoring panel.

    Sources ``latent_constraint/*`` (raw diag) or ``projection/latent_*`` (mapped aliases).
    Only populated when the frozen latent encoder is the active projection backend.
    """
    src_map = {
        "swd/pred_truth": (
            "latent_constraint/swd_pred_truth",
            "projection/latent_swd_pred_truth",
        ),
        "swd/truth_truth": (
            "latent_constraint/swd_truth_truth",
            "projection/latent_swd_truth_truth",
        ),
        "swd/ratio": (
            "latent_constraint/swd_ratio",
            "projection/latent_swd_ratio",
        ),
        "swd/C_norm": (
            "latent_constraint/C_norm",
            "projection/latent_C_norm",
        ),
        "swd/mask_count": (
            "latent_constraint/mask_count",
            "projection/latent_mask_count",
        ),
        "swd/skipped_small_mask": ("latent_constraint/skipped_small_mask",),
    }
    populated = False
    for dst, src_keys in src_map.items():
        for src in src_keys:
            val = out.get(src)
            if val is None:
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                out[dst] = fv
                populated = True
                break
    if populated:
        out["swd/active"] = 1.0
        skipped = out.get("swd/skipped_small_mask")
        if skipped is not None and float(skipped) >= 1.0:
            out["swd/active"] = 0.0


def _append_projection_summary_metrics(out: dict[str, Any]) -> None:
    """Add ``projection/summary/C_projected_minus_old`` for the W&B projection panel."""
    c_projected = out.get("projection/C_projected")
    c_old = out.get("projection/C_old", out.get("projection/C_raw"))
    if c_old is not None and c_projected is not None:
        try:
            out["projection/summary/C_projected_minus_old"] = float(c_projected) - float(c_old)
        except (TypeError, ValueError):
            pass

@torch.no_grad()
def _build_train_metrics(
    diag_last: dict[str, Tensor],
    rewards: Tensor,
    valid_b: Tensor,
    advantages: Tensor | None = None,
) -> dict[str, float]:
    """Training panels ``reward/monitor/*``, ``train/loss/*``, and ``parameter/*`` for Weights & Biases."""
    param = _parameter_panel_from_diag(diag_last)
    vb = valid_b.reshape(-1) > 0
    adv_gap = (
        compute_reward_advantage_pos_neg_gap(rewards, advantages, valid_b)
        if advantages is not None
        else float("nan")
    )
    if vb.sum() == 0:
        nan = float("nan")
        base: dict[str, float] = {
            "train/loss/total": float(diag_last["loss_total"].cpu()),
            "train/loss/velocity": _diag_scalar_float(
                diag_last, "loss_velocity_training"
            ),
            "train/loss/dgpo": float(diag_last["loss_main"].cpu()),
            "train/loss/kl": _diag_scalar_float(diag_last, "train/loss/kl"),
            "train/loss/L_cur": float(diag_last["L_cur_mean"].cpu()),
            "train/loss/L_ref": float(diag_last["L_ref_mean"].cpu()),
            "train/loss/delta": float(diag_last["delta_abs_mean"].cpu()),
            "reward/monitor/best_of_k": nan,
            "reward/monitor/median": nan,
            "reward/monitor/mean_gap": nan,
            "reward/monitor/last_place": nan,
            "reward/monitor/p10": nan,
            "reward/monitor/p30": nan,
            "reward/monitor/p70": nan,
            "reward/monitor/p90": nan,
            "reward/monitor/advantage_pos_neg_gap": adv_gap,
        }
        if not math.isfinite(base["train/loss/velocity"]):
            base["train/loss/velocity"] = float(diag_last["loss_total"].cpu())
        base.update(param)
        for dk, dv in diag_last.items():
            if isinstance(dk, str) and (
                dk.startswith(("projection/", "latent_constraint/", "train/regularization/"))
                or dk == "train/loss/variance_regularization"
            ):
                base[dk] = float(dv.detach().float().cpu())
        return base

    r = rewards[:, vb]
    K = r.shape[0]
    best_of_k = float(r.max(dim=0).values.mean().cpu())
    median_e = float(r.median(dim=0).values.mean().cpu())
    last_place = float(r.min(dim=0).values.mean().cpu())
    p10 = float(r.quantile(0.1, dim=0).mean().cpu()) if K >= 2 else last_place
    p30 = float(r.quantile(0.3, dim=0).mean().cpu()) if K >= 2 else last_place
    p70 = float(r.quantile(0.7, dim=0).mean().cpu()) if K >= 2 else last_place
    p90 = float(r.quantile(0.9, dim=0).mean().cpu()) if K >= 2 else last_place
    mean_gap = compute_reward_mean_gap(rewards, valid_b)

    lv = _diag_scalar_float(diag_last, "loss_velocity_training")
    if not math.isfinite(lv):
        lv = float(diag_last["loss_total"].cpu())
    out: dict[str, float] = {
        "train/loss/total": float(diag_last["loss_total"].cpu()),
        "train/loss/velocity": lv,
        "train/loss/dgpo": float(diag_last["loss_main"].cpu()),
        "train/loss/kl": _diag_scalar_float(diag_last, "train/loss/kl"),
        "train/loss/L_cur": float(diag_last["L_cur_mean"].cpu()),
        "train/loss/L_ref": float(diag_last["L_ref_mean"].cpu()),
        "train/loss/delta": float(diag_last["delta_abs_mean"].cpu()),
        "reward/monitor/best_of_k": best_of_k,
        "reward/monitor/median": median_e,
        "reward/monitor/mean_gap": mean_gap,
        "reward/monitor/last_place": last_place,
        "reward/monitor/p10": p10,
        "reward/monitor/p30": p30,
        "reward/monitor/p70": p70,
        "reward/monitor/p90": p90,
        "reward/monitor/advantage_pos_neg_gap": adv_gap,
    }
    out.update(param)
    for dk, dv in diag_last.items():
        if isinstance(dk, str) and (
            dk.startswith(("projection/", "latent_constraint/", "train/regularization/"))
            or dk == "train/loss/variance_regularization"
        ):
            out[dk] = float(dv.detach().float().cpu())
    return out


def policy_evaluation_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Tensor,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    t: Tensor | None = None,
    eps_rep: Tensor | None = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    dict[str, Any],
    Tensor,
]:
    """Shared ``eps`` per event (when ``shared_noise``).

    Returns:
        ``L_cur``, ``L_ref`` each ``(K, B)``, ``t`` ``(B,)``, ``model_v``, ``ref_v`` each
        ``(K*B, N_nu, F)``, ``noise_mask_rep`` ``(K*B, N_nu, 1)`` for KL anchor MSE,
        ``x_t``, ``target_v``, ``t_rep``, ``batch_rep``, and ``eps_rep`` ``(K*B, N_nu, F)``.
    """
    B = batch["x"].shape[0]
    inv_mask = batch["x_invisible_mask"]
    c_flat = candidates_phys.reshape(K * B, *candidates_phys.shape[2:])
    # Row order matches ``(K, B, ...)`` reshape: all events for k=0, then k=1, ...
    inv_kb = inv_mask.unsqueeze(0).expand(K, -1, -1).reshape(K * B, *inv_mask.shape[1:])
    eve = _unwrap_core_evenet(model)
    c_norm = _normalize_candidates_for_policy(eve, c_flat, inv_kb)

    if t is None:
        t = torch.rand(B, device=device, dtype=torch.float32) * (t_max - t_min) + t_min
    _, alpha, sigma = get_logsnr_alpha_sigma(t, shape=(B, 1, 1))
    alpha_rep = alpha.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, 1, 1).to(dtype)
    sigma_rep = sigma.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, 1, 1).to(dtype)

    N_nu, F_eff = c_norm.shape[1], c_norm.shape[2]
    expected_eps_shape = (K * B, N_nu, F_eff)
    if eps_rep is None:
        if shared_noise:
            eps = torch.randn(B, N_nu, F_eff, device=device, dtype=dtype)
            eps_rep = eps.unsqueeze(0).expand(K, -1, -1, -1).reshape(*expected_eps_shape)
        else:
            eps_rep = torch.randn(*expected_eps_shape, device=device, dtype=dtype)
    else:
        eps_rep = eps_rep.to(device=device, dtype=dtype)
        if tuple(eps_rep.shape) != expected_eps_shape:
            raise ValueError(
                f"policy_evaluation_step eps_rep must have shape {expected_eps_shape}, "
                f"got {tuple(eps_rep.shape)}"
            )

    x_t = alpha_rep * c_norm + sigma_rep * eps_rep
    target_v = alpha_rep * eps_rep - sigma_rep * c_norm

    t_rep = t.unsqueeze(0).expand(K, -1).reshape(K * B)
    batch_rep = repeat_batch_for_candidates(batch, K)
    noise_mask_rep = batch_rep["x_invisible_mask"].unsqueeze(-1)

    if isinstance(model, DDP):
        model_v = model(x_t, batch_rep, t_rep, noise_mask_rep)
    else:
        model_v = model.predict_diffusion_vector(
            noise_x=x_t,
            cond_x=batch_rep,
            time=t_rep,
            mode="neutrino",
            noise_mask=noise_mask_rep,
        )
    # ``predict_diffusion_vector`` already strips invisible_padding from its output,
    # so ``model_v`` and ``target_v`` share the same width (``invisible_input_dim``).
    L_cur = per_row_velocity_mse(model_v, target_v, noise_mask_rep, invisible_padding=0)

    with torch.no_grad():
        ref_v = ref_model.predict_diffusion_vector(
            noise_x=x_t,
            cond_x=batch_rep,
            time=t_rep,
            mode="neutrino",
            noise_mask=noise_mask_rep,
        )
        L_ref = per_row_velocity_mse(ref_v, target_v, noise_mask_rep, invisible_padding=0)

    return (
        L_cur.reshape(K, B),
        L_ref.reshape(K, B),
        t,
        model_v,
        ref_v,
        noise_mask_rep,
        x_t,
        target_v,
        t_rep,
        batch_rep,
        eps_rep,
    )


class _DgpoOptimizerWithSchedule:
    """Bundle ``(AdamW, LambdaLR)`` for DGPO: ``scheduler_step()`` once per batch.

    The train loop performs a single ``optimizer.step()`` per batch (sub-step
    gradients are accumulated).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.step(*args, **kwargs)

    def scheduler_step(self) -> None:
        self.scheduler.step()

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @property
    def state(self) -> dict[Any, Any]:
        return self.optimizer.state

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Any) -> None:
        if isinstance(state, dict) and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            return
        try:
            self.optimizer.load_state_dict(state)
        except (ValueError, RuntimeError):
            raise
        _log.warning(
            "[DGPO] Loaded legacy optimizer state dict without scheduler keys; "
            "LambdaLR keeps its current step counter (may be out of sync with resume step).",
        )


def _latent_swd_step_seed(base: int | None, sample_idx: int) -> int | None:
    """Per-(step, multi-sample) common-random-numbers seed for the latent SWD.

    Returns ``None`` (legacy per-call randomness) when ``base`` is ``None`` so any
    non-seeded caller is unaffected. The combination is stable within a CPO repair
    step (same ``base``) but advances across global steps so the projection space
    is still covered over training.
    """
    if base is None:
        return None
    return (int(base) * 1000003 + int(sample_idx)) & 0x7FFFFFFF


def _compute_projection_constraint_raw(
    *,
    constraint_state: ProjectionConstraintState,
    model_v: Tensor,
    x_t: Tensor,
    t_rep: Tensor,
    noise_mask_rep: Tensor,
    batch_kb: dict[str, Any],
    core_model: torch.nn.Module,
    cartesian: bool,
    K: int,
    candidate_weights_kb: Tensor | None,
    update_ema: bool = True,
    world_size: int = 1,
    constraint_seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Batch constraint scalar ``C_B`` from the frozen latent-SWD encoder.

    ``constraint_seed`` enables common random numbers for the stochastic SWD
    estimator (fixed projections + null split across the CPO repair's repeated
    evaluations within one step).
    """
    return compute_latent_swd_constraint(
        model_v=model_v,
        x_t=x_t,
        t_rep=t_rep,
        noise_mask_rep=noise_mask_rep,
        batch_kb=batch_kb,
        core_model=core_model,
        cartesian=cartesian,
        K=K,
        state=constraint_state,
        candidate_weights_kb=candidate_weights_kb,
        update_ema=update_ema,
        world_size=world_size,
        seed=constraint_seed,
    )


def _latent_projection_metrics(off_diag: Mapping[str, Tensor]) -> dict[str, float]:
    """Map latent-SWD constraint diagnostics (``latent_constraint/*``) to ``projection/*`` scalars."""
    key_map = {
        "projection/latent_swd_pred_truth": "latent_constraint/swd_pred_truth",
        "projection/latent_swd_truth_truth": "latent_constraint/swd_truth_truth",
        "projection/latent_swd_ratio": "latent_constraint/swd_ratio",
        "projection/latent_C_norm": "latent_constraint/C_norm",
        "projection/latent_mask_count": "latent_constraint/mask_count",
    }
    out: dict[str, float] = {}
    for dst, src in key_map.items():
        val = off_diag.get(src)
        if isinstance(val, Tensor) and int(val.numel()) == 1:
            fv = float(val.detach().reshape(-1)[0].cpu())
            if math.isfinite(fv):
                out[dst] = fv
    return out


def _projection_constraint_forward(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    t: Tensor | None = None,
    eps_rep: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    """Run policy evaluation and return tensors needed for detached constraint measurement."""
    (
        _L_cur,
        _L_ref,
        t_out,
        model_v,
        _ref_v,
        noise_mask_rep,
        x_t,
        _target_v,
        t_rep,
        batch_rep,
        eps_out,
    ) = policy_evaluation_step(
        model,
        ref_model,
        batch,
        candidates_phys,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        t=t,
        eps_rep=eps_rep,
        t_min=policy_eval_t_min,
        t_max=policy_eval_t_max,
    )
    return model_v, x_t, t_rep, noise_mask_rep, batch_rep


@torch.no_grad()
def _projection_constraint_C_detached(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    candidate_weights_kb: Tensor | None,
    proj_cfg: ProjectionConstraintConfig,
    constraint_state: ProjectionConstraintState,
    t: Tensor,
    eps_rep: Tensor,
    world_size: int = 1,
    constraint_seed: int | None = None,
) -> tuple[float, dict[str, Tensor]]:
    """Evaluate detached ``C_B`` at current model weights with frozen policy-eval randomness."""
    core = _unwrap_core_evenet(model)
    model_v, x_t, t_rep, noise_mask_rep, batch_rep = _projection_constraint_forward(
        model,
        ref_model,
        batch,
        candidates_phys,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        policy_eval_t_min=policy_eval_t_min,
        policy_eval_t_max=policy_eval_t_max,
        t=t,
        eps_rep=eps_rep,
    )
    C_raw_t, off_diag = _compute_projection_constraint_raw(
        constraint_state=constraint_state,
        model_v=model_v,
        x_t=x_t,
        t_rep=t_rep,
        noise_mask_rep=noise_mask_rep,
        batch_kb=batch_rep,
        core_model=core,
        cartesian=_truth_generation_cartesian(),
        K=K,
        candidate_weights_kb=candidate_weights_kb,
        update_ema=True,
        world_size=world_size,
        constraint_seed=constraint_seed,
    )
    return float(C_raw_t.detach().float().cpu()), off_diag


@torch.no_grad()
def _projection_constraint_C_detached_average(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    candidate_weights_kb: Tensor | None,
    proj_cfg: ProjectionConstraintConfig,
    constraint_state: ProjectionConstraintState,
    frozen_eval_inputs: list[tuple[Tensor, Tensor, int | None]],
    world_size: int = 1,
) -> tuple[float, dict[str, Tensor], dict[str, float]]:
    """Average detached ``C_B`` over frozen policy-eval ``(t, eps, seed)`` triples.

    The per-draw ``seed`` is reused from the ``theta_old`` evaluation so each
    post-AdamW proxy uses the **same** SWD projections + null split as its matching
    ``theta_old`` draw (common random numbers); only the parameters differ.
    """
    if not frozen_eval_inputs:
        raise ValueError("frozen_eval_inputs must be non-empty for detached constraint averaging")
    c_vals: list[float] = []
    off_diag_first: dict[str, Tensor] = {}
    for t_frozen, eps_frozen, seed_frozen in frozen_eval_inputs:
        c_i, off_diag = _projection_constraint_C_detached(
            model,
            ref_model,
            batch,
            candidates_phys,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            policy_eval_t_min=policy_eval_t_min,
            policy_eval_t_max=policy_eval_t_max,
            candidate_weights_kb=candidate_weights_kb,
            proj_cfg=proj_cfg,
            constraint_state=constraint_state,
            t=t_frozen,
            eps_rep=eps_frozen,
            world_size=world_size,
            constraint_seed=seed_frozen,
        )
        c_vals.append(float(c_i))
        if not off_diag_first:
            off_diag_first = off_diag
    c_mean = float(sum(c_vals) / len(c_vals))
    proxy_diag: dict[str, float] = {
        "projection/direct_post_adam_proxy/C_mean": c_mean,
    }
    if len(c_vals) > 1:
        proxy_diag["projection/direct_post_adam_proxy/C_std"] = float(
            torch.tensor(c_vals, dtype=torch.float64).std(unbiased=False).cpu()
        )
        proxy_diag["projection/direct_post_adam_proxy/C_min"] = float(min(c_vals))
        proxy_diag["projection/direct_post_adam_proxy/C_max"] = float(max(c_vals))
    else:
        proxy_diag["projection/direct_post_adam_proxy/C_std"] = 0.0
        proxy_diag["projection/direct_post_adam_proxy/C_min"] = c_mean
        proxy_diag["projection/direct_post_adam_proxy/C_max"] = c_mean
    return c_mean, off_diag_first, proxy_diag


def _projection_grad_debug(model: torch.nn.Module) -> str:
    """One-line diagnostic for why the projection constraint lost its grad graph."""
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    inf_mode = getattr(torch, "is_inference_mode_enabled", lambda: "n/a")()
    return (
        f"[grad_debug grad_enabled={torch.is_grad_enabled()} "
        f"inference_mode={inf_mode} "
        f"trainable_params={n_trainable}/{n_total}]"
    )


def _projection_constraint_value_and_grad_at_theta_old(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    candidate_weights_kb: Tensor | None,
    optimizer: Any,
    proj_cfg: ProjectionConstraintConfig,
    constraint_state: ProjectionConstraintState,
    world_size: int = 1,
    constraint_seed_base: int | None = None,
) -> tuple[Tensor, float, dict[str, Tensor], list[tuple[Tensor, Tensor, int | None]], dict[str, float]]:
    """Evaluate ``C_raw`` and flat ``b = nabla C`` at parameters already set to ``theta_old``.

    When ``(int(proj_cfg.multi_sample_count) > 1)`` is true, averages ``M`` independent
    policy-eval ``(t, eps)`` draws: ``C_bar = mean_i C_i``, ``b = grad C_bar``.
    Gradients are accumulated sequentially via ``(C_i / M).backward()`` to avoid
    retaining ``M`` full autograd graphs simultaneously.

    Returns all frozen ``(t, eps)`` pairs used for ``C_bar`` so post-Adam proxy
    evaluation can reuse the same common randomness.
    """
    core = _unwrap_core_evenet(model)
    ms_count = int(proj_cfg.multi_sample_count) if (int(proj_cfg.multi_sample_count) > 1) else 1
    ms_diag: dict[str, float] = {
        "projection/multi_sample/enabled": 1.0 if (int(proj_cfg.multi_sample_count) > 1) else 0.0,
        "projection/multi_sample/samples": float(ms_count),
        "projection/multi_sample/t_sampling_stratified": 0.0,
    }

    def _one_shot_constraint(
        t_batch: Tensor | None = None,
        constraint_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        # The post-AdamW CPO repair backpropagates through ``C_raw`` w.r.t. the policy
        # parameters, so this forward MUST be differentiable. Force grad tracking here
        # (``inference_mode(False)`` + ``enable_grad``) so an ambient inference_mode /
        # no_grad context inherited from the Ray/Lightning rollout or eval utilities
        # cannot silently yield a non-differentiable constraint and crash the backward.
        # Same guard pattern as ``DDIMSampler.sample_with_log_prob``.
        with torch.inference_mode(False), torch.enable_grad():
            (
                _L_cur,
                _L_ref,
                t_frozen,
                model_v,
                _ref_v,
                noise_mask_rep,
                x_t,
                _target_v,
                t_rep,
                batch_rep,
                eps_frozen,
            ) = policy_evaluation_step(
                model,
                ref_model,
                batch,
                candidates_phys,
                K=K,
                shared_noise=shared_noise,
                device=device,
                dtype=dtype,
                t=t_batch,
                eps_rep=None,
                t_min=policy_eval_t_min,
                t_max=policy_eval_t_max,
            )
            C_raw_t, off_diag = _compute_projection_constraint_raw(
                constraint_state=constraint_state,
                model_v=model_v,
                x_t=x_t,
                t_rep=t_rep,
                noise_mask_rep=noise_mask_rep,
                batch_kb=batch_rep,
                core_model=core,
                cartesian=_truth_generation_cartesian(),
                K=K,
                candidate_weights_kb=candidate_weights_kb,
                update_ema=True,
                world_size=world_size,
                constraint_seed=constraint_seed,
            )
        return C_raw_t, off_diag, t_frozen.detach(), eps_frozen.detach()

    if not (int(proj_cfg.multi_sample_count) > 1) or ms_count <= 1:
        seed_0 = _latent_swd_step_seed(constraint_seed_base, 0)
        C_raw_t, off_diag, t_frozen, eps_frozen = _one_shot_constraint(
            constraint_seed=seed_0
        )
        optimizer.zero_grad(set_to_none=True)
        if C_raw_t.requires_grad:
            C_raw_t.backward()
        else:
            _log.warning(
                "[DGPO] projection constraint C_raw is non-differentiable "
                "(requires_grad=False); skipping CPO repair this step. b=0. %s",
                _projection_grad_debug(model),
            )
        # ``b`` is already DDP-averaged by this backward (DDP forward, no ``no_sync``);
        # only the scalar ``C`` needs explicit cross-rank averaging.
        b_flat = flatten_param_grads(model)
        C_raw = float(C_raw_t.detach().float().cpu())
        C_raw = sync_projection_constraint_C_across_ranks(
            C_raw, device=b_flat.device, world_size=world_size,
        )
        ms_diag["projection/multi_sample/C_mean"] = C_raw
        ms_diag["projection/multi_sample/C_std"] = 0.0
        ms_diag["projection/multi_sample/C_min"] = C_raw
        ms_diag["projection/multi_sample/C_max"] = C_raw
        frozen_eval_inputs = [(t_frozen, eps_frozen, seed_0)]
        return b_flat, C_raw, off_diag, frozen_eval_inputs, ms_diag

    off_diag_first: dict[str, Tensor] = {}
    frozen_eval_inputs: list[tuple[Tensor, Tensor, int | None]] = []
    C_vals: list[float] = []
    inv_m = 1.0 / float(ms_count)
    optimizer.zero_grad(set_to_none=True)
    use_stratified_t = (
        True and ms_count > 1
    )
    if use_stratified_t:
        batch_size = int(batch["x"].shape[0])
        t_grid = projection_stratified_t_grid(
            ms_count,
            batch_size,
            t_min=policy_eval_t_min,
            t_max=policy_eval_t_max,
            device=device,
            dtype=dtype,
        )
        ms_diag["projection/multi_sample/t_sampling_stratified"] = 1.0
        ms_diag["projection/multi_sample/t_grid_first"] = float(t_grid[0][0].detach().cpu())
        ms_diag["projection/multi_sample/t_grid_last"] = float(t_grid[-1][0].detach().cpu())
    else:
        t_grid = [None] * ms_count
    for sample_idx in range(ms_count):
        seed_i = _latent_swd_step_seed(constraint_seed_base, sample_idx)
        C_raw_t, off_diag, t_frozen, eps_frozen = _one_shot_constraint(
            t_grid[sample_idx], constraint_seed=seed_i
        )
        C_vals.append(float(C_raw_t.detach().float().cpu()))
        frozen_eval_inputs.append((t_frozen, eps_frozen, seed_i))
        if not off_diag_first:
            off_diag_first = off_diag
        # Sequential (C_i / M).backward() accumulates grad C_bar without retaining M graphs.
        if C_raw_t.requires_grad:
            (C_raw_t * inv_m).backward()
        elif sample_idx == 0:
            _log.warning(
                "[DGPO] projection constraint C_raw is non-differentiable "
                "(requires_grad=False); skipping CPO repair this step. b=0. %s",
                _projection_grad_debug(model),
            )

    # ``b`` is already DDP-averaged across the per-sample backward passes (DDP forward,
    # no ``no_sync``); only the scalar ``C`` needs explicit cross-rank averaging.
    b_flat = flatten_param_grads(model)
    C_mean = float(sum(C_vals) / len(C_vals))
    C_mean = sync_projection_constraint_C_across_ranks(
        C_mean, device=b_flat.device, world_size=world_size,
    )
    if len(C_vals) > 1:
        C_std = float(torch.tensor(C_vals, dtype=torch.float64).std(unbiased=False).cpu())
    else:
        C_std = 0.0
    ms_diag["projection/multi_sample/C_mean"] = C_mean
    ms_diag["projection/multi_sample/C_std"] = C_std
    ms_diag["projection/multi_sample/C_min"] = float(min(C_vals))
    ms_diag["projection/multi_sample/C_max"] = float(max(C_vals))
    if len(C_vals) >= 1:
        ms_diag["projection/multi_sample/C_first"] = float(C_vals[0])
    return b_flat, C_mean, off_diag_first, frozen_eval_inputs, ms_diag


@dataclass
class _ProjectionRepairEstimate:
    """Resolved linear CPO constraint estimator for one projection repair step."""

    b_flat: Tensor
    C_selected: float
    c_margin: float
    violation_for_lambda: float
    b_dot_d0: float
    C_old: float = float("nan")
    C_adam_pred: float = float("nan")
    C_adam_proxy: float = float("nan")
    v_linear: float = float("nan")
    v_direct: float = float("nan")
    linearization_error: float = float("nan")
    b_proxy_d: Tensor | None = None
    off_diag_old: dict[str, Tensor] = field(default_factory=dict)
    off_diag_adam_pre: dict[str, Tensor] = field(default_factory=dict)
    ms_diag: dict[str, float] = field(default_factory=dict)
    proxy_diag: dict[str, float] = field(default_factory=dict)
    proxy_ok: bool = True
    frozen_eval_inputs: list[tuple[Tensor, Tensor, int | None]] = field(default_factory=list)


def _projection_repair_proxy_estimator(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    theta_old: dict[str, Tensor],
    theta_adam: dict[str, Tensor],
    proj_cfg: ProjectionConstraintConfig,
    candidate_weights_kb: Tensor | None,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    optimizer: Any,
    constraint_state: ProjectionConstraintState,
    world_size: int = 1,
    constraint_seed_base: int | None = None,
) -> _ProjectionRepairEstimate:
    """Linear post-Adam CPO estimator: ``v_linear = (C_old + b^T delta0) - epsilon``."""
    eps_v = float(proj_cfg.epsilon)
    d0 = flatten_param_delta(theta_adam, theta_old).to(dtype=torch.float64)

    assign_params_(model, theta_old)
    optimizer.zero_grad(set_to_none=True)
    b_proxy_flat, C_old, off_diag_old, frozen_eval_inputs, ms_diag = (
        _projection_constraint_value_and_grad_at_theta_old(
            model,
            ref_model,
            batch,
            candidates_phys,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            policy_eval_t_min=policy_eval_t_min,
            policy_eval_t_max=policy_eval_t_max,
            candidate_weights_kb=candidate_weights_kb,
            optimizer=optimizer,
            proj_cfg=proj_cfg,
            constraint_state=constraint_state,
            world_size=world_size,
            constraint_seed_base=constraint_seed_base,
        )
    )
    b_proxy_d = b_proxy_flat.to(dtype=torch.float64)
    b_proxy_dot_d0 = float(torch.dot(b_proxy_d, d0).detach().cpu())
    C_adam_pred = float(C_old) + b_proxy_dot_d0
    v_linear = C_adam_pred - eps_v

    assign_params_(model, theta_adam)
    C_adam_proxy, off_diag_adam_pre, proxy_diag = _projection_constraint_C_detached_average(
        model,
        ref_model,
        batch,
        candidates_phys,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        policy_eval_t_min=policy_eval_t_min,
        policy_eval_t_max=policy_eval_t_max,
        candidate_weights_kb=candidate_weights_kb,
        proj_cfg=proj_cfg,
        constraint_state=constraint_state,
        frozen_eval_inputs=frozen_eval_inputs,
        world_size=world_size,
    )
    v_direct = float(C_adam_proxy) - eps_v
    linearization_error = float(C_adam_proxy) - C_adam_pred
    proxy_diag["projection/direct_post_adam_proxy/active"] = 1.0

    return _ProjectionRepairEstimate(
        b_flat=b_proxy_flat,
        C_selected=float(C_old),
        c_margin=float(C_old) - eps_v,
        violation_for_lambda=v_linear,
        b_dot_d0=b_proxy_dot_d0,
        C_old=float(C_old),
        C_adam_pred=C_adam_pred,
        C_adam_proxy=float(C_adam_proxy),
        v_linear=v_linear,
        v_direct=v_direct,
        linearization_error=linearization_error,
        b_proxy_d=b_proxy_d,
        off_diag_old=off_diag_old,
        off_diag_adam_pre=off_diag_adam_pre,
        ms_diag=ms_diag,
        proxy_diag=proxy_diag,
        proxy_ok=math.isfinite(float(C_adam_proxy)),
        frozen_eval_inputs=frozen_eval_inputs,
    )


def _dgpo_projection_repair_after_adamw(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    theta_old: dict[str, Tensor],
    proj_cfg: ProjectionConstraintConfig,
    candidate_weights_kb: Tensor | None,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    optimizer: Any,
    diag_last: dict[str, Tensor],
    constraint_state: ProjectionConstraintState,
    world_size: int = 1,
    constraint_seed_base: int | None = None,
) -> None:
    """AdamW-metric CPO projection repair after the unconstrained AdamW step."""
    theta_adam = snapshot_params(model)
    delta0_flat = flatten_param_delta(theta_adam, theta_old)
    delta0_norm = float(torch.linalg.norm(delta0_flat).detach().cpu())
    eps_v = float(proj_cfg.epsilon)

    est = _projection_repair_proxy_estimator(
        model=model,
        ref_model=ref_model,
        batch=batch,
        candidates_phys=candidates_phys,
        theta_old=theta_old,
        theta_adam=theta_adam,
        proj_cfg=proj_cfg,
        candidate_weights_kb=candidate_weights_kb,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        policy_eval_t_min=policy_eval_t_min,
        policy_eval_t_max=policy_eval_t_max,
        optimizer=optimizer,
        constraint_state=constraint_state,
        world_size=world_size,
        constraint_seed_base=constraint_seed_base,
    )

    b_flat = est.b_flat
    C_selected = est.C_selected
    c_margin = est.c_margin
    violation_for_lambda = est.violation_for_lambda
    b_dot_d0 = est.b_dot_d0
    C_old = est.C_old
    C_adam_pred = est.C_adam_pred
    C_adam_proxy = est.C_adam_proxy
    v_linear = est.v_linear
    v_direct = est.v_direct
    linearization_error = est.linearization_error
    b_proxy_d = est.b_proxy_d
    off_diag_old = est.off_diag_old
    off_diag_adam_pre = est.off_diag_adam_pre
    ms_diag = est.ms_diag
    proxy_diag = est.proxy_diag

    b_d = b_flat.to(dtype=torch.float64)
    p_flat, precond_diag = flatten_adam_preconditioned_direction(
        model,
        b_flat,
        optimizer,
        use_adam_preconditioner=proj_cfg.use_adam_preconditioner,
    )

    assign_params_(model, theta_old)

    lam, proj_diag = compute_projection_lambda_from_violation(
        b_flat,
        p_flat,
        violation_for_lambda,
        proj_cfg,
        C_raw=C_selected,
        b_dot_delta0=b_dot_d0,
    )
    proj_diag.update(precond_diag)
    proj_diag.update(ms_diag)
    proj_diag.update(proxy_diag)
    # Reward<->constraint alignment: cosine of the constraint gradient b with the AdamW reward
    # step delta0 (b_dot_d0 = b.delta0). Persistently > 0 means the reward step keeps PUSHING the
    # constraint up -> a genuine reward<->constraint tension (the constraint must fight reward to
    # bind). Near 0 means they are ~orthogonal, so the constraint should not block reward and a
    # reward stall points elsewhere (coordinate artifact / margin).
    _b_norm_val = float(proj_diag.get("projection/b_norm", 0.0))
    proj_diag["projection/reward_constraint_align"] = float(b_dot_d0) / (
        _b_norm_val * float(delta0_norm) + 1e-12
    )
    proj_diag["projection/epsilon"] = float(eps_v)
    proj_diag["projection/C_old"] = float(C_old)
    proj_diag["projection/C_adam_pred"] = float(C_adam_pred)
    proj_diag["projection/C_adam_proxy"] = float(C_adam_proxy)
    proj_diag["projection/v_linear"] = float(v_linear)
    proj_diag["projection/v_direct"] = float(v_direct)
    if b_proxy_d is not None:
        proj_diag["projection/b_proxy_norm2"] = float(
            torch.dot(b_proxy_d, b_proxy_d).detach().cpu()
        )
    else:
        proj_diag["projection/b_proxy_norm2"] = float("nan")
    proj_diag["projection/v_before"] = float(v_linear)
    proj_diag["projection/v_selected"] = float(violation_for_lambda)
    proj_diag["projection/linearization_error"] = float(linearization_error)
    proj_diag["projection/C_linear_adam"] = float(C_adam_pred)
    proj_diag["projection/linearization_error_adam"] = float(linearization_error)
    proj_diag["projection/delta0_norm"] = delta0_norm
    repair_flat = max(0.0, float(lam)) * p_flat
    repair_norm = float(torch.linalg.norm(repair_flat.to(dtype=torch.float64)).detach().cpu())
    delta_projected_flat = delta0_flat - repair_flat
    delta_projected_norm = float(
        torch.linalg.norm(delta_projected_flat.to(dtype=torch.float64)).detach().cpu()
    )
    proj_diag["projection/repair_norm"] = repair_norm
    proj_diag["projection/delta_projected_norm"] = delta_projected_norm
    d_proj_norm = repair_norm
    proj_diag["projection/d_proj_norm"] = d_proj_norm
    proj_diag["projection/correction_norm_requested"] = d_proj_norm
    proj_diag["projection/cpo_trial/final_update_cap"] = 1.0
    proj_diag["projection/lambda_effective"] = float(lam)

    theta_projected = theta_adam
    proxy_ok = est.proxy_ok
    skip_projection = (
        not math.isfinite(C_old)
        or not proxy_ok
        or not math.isfinite(violation_for_lambda)
        or not torch.isfinite(b_flat).all()
        or not torch.isfinite(p_flat).all()
        or not torch.isfinite(delta_projected_flat).all()
    )
    if skip_projection:
        _log.warning(
            "[DGPO] projection skipped: non-finite C_raw, C_adam_proxy, grad C, or projected delta; "
            "keeping AdamW weights.",
        )
        assign_params_(model, theta_adam)
        proj_diag["projection/applied"] = 0.0
        proj_diag["projection/lambda"] = 0.0
        proj_diag["projection/reverted_nonfinite"] = 1.0
        proj_diag["projection/correction_norm"] = 0.0
        proj_diag["projection/final_update_norm"] = delta0_norm
        proj_diag["projection/final_update_norm_ratio"] = 1.0
        proj_diag["projection/final_update_scale"] = 1.0
    else:
        delta_final, cpo_diag = compute_cpo_adamw_final_update(
            model,
            delta0_flat,
            p_flat,
            lam,
            optimizer,
            proj_cfg,
        )
        proj_diag.update(cpo_diag)
        proj_diag["projection/trust_radius"] = float(
            cpo_diag.get("projection/cpo_trial/trust_radius_adamw", float("nan"))
        )
        proj_diag["projection/trust_scale"] = float(
            cpo_diag.get("projection/cpo_trial/final_update_scale", 1.0)
        )
        proj_diag["projection/trust_cap_active"] = float(
            cpo_diag.get("projection/cpo_trial/final_update_cap_active", 0.0)
        )
        corr_norm = assign_params_from_theta_old_delta_(model, theta_old, delta_final)
        final_update_norm = corr_norm
        proj_diag["projection/correction_norm"] = d_proj_norm
        proj_diag["projection/final_update_norm"] = final_update_norm
        eps_norm = 1e-12
        if delta0_norm > 0.0:
            proj_diag["projection/final_update_norm_ratio"] = final_update_norm / (
                delta0_norm + eps_norm
            )
            proj_diag["projection/correction_to_delta0_ratio"] = d_proj_norm / delta0_norm
        else:
            proj_diag["projection/final_update_norm_ratio"] = 0.0
            proj_diag["projection/correction_to_delta0_ratio"] = 0.0
        proj_diag["projection/final_update_scale"] = float(
            cpo_diag.get("projection/cpo_trial/final_update_scale", 1.0)
        )
        if not trainable_params_all_finite(model):
            _log.warning(
                "[DGPO] CPO projection produced non-finite weights (lambda=%.4g, "
                "final_update_norm_adamw=%.4g); reverting to AdamW step.",
                float(lam),
                float(cpo_diag.get("projection/cpo_trial/final_update_norm_adamw", float("nan"))),
            )
            assign_params_(model, theta_adam)
            proj_diag["projection/reverted_nonfinite"] = 1.0
            proj_diag["projection/applied"] = 0.0
        else:
            theta_projected = snapshot_params(model)
            proj_diag["projection/reverted_nonfinite"] = 0.0
            proj_diag["projection/applied"] = 1.0 if float(lam) > 0.0 else 0.0
            proj_diag["projection/lambda"] = float(lam)
    optimizer.zero_grad(set_to_none=True)

    delta_star_flat = flatten_param_delta(theta_projected, theta_old)
    d_star = delta_star_flat.to(dtype=torch.float64)
    b_dot_d_star = float(torch.dot(b_d, d_star).detach().cpu())
    v_after = c_margin + b_dot_d_star
    C_lin_projected = float(C_selected) + b_dot_d_star
    final_update_norm = float(torch.linalg.norm(delta_star_flat.to(dtype=torch.float64)).detach().cpu())
    if "projection/final_update_norm" not in proj_diag:
        proj_diag["projection/final_update_norm"] = final_update_norm
    if "projection/final_update_norm_ratio" not in proj_diag:
        eps_norm = 1e-12
        if delta0_norm > 0.0:
            proj_diag["projection/final_update_norm_ratio"] = final_update_norm / (
                delta0_norm + eps_norm
            )
        else:
            proj_diag["projection/final_update_norm_ratio"] = 0.0
    if "projection/final_update_scale" not in proj_diag:
        proj_diag["projection/final_update_scale"] = 1.0

    C_adam = float(C_adam_proxy) if math.isfinite(C_adam_proxy) else float("nan")
    off_diag_adam = off_diag_adam_pre
    proj_diag.update(
        {
            "projection/C_adam": float(C_adam),
            "projection/v_after": float(v_after),
            "projection/b_dot_delta_star": float(b_dot_d_star),
            "projection/actual_violation_adam": (
                max(0.0, float(C_adam) - eps_v) if math.isfinite(C_adam) else float("nan")
            ),
            "projection/C_linear_projected": float(C_lin_projected),
        }
    )
    for _off in (off_diag_old, off_diag_adam):
        if _off:
            proj_diag.update(_latent_projection_metrics(_off))

    for k, v in proj_diag.items():
        diag_last[k] = torch.tensor(float(v), device=device, dtype=torch.float64)
    if off_diag_old:
        diag_last.update(off_diag_old)


def train_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ema_rollout: Any | None,
    ema_save: Any | None,
    batch: dict[str, Any],
    optimizer: Any,
    sampler: DDIMSampler,
    reward_agg: RewardAggregator,
    *,
    beta: float,
    K: int,
    num_ddim_steps: int,
    rollout_parallel_chains: int = 1,
    global_step: int,
    epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    log_reward_dist: bool = False,
    log_diagnostic_dist: bool = False,
    diagnostic_plot_names: set[str] | None = None,
    diagnostic_plot_every: int = 1,
    num_train_timesteps: int = 1,
    adv_clip_max: float | None = None,
    grad_clip_norm: float = _GRAD_CLIP_NORM,
    policy_eval_t_min: float = 0.0,
    policy_eval_t_max: float = 1.0,
    constraint_state: ProjectionConstraintState | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    """Rollout once, accumulate the sub-step gradients, then one AdamW update + CPO repair.

    Frozen method: EMA-rollout candidate generation (when the rollout EMA exists),
    shared noise across the K candidates, z-score advantages, pure DGPO backward,
    post-AdamW latent-SWD CPO projection repair.
    """
    # Frozen method constant: all K candidates share the diffusion timestep t and
    # noise eps during policy evaluation (only the DDIM chain index varies).
    shared_noise = True
    model.train()
    ref_model.eval()
    freeze_reference_model(ref_model)
    proj_cfg = resolve_projection_constraint_config(global_config.dgpo)
    projection_active = bool(proj_cfg.active and constraint_state is not None)
    variance_reg_active = _variance_regularization_enabled(global_config.dgpo)
    variance_reg_weight = _variance_regularization_weight(global_config.dgpo)
    variance_reg_features = _variance_regularization_feature_names(global_config.dgpo)

    core = _unwrap_core_evenet(model)
    B = int(batch["x"].shape[0])

    # Candidate rollout uses the fast rollout-EMA shadow when available.
    buf: dict[str, Tensor] = {}
    if ema_rollout is not None:
        buf = _save_trainable_weights(model)
        ema_rollout.copy_to(core)
    try:
        with torch.no_grad():
            candidates_phys = generate_neutrino_candidates(
                core,
                batch,
                sampler,
                K=K,
                num_ddim_steps=num_ddim_steps,
                device=device,
                parallel_chains=rollout_parallel_chains,
            )
    finally:
        if buf:
            _restore_trainable_weights(model, buf)

    nonfinite_diag: dict[str, float] = {}
    cand_frac = _dgpo_nonfinite_fraction(candidates_phys)
    if cand_frac > 0.0:
        nonfinite_diag["train/candidate_nonfinite_fraction"] = cand_frac
        _log.warning(
            "[DGPO] non-finite generated candidates (frac=%.4g) at global_step=%s; zeroing.",
            cand_frac,
            global_step,
        )
        candidates_phys = _dgpo_zero_nonfinite(candidates_phys)

    rewards, reward_breakdown = reward_agg.compute(candidates_phys, batch)
    rewards, reward_breakdown, rew_nonfinite_diag = _dgpo_sanitize_rollout_rewards(
        rewards,
        reward_breakdown,
        global_step=int(global_step),
    )
    nonfinite_diag.update(rew_nonfinite_diag)
    valid_b = get_event_valid_mask(batch, B, device, dtype)
    advantages, _ = compute_per_event_advantage(rewards)
    advantages = _dgpo_zero_nonfinite(advantages)

    if adv_clip_max is not None:
        advantages = torch.clamp(advantages, -float(adv_clip_max), float(adv_clip_max))

    beta_dgpo = float(beta)
    if int(global_step) == 0:
        if projection_active:
            _log.info(
                "[DGPO] projection_constraint active (latent_swd, frozen encoder): backward "
                "uses pure DGPO main term only. Post-step AdamW-metric CPO on normalized "
                "C_norm = (SWD(z_pred,z_truth) - swd_tt) / (swd_tt + eps) with epsilon=%.4g. "
                "W&B: swd/* panel + projection/* CPO repair (x-axis global_step).",
                float(proj_cfg.epsilon),
            )
        else:
            _log.info(
                "[DGPO] pure DGPO mode: backward and optimizer step run without projection/CPO repair."
            )
        if variance_reg_active and variance_reg_weight > 0.0:
            _log.info(
                "[DGPO] anti-shrink variance regularization enabled: weight=%.4g, features=%s "
                "(defaults come from event_info.yaml unless overridden).",
                variance_reg_weight,
                ",".join(variance_reg_features) if variance_reg_features else "<none>",
            )

    candidate_weights_kb: Tensor | None = None
    if projection_active and proj_cfg.active_apply_to == "best_candidate":
        best_k = rewards.detach().argmax(dim=0)
        cols = torch.arange(B, device=device, dtype=torch.long)
        candidate_weights_kb = torch.zeros_like(rewards, dtype=torch.bool)
        candidate_weights_kb[best_k, cols] = True

    acc_steps = max(1, int(num_train_timesteps))
    optimizer_ran = False

    def _dgpo_substep() -> tuple[Tensor, dict[str, Tensor]]:
        (
            L_cur,
            L_ref,
            _,
            model_v,
            _ref_v,
            noise_mask_rep,
            x_t,
            target_v,
            t_rep,
            batch_rep,
            _eps_rep,
        ) = policy_evaluation_step(
            model,
            ref_model,
            batch,
            candidates_phys,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            t=None,
            t_min=policy_eval_t_min,
            t_max=policy_eval_t_max,
        )
        _dgpo_assert_train_step_invariants(L_ref, advantages, rewards)
        loss_vel, dlast = build_dgpo_loss(
            L_cur,
            L_ref,
            advantages,
            beta_dgpo,
            K,
        )
        beta_kl = max(0.0, float(_dgpo_cfg_get(global_config.dgpo, "beta_kl", 0.0)))
        kl_loss = per_row_velocity_mse(
            model_v,
            target_v,
            noise_mask_rep,
            invisible_padding=0,
        ).mean()

        variance_loss = x_t.new_zeros(())
        variance_diag: dict[str, Tensor] = {
            "train/regularization/variance/active": x_t.new_tensor(0.0, dtype=torch.float64),
            "train/regularization/variance/active_features": x_t.new_tensor(0.0, dtype=torch.float64),
            "train/regularization/variance/raw": x_t.new_tensor(0.0, dtype=torch.float64),
        }
        if variance_reg_active and variance_reg_weight > 0.0 and variance_reg_features:
            pred_x0_norm, _, _ = predict_x0_normalized_from_velocity_diffusion(
                x_t,
                model_v,
                t_rep,
            )
            pred_phys = core.invisible_normalizer.denormalize_grad(
                pred_x0_norm,
                mask=noise_mask_rep,
                remove_padding=True,
            )
            truth_phys = batch_rep["x_invisible"][..., :pred_phys.shape[-1]].to(
                device=pred_phys.device,
                dtype=pred_phys.dtype,
            )
            variance_loss, variance_diag = _variance_matching_penalty(
                pred_phys,
                truth_phys,
                noise_mask_rep,
                cartesian=_truth_generation_cartesian(),
                feature_names=_invisible_feature_names(),
                selected_features=variance_reg_features,
            )

        # Backward uses DGPO plus an optional supervised diffusion anchor and soft regularizers.
        # The latent-SWD constraint, when enabled, is still enforced post-AdamW by CPO repair.
        loss_backward = (
            loss_vel
            + beta_kl * kl_loss
            + float(variance_reg_weight) * variance_loss
        )
        dlast["loss_velocity_training"] = loss_vel.detach()
        dlast["train/loss/kl"] = (beta_kl * kl_loss.detach()).to(dtype=torch.float64)
        dlast["train/loss/variance_regularization"] = (
            float(variance_reg_weight) * variance_loss.detach()
        ).to(dtype=torch.float64)
        dlast["loss_total"] = loss_backward.detach()
        dlast["kl_weight_mean"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
        dlast["kl_weight_min"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
        dlast["kl_weight_max"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
        dlast["projection/active"] = torch.tensor(
            1.0 if projection_active else 0.0,
            device=device,
            dtype=torch.float64,
        )
        dlast["projection/pure_dgpo_backward"] = torch.tensor(
            1.0,
            device=device,
            dtype=torch.float64,
        )
        dlast.update(variance_diag)
        return loss_backward, dlast

    # Accumulate the sub-step gradients into ONE AdamW update per batch, then run
    # the post-AdamW CPO projection repair.
    grad_norm_pre_clip_max = 0.0
    grad_clip_active_any = False
    diag_last: dict[str, Tensor] = {}
    skipped_substeps = 0
    diags: list[dict[str, Tensor]] = []
    optimizer.zero_grad(set_to_none=True)
    theta_old_snap: dict[str, Tensor] | None = None
    for sub in range(1, acc_steps + 1):
        is_last = sub == acc_steps
        loss, dlast = _dgpo_substep()
        diags.append(dlast)
        if not torch.isfinite(loss):
            skipped_substeps += 1
            _log.warning(
                "[DGPO] non-finite substep loss (%s); skipping backward "
                "(step=%s sub=%s/%s).",
                float(loss.detach().float().cpu()),
                global_step, sub, acc_steps,
            )
            continue
        ctx = model.no_sync() if isinstance(model, DDP) and not is_last else nullcontext()
        with ctx:
            (loss / float(acc_steps)).backward()
    gn, clip_on = _grad_norm_pre_clip_and_clip_active(
        model, float(grad_clip_norm)
    )
    grad_norm_pre_clip_max = gn
    grad_clip_active_any = clip_on > 0.5
    if skipped_substeps == acc_steps or not math.isfinite(gn):
        _log.warning(
            "[DGPO] all substeps non-finite or grad-norm non-finite (%s); "
            "skipping optimizer.step at global_step=%s.",
            gn, global_step,
        )
        optimizer.zero_grad(set_to_none=True)
    else:
        theta_old_snap = snapshot_params(model)
        optimizer.step()
        optimizer_ran = True
    diag_last = _mean_diag_dict(diags)
    if optimizer_ran and theta_old_snap is not None and projection_active:
        _dgpo_projection_repair_after_adamw(
            model=model,
            ref_model=ref_model,
            batch=batch,
            candidates_phys=candidates_phys,
            theta_old=theta_old_snap,
            proj_cfg=proj_cfg,
            candidate_weights_kb=candidate_weights_kb,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            policy_eval_t_min=policy_eval_t_min,
            policy_eval_t_max=policy_eval_t_max,
            optimizer=optimizer,
            diag_last=diag_last,
            constraint_state=constraint_state,
            world_size=world_size,
            constraint_seed_base=int(global_step),
        )

    if optimizer_ran and ema_rollout is not None:
        ema_rollout.update(core, decay_=_dgpo_rollout_ema_decay(global_step))
    if ema_save is not None:
        ema_cfg = global_config.options.Training.get("EMA", None) or {}
        ema_every_n = max(1, int(ema_cfg.get("update_every_n_steps", 1)))
        if global_step % ema_every_n == 0:
            ema_save.update(core)

    out: dict[str, Any] = _build_train_metrics(
        diag_last,
        rewards,
        valid_b,
        advantages=advantages,
    )

    if log_reward_dist:
        out.update(build_reward_distribution_histograms(rewards, valid_b))
    plot_names = diagnostic_plot_names or set()
    plot_every = max(1, int(diagnostic_plot_every))
    log_diag_images = bool(log_diagnostic_dist) and (
        int(global_step) % plot_every == 0
    )
    collect_profile_accum = log_diag_images and ("pt_profile_accumulated" in plot_names)
    out.update(
        _build_reward_extra_metrics(
            rewards,
            valid_b,
            reward_agg,
            reward_breakdown,
            log_distribution=log_diag_images,
            collect_profile_accum=collect_profile_accum,
            diagnostic_plot_names=plot_names,
        )
    )
    out.update(
        _build_reference_bias_metrics(
            candidates_phys,
            batch,
            cartesian=_truth_generation_cartesian(),
            log_distribution=log_diag_images,
            diagnostic_plot_names=plot_names,
        )
    )
    out["train/grad/global_norm_pre_clip"] = float(grad_norm_pre_clip_max)
    out["train/grad/clip_active"] = 1.0 if grad_clip_active_any else 0.0
    out["projection/active"] = 1.0 if projection_active else 0.0
    out["projection/pure_dgpo_backward"] = 1.0
    _append_projection_summary_metrics(out)
    _append_projection_constraint_panel_metrics(out)
    _append_swd_panel_metrics(out)
    out.update(nonfinite_diag)

    cartesian = _truth_generation_cartesian()
    all_plot_feature_names = _generation_monitor_feature_names(cartesian=cartesian)
    if _supports_legacy_invisible_kinematics(
        cartesian=cartesian,
        feature_dim=int(batch["x_invisible"].shape[-1]),
    ):
        k_sel = _kin_hist_candidate_indices_per_event(
            rewards, candidates_phys, batch, cartesian=cartesian
        )
        ppt, peta, pphi, tpt, teta, tphi = _val_pred_truth_kin_flat(
            candidates_phys, batch, k_sel, cartesian=cartesian, device=device
        )
        k1_sel = torch.zeros(B, device=device, dtype=torch.long)
        k1_pt, k1_eta, k1_phi, k1_tpt, k1_teta, k1_tphi = _val_pred_truth_kin_flat(
            candidates_phys, batch, k1_sel, cartesian=cartesian, device=device
        )
        _td_pt_edges = _diagnostic_bin_edges("pt")
        _td_eta_edges = _diagnostic_bin_edges("eta")
        _td_phi_edges = _diagnostic_bin_edges("phi")
        out["_kin_h_pt_p"] = np.histogram(ppt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_pt_t"] = np.histogram(tpt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_e_p"] = np.histogram(peta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_e_t"] = np.histogram(teta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_p_p"] = np.histogram(pphi, bins=_td_phi_edges)[0].astype(np.float64)
        out["_kin_h_p_t"] = np.histogram(tphi, bins=_td_phi_edges)[0].astype(np.float64)
        out["_kin_h_pt_k1_p"] = np.histogram(k1_pt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_pt_k1_t"] = np.histogram(k1_tpt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_e_k1_p"] = np.histogram(k1_eta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_e_k1_t"] = np.histogram(k1_teta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_p_k1_p"] = np.histogram(k1_phi, bins=_td_phi_edges)[0].astype(np.float64)
        out["_kin_h_p_k1_t"] = np.histogram(k1_tphi, bins=_td_phi_edges)[0].astype(np.float64)
        if cartesian:
            all_px_p, all_py_p, all_pz_p, all_px_t, all_py_t, all_pz_t = (
                _val_pred_truth_cartesian_flat_all_candidates(
                    candidates_phys,
                    batch,
                    device=device,
                    dtype=dtype,
                )
            )
            out["_kin_all_px_p"] = all_px_p
            out["_kin_all_px_t"] = all_px_t
            out["_kin_all_py_p"] = all_py_p
            out["_kin_all_py_t"] = all_py_t
            out["_kin_all_pz_p"] = all_pz_p
            out["_kin_all_pz_t"] = all_pz_t
    if not cartesian:
        feature_arrays = _val_pred_truth_feature_flat_all_candidates(
            candidates_phys,
            batch,
            feature_names=all_plot_feature_names,
            device=device,
        )
        for key, values in feature_arrays.items():
            suffix = "p" if key.endswith("_pred") else "t"
            feature_name = key.rsplit("_", 1)[0]
            out[f"_kin_all_{feature_name}_{suffix}"] = values

    optimizer.scheduler_step()
    return out



def build_optimizer(
    model: torch.nn.Module,
    *,
    steps_per_epoch: int,
    warmup_steps: int,
    is_rank0: bool = True,
) -> _DgpoOptimizerWithSchedule:
    """AdamW with EveNet-style ``optimizer_group`` LR/WD (no world-size scaling) + linear warmup then constant LR.

    ``warmup_steps`` counts **batches** (one ``scheduler_step`` per call to :func:`train_step`;
    each ``train_step`` performs one accumulated ``optimizer.step()``).

    Parameters
    ----------
    model:
        ``EveNetModel`` or ``DDP(_DGPODDPForward(EveNetModel))``; parameters are taken from the
        unwrapped core for grouping (same tensor objects as ``model.parameters()``).
    steps_per_epoch:
        Logged on rank 0 for traceability (matches the worker's batch count per epoch).
    warmup_steps:
        Batches for linear LR ramp ``min(1, epoch / warmup_steps)`` on groups with ``warm_up: true``.
    is_rank0:
        When True, log one line per optimizer group on construction.
    """
    core = _unwrap_core_evenet(model)
    train_opt = global_config.options.Training
    components = train_opt.Components
    default_lr = float(train_opt.learning_rate)
    default_wd = float(train_opt.weight_decay)

    group_meta: dict[str, dict[str, Any]] = {}
    group_modules: dict[str, list[str]] = defaultdict(list)

    for comp_key, cfg in components.items():
        if cfg is None:
            continue
        group = cfg.get("optimizer_group", None)
        if not group:
            continue
        module_attr = getattr(core, comp_key, None)
        if module_attr is None:
            continue
        gname = str(group)
        group_modules[gname].append(str(comp_key))
        if gname not in group_meta:
            lr = float(cfg.get("learning_rate", default_lr))
            wd = float(cfg.get("weight_decay", default_wd))
            warm_up = bool(cfg.get("warm_up", True))
            opt_type = str(cfg.get("optimizer_type", "AdamW"))
            group_meta[gname] = {
                "lr": lr,
                "weight_decay": wd,
                "warm_up": warm_up,
                "optimizer_type": opt_type,
            }

    bad = [
        (g, m["optimizer_type"])
        for g, m in group_meta.items()
        if str(m["optimizer_type"]).lower() != "adamw"
    ]
    if bad:
        raise ValueError(
            "DGPO build_optimizer only supports AdamW parameter groups; got: "
            + ", ".join(f"{g}={t!r}" for g, t in bad)
        )

    ws = max(1, int(warmup_steps))
    param_groups: list[dict[str, Any]] = []
    lr_lambdas: list[Any] = []
    nonempty_group_order: list[str] = []

    for gname, meta in group_meta.items():
        seen_ids: set[int] = set()
        params: list[nn.Parameter] = []
        for comp_key in group_modules[gname]:
            mod = getattr(core, comp_key, None)
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    params.append(p)
        if not params:
            continue
        nonempty_group_order.append(gname)
        param_groups.append(
            {
                "params": params,
                "lr": float(meta["lr"]),
                "weight_decay": float(meta["weight_decay"]),
            }
        )
        if meta["warm_up"]:
            lr_lambdas.append(lambda epoch, _ws=ws: min(1.0, float(epoch) / float(_ws)))
        else:
            lr_lambdas.append(lambda _epoch: 1.0)

    if not param_groups:
        raise ValueError(
            "[DGPO] build_optimizer: no trainable parameters matched Components with "
            "optimizer_group (check include/freeze settings)."
        )

    # Fallback group for any trainable parameter not covered by an optimizer_group
    # (preserves the original permissive behavior of optimizing everything trainable).
    assigned = {id(p) for pg in param_groups for p in pg["params"]}
    leftover = [
        p for p in core.parameters() if p.requires_grad and id(p) not in assigned
    ]
    if leftover:
        if is_rank0:
            n_leftover = sum(p.numel() for p in leftover)
            _log.warning(
                "[DGPO] %s trainable parameters (%s elements) are not covered by any "
                "Components.<X>.optimizer_group; adding to a fallback group with "
                "lr=%s wd=%s warm_up=true.",
                len(leftover),
                n_leftover,
                default_lr,
                default_wd,
            )
        nonempty_group_order.append("__fallback__")
        param_groups.append(
            {
                "params": leftover,
                "lr": default_lr,
                "weight_decay": default_wd,
            }
        )
        lr_lambdas.append(lambda epoch, _ws=ws: min(1.0, float(epoch) / float(_ws)))

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    if is_rank0:
        _log.info(
            "[DGPO] Optimizer: AdamW groups=%s steps/epoch≈%s warmup_batches=%s (linear→constant).",
            len(param_groups),
            int(steps_per_epoch),
            ws,
        )
        for i, gname in enumerate(nonempty_group_order):
            pg = optimizer.param_groups[i]
            npar = sum(p.numel() for p in pg["params"])
            if gname in group_meta:
                mods = ", ".join(group_modules[gname])
                warm = group_meta[gname]["warm_up"]
            else:
                mods = "<fallback>"
                warm = True
            _log.info(
                "[DGPO]   group %r modules=[%s] params=%s lr=%s wd=%s warm_up=%s",
                gname,
                mods,
                npar,
                pg["lr"],
                pg["weight_decay"],
                warm,
            )

    return _DgpoOptimizerWithSchedule(optimizer, scheduler)


def _dgpo_wandb_metric_definition_map() -> dict[str, str]:
    """Explicit definitions for W&B Config → dgpo_metric_definitions (visible in the UI)."""
    return {
        "epoch": "Training epoch index (x-axis for most plots).",
        # --- reward/dist (overlaid figure, every log_reward_dist_every steps) ---
        "reward/dist/overlap": "Matplotlib figure: three overlapped 1D density histograms (best / worst / median per valid event). wandb.Image — use the media step slider to compare across training steps.",
        # --- reward/monitor (scalars, every step) ---
        "reward/monitor/best_of_k": "Mean reward of the argmax (best) candidate per valid event.",
        "reward/monitor/median": "Mean over events of the median reward along K.",
        "reward/monitor/mean_gap": "Mean over events: (mean reward strictly above per-event median) − (mean reward strictly below median).",
        "reward/monitor/last_place": "Mean reward of the worst (min) candidate per valid event.",
        "reward/monitor/p10": "Mean over events of the 10th percentile of rewards along K.",
        "reward/monitor/p30": "Mean over events of the 30th percentile of rewards along K.",
        "reward/monitor/p70": "Mean over events of the 70th percentile of rewards along K.",
        "reward/monitor/p90": "Mean over events of the 90th percentile of rewards along K.",
        "reward/monitor/advantage_pos_neg_gap": "Mean reward where advantage > 0 minus mean where advantage < 0 (valid slots).",
        "reward/sources/*/mean": "Raw mean reward for each additive reward source over all valid rollout candidates.",
        "reward/sources/*/weighted_mean": "Configured reward weight times reward/sources/*/mean; these add up to reward/raw/mean.",
        "reward/sources/*/selected_by_total_mean": "Per-source raw reward after selecting the candidate with highest combined total reward per valid event.",
        "reward/sources/*/selected_by_total_weighted_mean": "Configured reward weight times selected_by_total_mean.",
        "reward/sources/*/source_best_of_k": "For each source alone, mean over events of max_k source_reward[k,event].",
        "reward/sources/*/source_last_place": "For each source alone, mean over events of min_k source_reward[k,event].",
        "reward/sources/*/selection_gap": "selected_by_total_mean - mean. Large shifts show how combined reward selection biases that source.",
        # --- dgpo (train scalars) ---
        "projection/active": "1 when projection_constraint repair runs after AdamW on this step.",
        "projection/pure_dgpo_backward": "1 when the main policy objective stays DGPO-style in backward. Optional soft regularizers may be added, while any latent-SWD CPO repair still happens only post-AdamW.",
        # --- train/loss ---
        "train/loss/total": "Scalar passed to backward(): DGPO main term plus any enabled supervised diffusion anchor and soft regularizers. Post-step latent-SWD CPO repair is after AdamW only.",
        "train/loss/dgpo": "DGPO main term (detached gate × advantage × L_cur). Lower is better.",
        "train/loss/L_cur": "Mean velocity MSE for the trainable policy (DDIM target). Lower is better.",
        "train/loss/L_ref": "Mean velocity MSE for the frozen reference policy. Lower is better.",
        "train/loss/delta": "mean(|L_cur - L_ref|): average absolute gap between current and reference velocity MSE. Shows how far the policy has moved from frozen ref_model (not rollout EMA).",
        "train/loss/velocity": "Detached velocity objective slice: pure DGPO main term used for backward.",
        "train/loss/kl": "Weighted supervised diffusion anchor beta_kl * mean_row |v_pred - v_truth|^2 on the same noisy inputs. Keeps the original diffusion preference toward the denoising target while DGPO adds physics steering.",
        "train/loss/variance_regularization": "Weighted anti-shrink soft penalty added to backward: lambda_var * mean(relu((std_truth - std_pred) / std_truth)^2) over the selected event_info-driven angular features.",
        "train/regularization/variance/active": "1 when batch-level variance anti-shrink regularization found at least one selected feature in the current feature layout.",
        "train/regularization/variance/active_features": "How many selected features contributed to the anti-shrink regularizer on this batch.",
        "train/regularization/variance/raw": "Unweighted anti-shrink penalty before multiplying by dgpo.variance_regularization.weight.",
        "train/regularization/variance/*/std_truth": "Per-feature truth std over valid batch slots for the anti-shrink monitor.",
        "train/regularization/variance/*/std_pred": "Per-feature prediction std over valid batch slots for the anti-shrink monitor.",
        "train/regularization/variance/*/std_delta_ratio": "Signed relative std shift (std_pred - std_truth) / std_truth. Negative means the prediction is narrower than truth; positive means wider.",
        "train/regularization/variance/*/std_gap": "Per-feature positive relative shrinkage max((std_truth - std_pred) / std_truth, 0). Non-zero means the prediction is narrower than truth.",
        "train/regularization/variance/*/penalty": "Per-feature raw anti-shrink penalty relu((std_truth - std_pred) / std_truth)^2.",
        # --- projection (W&B panel: five CPO repair scalars only) ---
        "projection/v_linear": "Linear post-Adam violation estimate: C_adam_pred - epsilon. Drives lambda when positive.",
        "projection/C_adam_pred": "First-order Taylor prediction C_old + b^T delta0 at theta_adam (linear constraint after AdamW step).",
        "projection/lambda": "Closed-form CPO multiplier lambda_star = [v / (b^T p + damping)]_+ applied to the repair direction.",
        "projection/final_update_norm": "L2 norm ||theta_final - theta_old|| after projection (includes final-update cap when active).",
        "projection/summary/C_projected_minus_old": "C_projected - C_old on frozen (t, eps); negative means projection reduced the constraint vs pre-step weights.",
        "projection/multi_sample/C_mean": "Mean normalized constraint C_norm over multi-sample draws at theta_old; per-batch trace for sawtooth / oscillation diagnostics.",
        # --- swd (W&B panel: frozen latent-SWD constraint monitoring) ---
        "swd/active": "1 when latent-SWD constraint diagnostics were logged this step; 0 when the batch was skipped (too few valid rows).",
        "swd/pred_truth": "Sliced Wasserstein distance SWD(z_pred, z_truth) in the frozen encoder latent space.",
        "swd/truth_truth": "Null-floor SWD: random truth/truth split SWD_tt within the batch.",
        "swd/ratio": "swd_pred_truth / (swd_truth_truth + eps); raw ratio before null-excess normalization.",
        "swd/C_norm": "Ratio-normalized constraint (swd_pred_truth - swd_truth_truth) / (swd_truth_truth + eps); drives CPO when > margin.",
        "swd/mask_count": "Number of valid (event, candidate) rows encoded for SWD this step.",
        "swd/skipped_small_mask": "1 when mask_count < latent_swd.min_samples and the constraint was skipped.",
        "projection/reward_constraint_align": "cos(b, delta0) = b.delta0/(|b||delta0|) for the CPO repair. >0 => the reward step keeps pushing the constraint UP (genuine reward<->constraint tension); ~0 => orthogonal, so a reward stall is NOT caused by the constraint.",
        "projection/constraint/swd_pred_truth": "Alias of swd/pred_truth under projection/constraint/* (legacy projection panel).",
        "projection/constraint/swd_truth_truth": "Alias of swd/truth_truth under projection/constraint/*.",
        "projection/constraint/swd_ratio": "Alias of swd/ratio under projection/constraint/*.",
        "projection/constraint/C_norm": "Alias of swd/C_norm under projection/constraint/*.",
        "diagnostics/reward_hacking/all/px/reward_mean": "Mean px reward contribution (ν1+ν2, negative normalized squared error) over all valid rollout candidates.",
        "diagnostics/reward_hacking/all/py/reward_mean": "Mean py reward contribution over all valid rollout candidates.",
        "diagnostics/reward_hacking/all/pz/reward_mean": "Mean pz reward contribution over all valid rollout candidates.",
        "diagnostics/reward_hacking/best/px/reward_mean": "Mean px reward contribution after selecting the combined-reward argmax candidate per valid event.",
        "diagnostics/reward_hacking/best/py/reward_mean": "Mean py reward contribution on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/reward_mean": "Mean pz reward contribution on reward-best candidates.",
        "diagnostics/reward_hacking/all/px/delta_mean": "Signed mean px residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/py/delta_mean": "Signed mean py residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pz/delta_mean": "Signed mean pz residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/px/delta_abs_mean": "Mean absolute px residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/py/delta_abs_mean": "Mean absolute py residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pz/delta_abs_mean": "Mean absolute pz residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/px/delta_mean": "Signed mean px residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/py/delta_mean": "Signed mean py residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/delta_mean": "Signed mean pz residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/px/delta_abs_mean": "Mean absolute px residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/py/delta_abs_mean": "Mean absolute py residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/delta_abs_mean": "Mean absolute pz residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/all/pt/delta_mean": "Signed mean pT residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pt/delta_abs_mean": "Mean absolute pT residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/pt/delta_mean": "Signed mean pT residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pt/delta_abs_mean": "Mean absolute pT residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/all/eta/delta_mean": "Signed mean η residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/eta/delta_abs_mean": "Mean absolute η residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/eta/delta_mean": "Signed mean η residual on reward-best candidates.",
        "diagnostics/reward_hacking/best/eta/delta_abs_mean": "Mean absolute η residual on reward-best candidates.",
        "diagnostics/reward_hacking/all/phi/delta_mean": "Signed mean wrapped φ residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/phi/delta_abs_mean": "Mean absolute wrapped φ residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/phi/delta_mean": "Signed mean wrapped φ residual on reward-best candidates.",
        "diagnostics/reward_hacking/best/phi/delta_abs_mean": "Mean absolute wrapped φ residual on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/cos_opening": "Mean cos(opening angle) between the two reconstructed tau directions over all valid rollout candidates. Ideal back-to-back topology is near -1.",
        "diagnostics/ztautau_back_to_back/best/cos_opening": "Same cos(opening angle) metric after selecting the combined-reward argmax candidate per valid event.",
        "diagnostics/ztautau_back_to_back/all/delta_phi_to_pi": "Mean ||Delta phi| - pi| over all valid rollout candidates. Smaller is more back-to-back in azimuth.",
        "diagnostics/ztautau_back_to_back/best/delta_phi_to_pi": "Same azimuthal back-to-back metric on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/back_to_back_loss": "Mean (cos_opening + 1)^2 + (|Delta phi| - pi)^2 over all valid rollout candidates.",
        "diagnostics/ztautau_back_to_back/best/back_to_back_loss": "Same combined back-to-back loss on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/calibration_deltaR_a": "Mean post-calibration direction change DeltaR for tau-a over all valid rollout candidates. Smaller is more physics-consistent.",
        "diagnostics/ztautau_back_to_back/best/calibration_deltaR_a": "Same tau-a post-calibration DeltaR on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/calibration_deltaR_b": "Mean post-calibration direction change DeltaR for tau-b over all valid rollout candidates.",
        "diagnostics/ztautau_back_to_back/best/calibration_deltaR_b": "Same tau-b post-calibration DeltaR on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/calibration_deltaR_sum": "Mean calibration magnitude DeltaR_a + DeltaR_b over all valid rollout candidates. This is the physics-consistency reward when reward_config.type=calibration_magnitude.",
        "diagnostics/ztautau_back_to_back/best/calibration_deltaR_sum": "Same calibration magnitude on reward-best candidates.",
        "diagnostics/reward_hacking/all/rel_pt/mean": "Mean pT_pred / pT_truth - 1 over all valid rollout candidates and both ν slots. Negative values indicate pT shrink.",
        "diagnostics/reward_hacking/all/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/rel_pt/mean": "Mean pT_pred / pT_truth - 1 after selecting the combined-reward argmax candidate per valid event. Compare to all/rel_pt/mean to spot reward-driven pT shrink.",
        "diagnostics/reward_hacking/best/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) on reward-best candidates.",
        "diagnostics/reward_hacking/dist/rel_pt": "Matplotlib density overlay of pT_pred / pT_truth - 1 for all rollout candidates vs reward-best candidates (wandb.Image).",
        "diagnostics/reward_hacking/pt_oracle/pt/delta_mean": "Signed mean pT residual pred−truth in GeV after selecting, per event, the candidate with smallest |ΔpT_nu1| + |ΔpT_nu2|. This is a truth oracle for candidate-support diagnosis, not a deployable selector.",
        "diagnostics/reward_hacking/pt_oracle/pt/delta_abs_mean": "Mean absolute pT residual |pred−truth| in GeV for the pT-oracle-best candidate.",
        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt": "Profile plot by truth-pT bin: top panel compares mean delta pT = pT_pred - pT_truth for all rollout candidates, reward-best candidates, and pT-oracle-best candidates; bottom panel shows reward-best minus all and pT-oracle minus all. Gray bars show truth event-slot counts per bin, not K-times all-candidate counts. If pT-oracle fixes high-pT bins, support exists and ranking/reward is the bottleneck; if pT-oracle remains low, generator support is insufficient.",
        "diagnostics/reward_hacking/profile/eta_delta_vs_truth_eta": "Profile plot by truth-eta bin, with the same all/reward-best/eta-oracle comparison used by the pT residual profile.",
        "diagnostics/reward_hacking/profile/phi_delta_vs_truth_phi": "Profile plot by truth-phi bin using wrapped phi residuals, with the same all/reward-best/phi-oracle comparison used by the pT residual profile.",
        "diagnostics/reward_hacking/profile/px_delta_vs_truth_px": "Profile plot by truth-px bin, comparing mean px residual for all rollout, reward-best, and px-oracle candidates.",
        "diagnostics/reward_hacking/profile/py_delta_vs_truth_py": "Profile plot by truth-py bin, comparing mean py residual for all rollout, reward-best, and py-oracle candidates.",
        "diagnostics/reward_hacking/profile/pz_delta_vs_truth_pz": "Profile plot by truth-pz bin, comparing mean pz residual for all rollout, reward-best, and pz-oracle candidates.",
        "diagnostics/reward_hacking/profile/pt_delta_first10_vs_fullK": "Profile plot by truth-pT bin comparing first-10-candidate selection against full-K selection on the same rollout pool. Curves show reward-best first 10, reward-best full K, pT-oracle first 10, and pT-oracle full K; bottom panel shows full-K minus first-10 gains.",
        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt_accumulated": "Same as diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt, but concatenates raw diagnostic samples over dgpo.diagnostic_profile_accumulate_steps train batches before plotting. Use this for more stable high-pT tail statistics.",
        "diagnostics/reward_hacking/profile/eta_delta_vs_truth_eta_accumulated": "Accumulated truth-bin eta residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/phi_delta_vs_truth_phi_accumulated": "Accumulated truth-bin phi residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/px_delta_vs_truth_px_accumulated": "Accumulated truth-bin px residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/py_delta_vs_truth_py_accumulated": "Accumulated truth-bin py residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/pz_delta_vs_truth_pz_accumulated": "Accumulated truth-bin pz residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reference_bias/all/pt/delta_mean": "Signed mean pT residual pred−truth in GeV over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/pt/delta_abs_mean": "Mean absolute pT residual |pred−truth| in GeV over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/eta/delta_mean": "Signed mean η residual pred−truth over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/eta/delta_abs_mean": "Mean absolute η residual |pred−truth| over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/phi/delta_mean": "Signed mean wrapped φ residual pred−truth over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/phi/delta_abs_mean": "Mean absolute wrapped φ residual |pred−truth| over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/rel_pt/mean": "Mean pT_pred / pT_truth - 1 over all rollout candidates and the first two ν slots. Negative values indicate pT shrink. Freeze rollout updates to isolate initial/reference bias.",
        "diagnostics/reference_bias/all/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/dist/rel_pt": "Matplotlib density plot of pT_pred / pT_truth - 1 for rollout candidates (wandb.Image). With frozen rollout EMA, this is the initial/reference model's pT bias distribution.",
        "diagnostics/reference_bias/profile/pt_delta_vs_truth_pt": "Profile plot: x-axis truth pT bin [GeV], y-axis mean delta pT = pT_pred - pT_truth [GeV] over all rollout candidates and the first two ν slots. Negative high-pT bins indicate tail shrink / dynamic-range compression.",
        # --- train/grad (one accumulated optimizer step per batch) ---
        "train/grad/global_norm_pre_clip": "Total L2 norm of trainable gradients before clip_grad_norm_ (max over sub-steps in the batch). Compare to dgpo.grad_clip_norm in run config.",
        "train/grad/clip_active": "1.0 if any sub-step had pre-clip norm > dgpo.grad_clip_norm (clipping applied); else 0.0.",
        "train/candidate_nonfinite_fraction": "Fraction of generated candidate tensor elements that were non-finite before zeroing (DDIM / kinematics blow-up).",
        "train/reward_nonfinite_fraction": "Fraction of (K,B) total rewards that were non-finite before zeroing.",
        "train/reward_nonfinite_fraction/*": "Per reward-source non-finite fraction before zeroing.",
        # --- parameter (scalars, every step; extend with more keys later) ---
        "parameter/w_e/mean": "Mean per-event gate w_e = sigmoid(M_e) in [0,1].",
        "parameter/w_e/std": "Std of w_e across events in the batch (population std).",
        "parameter/w_e/min": "Min w_e in the batch.",
        "parameter/w_e/max": "Max w_e in the batch.",
        # --- val (epoch-end) ---
        "val/reward/mean": "Mean combined reward over all valid (candidate × event) pairs. Used for top-k checkpoint selection.",
        "val/reward/median": "Global median of per-event reward across valid events (with val_K=1: single prediction per event; with val_K>1: best-of-K per event). Epoch x-axis.",
        "val/reward/p10": "10th percentile of per-event reward (val_K=1: single pred; val_K>1: best-of-K per event).",
        "val/reward/p30": "30th percentile.",
        "val/reward/p70": "70th percentile.",
        "val/reward/p90": "90th percentile.",
        "val/winrate": "Fraction of valid events where the current policy's reward-best validation candidate beats the reference-policy sample on combined reward. NaN if validation_compute_winrate=false.",
        "val_diagnostics/profile/pt_delta_vs_truth_pt": "Validation profile plot by truth-pT bin: selected-candidate mean delta pT = pT_pred - pT_truth, with the initial pre-DGPO validation profile overlaid after the baseline pass.",
        "val_diagnostics/profile/eta_delta_vs_truth_eta": "Validation profile plot by truth-eta bin: selected-candidate mean eta residual, with the initial pre-DGPO validation profile overlaid after the baseline pass.",
        "val_diagnostics/profile/pt/delta_mean": "Global validation mean pT residual, pT_pred - pT_truth, over selected candidates and valid neutrino slots.",
        "val_diagnostics/profile/pt/slope": "Linear fit slope of the validation binned mean delta-pT profile versus truth pT.",
        "val_diagnostics/profile/pt/zero_delta_truth": "Truth pT value where the fitted validation mean delta-pT profile crosses zero.",
        "val_diagnostics/profile/eta/delta_mean": "Global validation mean eta residual over selected candidates and valid neutrino slots.",
        "val_diagnostics/profile/eta/slope": "Linear fit slope of the validation binned mean delta-eta profile versus truth eta.",
        "val_diagnostics/profile/eta/zero_delta_truth": "Truth eta value where the fitted validation mean delta-eta profile crosses zero.",
        "val_diagnostics/profile/pt_delta_mean_vs_epoch": "History plot with x-axis epoch and y-axis global validation mean pT residual. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/eta_delta_mean_vs_epoch": "History plot with x-axis epoch and y-axis global validation mean eta residual. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/pt_slope_vs_epoch": "History plot with x-axis epoch and y-axis fitted validation pT-profile slope. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/eta_slope_vs_epoch": "History plot with x-axis epoch and y-axis fitted validation eta-profile slope. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/pt_zero_delta_truth_vs_epoch": "History plot with x-axis epoch and y-axis fitted truth-pT zero-crossing where mean delta pT is zero.",
        "val_diagnostics/profile/eta_zero_delta_truth_vs_epoch": "History plot with x-axis epoch and y-axis fitted truth-eta zero-crossing where mean delta eta is zero.",
        "val_diagnostics/profile/pt_zero_delta_vs_slope": "History plot with x-axis fitted pT-profile slope and y-axis fitted truth-pT zero-crossing.",
        "val_diagnostics/profile/eta_zero_delta_vs_slope": "History plot with x-axis fitted eta-profile slope and y-axis fitted truth-eta zero-crossing.",
        "val/response/reward_initial_vs_current": "2D validation response matrix with x-axis initial pre-DGPO event reward and y-axis current event reward, logged under one W&B image key each epoch so the Images panel has an epoch slider.",
        "val/response/pt_delta_mean_initial_vs_current": "2D validation response matrix with x-axis initial pre-DGPO event mean delta pT and y-axis current event mean delta pT, logged under one W&B image key each epoch so the Images panel has an epoch slider.",
        "val_neutrino/pt": "1D density overlay: truth vs current-policy vs frozen-reference prediction for pT [GeV] (original scale, expm1 of log1p(pT)) (wandb.Image); x-axis **epoch**. Current-policy histogram uses the same per-event candidate index rule as train_dist/* (combined-reward argmax).",
        "val_neutrino/eta": "Same three-way overlay for η; same candidate selection as val_neutrino/pt.",
        "val_neutrino/phi": "Same three-way overlay for φ [rad]; same candidate selection as val_neutrino/pt.",
        "val_neutrino/all/pt_truth_vs_pred": "Example 2D truth-vs-pred key. Actual validation 2D keys follow event_info.invisible_feature_names as val_neutrino/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "val_neutrino/all/eta_truth_vs_pred": "Example 2D truth-vs-pred key. Actual validation 2D keys follow event_info.invisible_feature_names as val_neutrino/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "val_neutrino/all/phi_truth_vs_pred": "Example 2D truth-vs-pred key. Actual validation 2D keys follow event_info.invisible_feature_names as val_neutrino/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "val_neutrino/all_metrics/*/*": "Scalar summaries for validation truth-vs-pred 2D monitors over all candidates: count, mae, rmse, bias=mean(pred-truth), pearson_r, slope, intercept.",
        "val_neutrino/px": "Same three-way overlay for neutrino p_x [GeV]; truth is denormalized invisible target, pred/ref from DDIM output.",
        "val_neutrino/py": "Same three-way overlay for neutrino p_y [GeV].",
        "val_neutrino/pz": "Same three-way overlay for neutrino p_z [GeV].",
        "val_neutrino/jsd/current/*": "Histogram Jensen-Shannon distance between truth and current-policy validation distributions for the named kinematic. Feature names follow event_info.yaml invisible_feature_names (or px/py/pz in cartesian mode). Lower is better.",
        "val_neutrino/jsd/ref/*": "Histogram Jensen-Shannon distance between truth and frozen-reference validation distributions for the named kinematic. Feature names follow event_info.yaml invisible_feature_names (or px/py/pz in cartesian mode). Lower is better.",
        "val_mass/w_mass": "W-boson mass reconstruction (assigned lepton + neutrino) vs truth-neutrino resonance mass, truth vs current policy vs frozen reference (wandb.Image); x-axis **epoch**.",
        "val_mass/top_mass": "Top mass reconstruction (assigned b + W) vs truth-neutrino resonance mass; same three-way overlay as val_mass/w_mass.",
        "val_mass/jsd/current/*": "Histogram Jensen-Shannon distance between truth and current-policy validation mass distributions. Lower is better.",
        "val_mass/jsd/ref/*": "Histogram Jensen-Shannon distance between truth and frozen-reference validation mass distributions. Lower is better.",
        # --- train_dist (epoch end, accumulated over all training batches; own wandb panel) ---
        "train_dist/pt": "1D density overlay: truth vs best-of-K training prediction for pT [GeV] (original scale), accumulated over all training batches in the epoch (wandb.Image). x-axis **epoch**. \"Best\" = combined-reward argmax among K candidates.",
        "train_dist/eta": "Same overlay for η (training); same candidate selection as train_dist/pt.",
        "train_dist/phi": "Same overlay for φ [rad] (training); same candidate selection as train_dist/pt.",
        "train_dist/jsd/current/*": "Histogram Jensen-Shannon distance between truth and reward-best train rollout distributions, accumulated over the epoch. Feature names follow event_info.yaml invisible_feature_names (or px/py/pz in cartesian mode). Lower is better.",
        "train_dist/all/pt_truth_vs_pred": "Example 2D truth-vs-pred key. Actual train epoch-end 2D keys follow event_info.invisible_feature_names as train_dist/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "train_dist/all/eta_truth_vs_pred": "Example 2D truth-vs-pred key. Actual train epoch-end 2D keys follow event_info.invisible_feature_names as train_dist/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "train_dist/all/phi_truth_vs_pred": "Example 2D truth-vs-pred key. Actual train epoch-end 2D keys follow event_info.invisible_feature_names as train_dist/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "train_dist/all_metrics/*/*": "Scalar summaries for train truth-vs-pred 2D monitors over all candidates: count, mae, rmse, bias=mean(pred-truth), pearson_r, slope, intercept.",
        "train_dist_k1/pt": "1D density overlay: truth vs candidate-0 training rollout prediction for pT [GeV], accumulated over all training batches in the epoch. This is a K=1 / single-sample proxy on the train rollout pool, separate from reward-best train_dist/*.",
        "train_dist_k1/eta": "Same overlay for η using candidate 0 as the train K=1 proxy.",
        "train_dist_k1/phi": "Same overlay for φ using candidate 0 as the train K=1 proxy.",
    }


def _dgpo_wandb_hyperparameter_definitions() -> dict[str, str]:
    """Explicit definitions for DGPO hyperparameters (visible in W&B Config).

    Explains what each parameter in the ``dgpo:`` and ``reward_config:`` sections does.
    """
    return {
        # --- dgpo: core RL hyperparameters ---
        "dgpo.beta": "beta_dgpo: Temperature parameter that scales the event-level gate logit M_e. Higher beta makes the gate more sensitive to the advantage-weighted velocity gap. M_e = (beta / K) * sum_over_candidates(advantage * Delta). Typical range: 0.1 to 1.0. Current value controls how aggressively events are up/down-weighted in the loss.",
        "dgpo.grad_clip_norm": "Global L2 gradient clip for AdamW (torch.nn.utils.clip_grad_norm_). Compare train/grad/global_norm_pre_clip to this value.",
        "dgpo.K": "Number of DDIM candidate samples generated per event **during training** (rollout + DGPO). Each event gets K neutrino reconstructions, and the reward function ranks them. Larger K = more candidates to choose from (better oracle performance) but slower generation. Typical values: 4-16.",
        "dgpo.validation_K": "Candidates per event during **validation** only (independent of training K). Default **1**: one current-policy DDIM sample per event; with validation_compute_winrate, one additional ref-policy DDIM per event for reward-based winrate. Does not advance the training global_step or train-panel x-axis.",
        "dgpo.rollout_parallel_chains": "How many DDIM chains to batch together per model call during training rollout. Keeps total K the same, but runs up to this many chains in one larger forward pass. Higher can be faster if GPU headroom exists; too high can OOM.",
        "dgpo.validation_rollout_parallel_chains": "Validation-only version of rollout_parallel_chains. If unset, falls back to the training value.",
        "dgpo.num_ddim_steps": "Number of DDIM denoising steps (T_sample) used for online candidate generation during **training** only. More steps = higher-quality samples but slower. Typical values: 20-100.",
        "dgpo.validation_num_ddim_steps": "Number of DDIM denoising steps used for candidate generation during **validation** only (independent of training num_ddim_steps). If null, falls back to num_ddim_steps. Lets you validate at higher fidelity than the training rollout without slowing training.",
        "dgpo.diagnostic_profile_accumulate_steps": "Number of train batches to concatenate before logging accumulated diagnostics/reward_hacking/profile/*_delta_vs_truth_* images. Larger values stabilize sparse bins but update the W&B images less often.",
        "dgpo.log_every": "Log Python INFO messages (loss, reward, etc.) to the console every N optimizer steps. Does not affect wandb logging frequency (wandb logs every step when enabled). Typical: 1-10.",
        "dgpo.log_reward_dist_every": "Log reward/dist/overlap every N optimizer steps when wandb is enabled. Defaults to dgpo.diagnostic_plots.plot_every when omitted.",
        "dgpo.validation_every_n_epochs": "Run end-of-epoch DDIM validation when (epoch+1) is divisible by this value (default 1 = every epoch). Epoch -1 baseline at train start is unaffected. Top-K checkpointing runs only on validation epochs.",
        "dgpo.validation_max_batches": "If set (e.g. 20): stop validation after this many batches per epoch (faster validation for debugging). If null: run full validation set. Typical: null for real training, 5-20 for smoke tests.",
        "dgpo.validation_compute_winrate": "If true: each validation batch generates one extra reference-policy DDIM sample to compute reward-based val/winrate. Adds ~50% validation time. If false: skip winrate (logged as NaN). Typical: false (cheaper validation).",
        "dgpo.validation_log_batches": "If true: log INFO messages for each validation batch (start time, DDIM wall time). Useful for monitoring long validation runs. Typical: true.",
        "dgpo.validation_tqdm_k_chains": "If true: show a tqdm progress bar over the K DDIM chains per validation batch. Typical: true (helps see validation progress).",
        "dgpo.validation_tqdm_ddim": "If true: show a tqdm progress bar for every DDIM step within each chain (very verbose). Typical: false (too much output).",
        # --- reward_config ---
        "reward_config.type": "Primary reward backend. 'component_normalized_truth_distance' uses truth-matching squared error; 'calibration_magnitude' uses post-calibration tau direction-change magnitude (smaller is better) as the primary physics-consistency reward.",
        "reward_config.weight": "Weight of the primary reward selected by reward_config.type before summing into the combined DGPO reward. Typical: 1.0.",
        "reward_config.component_normalized.weight": "Optional additive weight for the original truth-matching reward when the primary reward is calibration_magnitude. Set >0 to add the MSE-style term back into the combined reward; ignored when reward_config.type=component_normalized_truth_distance because that mode already uses reward_config.weight.",
        "reward_config.component_normalized.eps": "Numerical stability added to per-component scale denominators in the truth-distance reward.",
    }


def _dgpo_wandb_publish_metric_docs() -> None:
    """Expose metric definitions in the W&B UI (Config, Summary, Notes, Artifact).

    W&B does not show definitions next to each chart; Config + Artifact are the supported surfaces.
    """
    import wandb

    run = wandb.run
    if run is None:
        return
    defs = _dgpo_wandb_metric_definition_map()
    param_defs = _dgpo_wandb_hyperparameter_definitions()
    try:
        wandb.config.update(
            {
                "dgpo_metric_definitions": defs,
                "dgpo_hyperparameter_definitions": param_defs,
                "dgpo_metrics_full_doc_repo_path": (
                    "RL/DGPO_neutrino/diagnostics/metrics_reference.md"
                ),
                "dgpo_dynamic_reward_keys_note": (
                    "Training metrics use W&B step=global_step; val/* and train_dist/* use epoch as x-axis. "
                    "Groups: reward/dist, reward/monitor, train/loss, train/grad, parameter/*, "
                    "components/*, diagnostics/reward_hacking/* (including all/ vs best/), "
                    "diagnostics/reference_bias/*; projection/* + swd/* (latent-SWD CPO repair, x-axis global_step); "
                    "train_dist/*, train_dist_k1/*; val/reward/*, val/winrate, "
                    "val_diagnostics/*, val_neutrino/*."
                ),
            },
            allow_val_change=True,
        )
    except Exception as e:
        _log.warning("[DGPO] wandb.config metric definitions failed: %s", e)
    try:
        run.summary["dgpo_how_to_read_metrics"] = (
            "Config → dgpo_metric_definitions (metrics) + dgpo_hyperparameter_definitions (params). "
            "Artifacts → dgpo-metrics-reference → metrics_reference.md (full doc)."
        )
    except Exception as e:
        _log.warning("[DGPO] wandb.summary metric pointer failed: %s", e)
    try:
        run.notes = (
            "DGPO metric + hyperparameter definitions: open this run's **Config** "
            "(dgpo_metric_definitions, dgpo_hyperparameter_definitions) "
            "or **Artifacts** (dgpo-metrics-reference). "
            "Repo copy: RL/DGPO_neutrino/diagnostics/metrics_reference.md"
        )
    except Exception as e:
        _log.warning("[DGPO] wandb run notes failed: %s", e)
    md_path = Path(__file__).resolve().parent / "diagnostics" / "metrics_reference.md"
    if md_path.is_file():
        try:
            art = wandb.Artifact("dgpo-metrics-reference", type="documentation")
            art.add_file(str(md_path), name="metrics_reference.md")
            run.log_artifact(art)
        except Exception as e:
            _log.warning("[DGPO] wandb artifact for metrics_reference.md failed: %s", e)


def _wandb_is_media_value(v: Any) -> bool:
    """True for ``wandb`` loggable media types (histograms, images, etc.)."""
    mod = getattr(type(v), "__module__", "") or ""
    name = getattr(type(v), "__name__", "")
    return mod.startswith("wandb") and name in (
        "Histogram",
        "Image",
        "Plotly",
        "Video",
        "Html",
    )


_PROJECTION_WANDB_SCALAR_KEYS = frozenset({
    "projection/v_linear",
    "projection/C_adam_pred",
    "projection/lambda",
    "projection/final_update_norm",
    "projection/summary/C_projected_minus_old",
    "projection/multi_sample/C_mean",
})

_SWD_WANDB_SCALAR_KEYS = frozenset({
    "swd/active",
    "swd/pred_truth",
    "swd/truth_truth",
    "swd/ratio",
    "swd/C_norm",
    "swd/mask_count",
    "swd/skipped_small_mask",
})

_PROJECTION_CONSTRAINT_WANDB_SCALAR_PREFIX = "projection/constraint/"


def _wandb_train_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize logged keys: known prefixes pass through; bare keys get ``train/``.

    Passes through ``wandb.Histogram`` / ``wandb.Image`` values under ``reward/`` or ``val/``.
    Known prefixes ``train/``, ``val/``, ``reward/``, ``parameter/``, ``components/``,
    and ``diagnostics/`` are logged as-is. ``components/`` is the per-component reward
    breakdown panel; ``diagnostics/`` is for monitoring-only reward-hacking checks.
    Keys starting with ``_`` are internal (e.g. ``_kin_h_*`` histogram arrays) and are skipped.
    """
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        if k.endswith("_hist") and isinstance(v, np.ndarray):
            try:
                import wandb

                out[k] = wandb.Histogram(v)
            except Exception:
                continue
        elif k.startswith("projection/"):
            if k in _PROJECTION_WANDB_SCALAR_KEYS or k.startswith(_PROJECTION_CONSTRAINT_WANDB_SCALAR_PREFIX):
                out[k] = v
        elif k.startswith("swd/"):
            if k in _SWD_WANDB_SCALAR_KEYS:
                out[k] = v
        elif k.startswith((
            "train/",
            "val/",
            "reward/",
            "parameter/",
            "components/",
            "diagnostics/",
            "dgpo/",
        )):
            out[k] = v
        elif _wandb_is_media_value(v):
            out[k] = v
        else:
            out[f"train/{k}"] = v
    return out


def _dgpo_wandb_yaml_section() -> tuple[Any | None, str]:
    """Resolve W&B settings the same way as ``evenet/train.py`` + ``WandbLogger``.

    Prefer ``logger.wandb`` when it defines a project (standard EveNet YAML). Otherwise
    use the top-level ``wandb:`` block (DGPO configs often keep it there alongside
    ``logger:`` for tensorboard-only fields).

    Returns:
        ``(section_dict, source_label)`` where ``source_label`` is ``\"logger.wandb\"``
        or ``\"wandb\"`` for logging.
    """
    gc = global_config._global_config
    logger = gc.get("logger")
    if isinstance(logger, dict):
        nested = logger.get("wandb")
        if isinstance(nested, dict) and nested.get("project") is not None:
            return nested, "logger.wandb"
    top = gc.get("wandb")
    if isinstance(top, dict) and len(top) > 0:
        return top, "wandb"
    return None, ""


_DEFAULT_DIAGNOSTIC_PLOTS = {
    "pt_profile_accumulated",
    "projection_violation_compare",
}
_DEFAULT_DIAGNOSTIC_BIN_RANGES: dict[str, tuple[float, float]] = {
    "pt": (0.0, 300.0),
    "eta": (-4.0, 4.0),
    "phi": (-3.2, 3.2),
    "px": (-300.0, 300.0),
    "py": (-300.0, 300.0),
    "pz": (-800.0, 800.0),
    "wmass": (50.0, 120.0),
    "topmass": (100.0, 250.0),
}


def _diagnostic_plot_block(dg_cfg: Any) -> Any | None:
    """Return ``dgpo.diagnostic_plots`` when present, else ``None``."""
    return _dgpo_cfg_get(dg_cfg, "diagnostic_plots", None)


def _resolve_diagnostic_num_bins(dg_cfg: Any | None = None) -> int:
    """Resolve W&B diagnostic histogram bin count with legacy-key fallback."""
    cfg = dg_cfg if dg_cfg is not None else getattr(global_config, "dgpo", None)
    block = _diagnostic_plot_block(cfg)
    raw = _dgpo_cfg_get(block, "num_bins", None)
    if raw is None:
        raw = _dgpo_cfg_get(cfg, "diagnostic_num_bins", _VAL_KIN_NUM_BINS)
    return max(2, int(raw))


def _resolve_diagnostic_bin_range(name: str, dg_cfg: Any | None = None) -> tuple[float, float] | None:
    """Resolve per-variable diagnostic plot ranges from config or defaults."""
    cfg = dg_cfg if dg_cfg is not None else getattr(global_config, "dgpo", None)
    block = _diagnostic_plot_block(cfg)
    raw_ranges = _dgpo_cfg_get(block, "bin_ranges", None)
    if raw_ranges is None:
        raw_ranges = _dgpo_cfg_get(cfg, "diagnostic_bin_ranges", None)
    raw = None
    if isinstance(raw_ranges, Mapping):
        raw = raw_ranges.get(name, None)
    if raw is not None:
        try:
            lo, hi = [float(x) for x in raw]
        except (TypeError, ValueError):
            lo = hi = float("nan")
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi
    return _DEFAULT_DIAGNOSTIC_BIN_RANGES.get(name)


def _diagnostic_bin_edges(name: str, dg_cfg: Any | None = None) -> np.ndarray | None:
    """Fixed diagnostic histogram edges for known/configured variables."""
    bounds = _resolve_diagnostic_bin_range(name, dg_cfg=dg_cfg)
    if bounds is None:
        return None
    lo, hi = bounds
    return np.linspace(lo, hi, _resolve_diagnostic_num_bins(dg_cfg=dg_cfg) + 1)


def _resolve_diagnostic_plot_settings(dg_cfg: Any) -> tuple[set[str], int]:
    """Resolve W&B media diagnostics from ``dgpo.diagnostic_plots`` (names, plot_every)."""
    block = _diagnostic_plot_block(dg_cfg)
    if block is None:
        raw_names = _dgpo_cfg_get(dg_cfg, "diagnostic_plot_names", None)
        plot_every = max(1, int(_dgpo_cfg_get(dg_cfg, "diagnostic_plot_every", 1)))
        if raw_names is None:
            return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every
        if isinstance(raw_names, str):
            return {raw_names}, plot_every
        try:
            return {str(name) for name in raw_names}, plot_every
        except TypeError:
            return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every
    enabled = _dgpo_cfg_get(block, "enabled", None)
    if enabled is not None and not bool(enabled):
        return set(), 1
    plot_every = max(1, int(_dgpo_cfg_get(block, "plot_every", 1)))
    raw_names = _dgpo_cfg_get(block, "include", None)
    if raw_names is None:
        return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every
    if isinstance(raw_names, str):
        return {raw_names}, plot_every
    try:
        return {str(name) for name in raw_names}, plot_every
    except TypeError:
        return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every


def _resolve_diagnostic_plot_names(dg_cfg: Any) -> set[str]:
    """Resolve opt-in W&B media plot names from ``dgpo.diagnostic_plots``."""
    names, _ = _resolve_diagnostic_plot_settings(dg_cfg)
    return names


def _wandb_sanitize_log_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop non-finite floats so one bad scalar does not invalidate the whole W&B step.

    Preserves ``wandb`` media objects (histograms, images) unchanged.
    """
    out: dict[str, Any] = {}
    for k, v in data.items():
        if _wandb_is_media_value(v):
            out[k] = v
            continue
        if isinstance(v, bool):
            out[k] = float(v)
            continue
        if isinstance(v, int) and not isinstance(v, bool):
            out[k] = v
            continue
        if isinstance(v, float):
            if math.isfinite(v):
                out[k] = v
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
        except (TypeError, ValueError):
            pass
    return out


def _wandb_reset_step_tracker() -> None:
    """Reset monotonic W&B step tracking (call once after ``wandb.init``)."""
    global _wandb_committed_step
    _wandb_committed_step = -1


def _wandb_train_step(global_step: int) -> int:
    """W&B ``step`` for DGPO training metrics — identical to ``global_step``."""
    return int(global_step)


def _wandb_epoch_end_step(global_step: int) -> int:
    """W&B ``step`` for epoch-end panels tied to the last training step of the epoch."""
    return _wandb_train_step(max(int(global_step) - 1, 0))


def _wandb_log_with_step(wandb_mod: Any, payload: dict[str, Any], *, step: int) -> None:
    """``wandb.log`` with an explicit monotonic ``step=`` (required for all DGPO logging)."""
    global _wandb_committed_step
    clean = _wandb_sanitize_log_dict(payload)
    if not clean:
        return
    s = int(step)
    if _wandb_committed_step >= 0 and s < _wandb_committed_step:
        _log.warning(
            "[DGPO] wandb step=%s < committed=%s; skipping to avoid step regression.",
            s,
            _wandb_committed_step,
        )
        return
    try:
        wandb_mod.log(clean, step=s)
        _wandb_committed_step = max(_wandb_committed_step, s)
    except Exception as e:
        _log.warning("[DGPO] wandb.log failed at step=%s: %s", s, e)


def _wandb_log_step(wandb_mod: Any, payload: dict[str, Any], *, step: int) -> None:
    """Training / projection metrics: W&B Step axis tracks ``global_step`` (LinearPostAdam style)."""
    _wandb_log_with_step(wandb_mod, payload, step=_wandb_train_step(step))


def _wandb_log_validation(
    wandb_mod: Any,
    val_metrics: dict[str, Any],
    *,
    epoch: int,
    wandb_step: int,
) -> None:
    """Log ``val/*`` with ``epoch`` as the chart x-axis (``define_metric``) and explicit ``step=``.

    ``wandb_step`` is the last training step of the epoch (or 0 for the epoch=-1 baseline).
    Charts still use **epoch** as x-axis; ``step=`` only keeps W&B's internal counter aligned
    with training ``global_step``.
    """
    clean = _wandb_sanitize_log_dict(dict(val_metrics))
    if not clean:
        return
    clean["epoch"] = float(epoch)
    _wandb_log_with_step(wandb_mod, clean, step=wandb_step)


def _start_wandb_run(*, disable: bool = False) -> bool:
    """Initialize wandb like Lightning's ``WandbLogger`` in ``evenet/train.py``.

    Uses ``logger.wandb`` when present, else top-level ``wandb``. Applies
    ``Settings(start_method=\"thread\")`` for compatibility with Ray Train workers
    (forked subprocesses + threads do not mix well with the default ``fork`` method).
    """
    if disable:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        _log.info("[DGPO] WANDB_DISABLED set; skipping wandb.")
        return False
    try:
        import wandb
    except ImportError:
        _log.warning("[DGPO] wandb not installed; skipping experiment logging.")
        return False
    wb, wb_source = _dgpo_wandb_yaml_section()
    if wb is None:
        _log.info(
            "[DGPO] No wandb settings (add top-level ``wandb:`` or ``logger.wandb:``); "
            "skipping wandb.init."
        )
        return False
    project = wb.get("project")
    if project is None:
        _log.warning("[DGPO] wandb section missing ``project``; skipping wandb.init.")
        return False
    run_name = wb.get("run_name") or wb.get("name")
    tags = wb.get("tags") or []
    if not isinstance(tags, list):
        tags = list(tags)
    run_id = wb.get("id")
    resume = wb.get("resume")
    init_kw: dict[str, Any] = {
        "project": str(project),
        "entity": wb.get("entity"),
        "name": run_name,
        "tags": tags,
        "id": run_id,
        "config": global_config.to_logger(),
    }
    if resume is not None:
        init_kw["resume"] = resume
    try:
        init_kw["settings"] = wandb.Settings(start_method="thread")
    except Exception:
        pass
    wandb.init(**init_kw)
    _log.info(
        "[DGPO] wandb.init project=%s name=%s (config_key=%s)",
        project,
        run_name,
        wb_source or "unknown",
    )
    try:
        # Train / projection: W&B Step = ``global_step``.
        # Val / train_dist: chart x-axis is ``epoch`` (not global_step).
        wandb.define_metric("epoch")
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("val_diagnostics/*", step_metric="epoch")
        wandb.define_metric("val_neutrino/*", step_metric="epoch")
        wandb.define_metric("val_mass/*", step_metric="epoch")
        wandb.define_metric("train_dist/*", step_metric="epoch")
        wandb.define_metric("train_dist_k1/*", step_metric="epoch")
        wandb.define_metric("projection/*", step_metric="global_step")
        wandb.define_metric("swd/*", step_metric="global_step")
    except Exception as e:
        _log.warning("[DGPO] wandb.define_metric(val/*) failed (val may share step with train): %s", e)
    _wandb_reset_step_tracker()
    _dgpo_wandb_publish_metric_docs()
    return True


def _finish_wandb_run(active: bool) -> None:
    if not active:
        return
    try:
        import wandb

        wandb.finish()
    except Exception as e:
        _log.warning("[DGPO] wandb.finish() failed: %s", e)


def _dgpo_constraint_checkpoint_payload(
    constraint_state: ProjectionConstraintState,
) -> dict[str, Any] | None:
    """Serializable projection-constraint state for Lightning-compatible DGPO checkpoints.

    The frozen latent-SWD state serializes only its checkpoint/normalization provenance.
    """
    if constraint_state is None:
        return None
    return constraint_state.checkpoint_payload()


def _dgpo_save_last_ckpt(
    model: torch.nn.Module,
    ema_save: Any | None,
    optimizer: torch.optim.Optimizer,
    ref_model: torch.nn.Module,
    *,
    last_completed_epoch: int,
    dgpo_next_epoch: int,
    global_step: int,
    ema_rollout: Any | None = None,
    dgpo_projection_constraint_state: dict[str, Any] | None = None,
) -> None:
    """Write ``last.ckpt`` with live trainable weights in ``state_dict`` plus optional ``ema_state_dict``.

    Refuses to overwrite ``last.ckpt`` when any trainable parameter is non-finite, so a single
    bad batch cannot poison the resume state of a long-running DGPO job.
    """
    save_dir = global_config.options.Training.get("model_checkpoint_save_path", None)
    if not save_dir:
        _log.debug("[DGPO] model_checkpoint_save_path unset; skipping checkpoint save.")
        return
    core_for_check = _unwrap_core_evenet(model)
    bad = [n for n, p in core_for_check.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        _log.warning(
            "[DGPO] last.ckpt save SKIPPED at epoch=%s step=%s: %s non-finite trainable params "
            "(first: %s). Existing last.ckpt is preserved.",
            last_completed_epoch, global_step, len(bad), bad[:3],
        )
        return
    path = Path(str(save_dir)).expanduser().resolve() / "last.ckpt"
    save_lightning_compatible_checkpoint(
        path,
        model,
        ema_save,
        global_config,
        last_completed_epoch=last_completed_epoch,
        dgpo_next_epoch=dgpo_next_epoch,
        global_step=global_step,
        optimizer=optimizer,
        ref_model=ref_model,
        ema_rollout=ema_rollout,
        dgpo_projection_constraint_state=dgpo_projection_constraint_state,
    )


class _DgpoCheckpointTopK:
    """Keep the best ``val/reward/mean`` checkpoints (higher is better) under ``model_checkpoint_save_path``."""

    def __init__(self, save_dir: Path, top_k: int) -> None:
        self._save_dir = save_dir
        self._top_k = max(0, int(top_k))
        # (val_reward_mean, path_str) smallest reward first for easy pop
        self._worst_heap: list[tuple[float, str]] = []

    def maybe_save(
        self,
        *,
        val_reward_mean: float,
        last_completed_epoch: int,
        dgpo_next_epoch: int,
        global_step: int,
        model: torch.nn.Module,
        ema_save: Any | None,
        optimizer: torch.optim.Optimizer,
        ref_model: torch.nn.Module,
        ema_rollout: Any | None = None,
        dgpo_projection_constraint_state: dict[str, Any] | None = None,
    ) -> None:
        if self._top_k <= 0:
            return
        if not math.isfinite(val_reward_mean):
            _log.warning("[DGPO] val/reward/mean is non-finite; skipping top-k checkpoint.")
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"dgpo-top-val_reward_mean={val_reward_mean:.6f}-"
            f"next_ep={dgpo_next_epoch}-step={global_step}.ckpt"
        )
        path = self._save_dir / fname
        save_lightning_compatible_checkpoint(
            path,
            model,
            ema_save,
            global_config,
            last_completed_epoch=last_completed_epoch,
            dgpo_next_epoch=dgpo_next_epoch,
            global_step=global_step,
            optimizer=optimizer,
            ref_model=ref_model,
            ema_rollout=ema_rollout,
            dgpo_projection_constraint_state=dgpo_projection_constraint_state,
        )

        heapq.heappush(self._worst_heap, (val_reward_mean, str(path)))

        # Keep ``best.ckpt`` symlink pointing to the highest val/reward/mean seen so far.
        best_score, best_path_str = max(self._worst_heap, key=lambda x: x[0])
        best_link = self._save_dir / "best.ckpt"
        try:
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(Path(best_path_str).name)
            _log.info(
                "[DGPO] best.ckpt → %s (val/reward/mean=%.6f)",
                Path(best_path_str).name,
                best_score,
            )
        except OSError as e:
            _log.warning("[DGPO] Could not update best.ckpt symlink: %s", e)

        while len(self._worst_heap) > self._top_k:
            _worst_score, worst_path = heapq.heappop(self._worst_heap)
            wp = Path(worst_path)
            if wp.is_file():
                try:
                    wp.unlink()
                    _log.info(
                        "[DGPO] Removed checkpoint outside top-%s: %s (val/reward/mean=%.6f)",
                        self._top_k,
                        wp.name,
                        _worst_score,
                    )
                except OSError as e:
                    _log.warning("[DGPO] Failed to remove old checkpoint %s: %s", wp, e)


_VAL_KIN_NUM_BINS = 50


@torch.no_grad()
def _val_pred_truth_kin_flat(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``pt`` (GeV, original scale), ``eta``, ``phi`` for selected-candidate pred vs truth (same slots).

    The first invisible feature is ``log1p(pT)``; this function inverts that via ``expm1`` so the
    returned ``ppt`` / ``tpt`` arrays are in GeV (original physics scale), not log space.
    """
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    b_idx = torch.arange(B, device=device)
    pred = candidates[k_sel, b_idx]
    if cartesian:
        t = batch_d["x_invisible_cartesian"]
        plp, pe, pp = cartesian_to_log_pt_eta_phi(pred[..., 0], pred[..., 1], pred[..., 2])
        tlp, te, tp = cartesian_to_log_pt_eta_phi(t[..., 0], t[..., 1], t[..., 2])
    else:
        t = batch_d["x_invisible"]
        plp, pe, pp = pred[..., 0], pred[..., 1], pred[..., 2]
        tlp, te, tp = t[..., 0], t[..., 1], t[..., 2]
    m = (mask > 0).reshape(B, N_nu)
    # Invert log1p to recover pT in GeV (original physics scale).
    ppt = np.expm1(plp[m].detach().float().cpu().numpy())
    pe = pe[m].detach().float().cpu().numpy()
    pp = pp[m].detach().float().cpu().numpy()
    tpt = np.expm1(tlp[m].detach().float().cpu().numpy())
    te = te[m].detach().float().cpu().numpy()
    tp = tp[m].detach().float().cpu().numpy()
    return ppt, pe, pp, tpt, te, tp


@torch.no_grad()
def _val_pred_truth_kin_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    cartesian: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``pt``/``eta``/``phi`` for all candidates vs repeated truth slots."""
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    mask = (mask > 0).reshape(B, N_nu)
    mask_k = mask.unsqueeze(0).expand(K, -1, -1)
    if cartesian:
        truth = batch_d["x_invisible_cartesian"]
        plp, peta, pphi = cartesian_to_log_pt_eta_phi(
            candidates[..., 0], candidates[..., 1], candidates[..., 2]
        )
        tlp, teta, tphi = cartesian_to_log_pt_eta_phi(
            truth[..., 0], truth[..., 1], truth[..., 2]
        )
    else:
        truth = batch_d["x_invisible"]
        plp, peta, pphi = candidates[..., 0], candidates[..., 1], candidates[..., 2]
        tlp, teta, tphi = truth[..., 0], truth[..., 1], truth[..., 2]
    tlp = tlp.unsqueeze(0).expand(K, -1, -1)
    teta = teta.unsqueeze(0).expand(K, -1, -1)
    tphi = tphi.unsqueeze(0).expand(K, -1, -1)
    ppt = np.expm1(plp[mask_k].detach().float().cpu().numpy())
    peta = peta[mask_k].detach().float().cpu().numpy()
    pphi = pphi[mask_k].detach().float().cpu().numpy()
    tpt = np.expm1(tlp[mask_k].detach().float().cpu().numpy())
    teta = teta[mask_k].detach().float().cpu().numpy()
    tphi = tphi[mask_k].detach().float().cpu().numpy()
    return ppt, peta, pphi, tpt, teta, tphi


@torch.no_grad()
def _val_pred_truth_feature_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    feature_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Masked flattened feature arrays for all candidates vs repeated truth slots."""
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    max_features = min(len(feature_names), int(candidates.shape[-1]), int(batch_d["x_invisible"].shape[-1]))
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    mask = (mask > 0).reshape(B, N_nu)
    mask_k = mask.unsqueeze(0).expand(K, -1, -1)
    truth = batch_d["x_invisible"]
    out: dict[str, np.ndarray] = {}
    for i in range(max_features):
        feature_name = str(feature_names[i])
        pred = candidates[..., i][mask_k].detach().float().cpu().numpy()
        target = truth[..., i].unsqueeze(0).expand(K, -1, -1)[mask_k].detach().float().cpu().numpy()
        out[f"{feature_name}_pred"] = pred
        out[f"{feature_name}_truth"] = target
    return out


@torch.no_grad()
def _truth_invisible_kin_phys(
    batch_d: dict[str, Any],
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Truth invisible features in physical units, shape ``(B, N_nu, 3)``.

    The batch stores physical (log-space) targets — ``x_invisible`` is ``[log1p(pT), η, φ]``
    and ``x_invisible_cartesian`` is ``[px, py, pz]`` — exactly the space the DDIM candidates
    are denormalized into. The model normalizes internally during forward, so NO extra
    denormalization is applied here (doing so would double-process and corrupt the values).
    """
    key = "x_invisible_cartesian" if cartesian else "x_invisible"
    raw = batch_d[key].to(device=device, dtype=dtype)
    return raw[..., :3]


@torch.no_grad()
def _kin_to_xyz(kin: Tensor, *, cartesian: bool) -> Tensor:
    """``(B, N, 3)`` kinematics → Cartesian momentum ``(B, N, 3)`` in GeV."""
    if cartesian:
        return kin[..., :3]
    return log_pt_eta_phi_to_cartesian(
        kin[..., 0].clamp(-10.0, 10.0),
        kin[..., 1],
        kin[..., 2],
    )


@torch.no_grad()
def _val_pred_truth_cartesian_flat(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``px``, ``py``, ``pz`` [GeV] for selected-candidate pred vs truth."""
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device) > 0
    else:
        mask = xm.to(device=device) > 0
    mask = mask.reshape(B, N_nu)
    b_idx = torch.arange(B, device=device)
    pred_kin = candidates[k_sel, b_idx][:, :N_nu, :3]
    truth_kin = _truth_invisible_kin_phys(
        batch_d, cartesian=cartesian, device=device, dtype=dtype
    )[:, :N_nu, :]
    pred_xyz = _kin_to_xyz(pred_kin, cartesian=cartesian)
    truth_xyz = _kin_to_xyz(truth_kin, cartesian=cartesian)
    m = mask
    ppx = pred_xyz[..., 0][m].detach().float().cpu().numpy()
    ppy = pred_xyz[..., 1][m].detach().float().cpu().numpy()
    ppz = pred_xyz[..., 2][m].detach().float().cpu().numpy()
    tpx = truth_xyz[..., 0][m].detach().float().cpu().numpy()
    tpy = truth_xyz[..., 1][m].detach().float().cpu().numpy()
    tpz = truth_xyz[..., 2][m].detach().float().cpu().numpy()
    return ppx, ppy, ppz, tpx, tpy, tpz


@torch.no_grad()
def _val_pred_truth_cartesian_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``px``, ``py``, ``pz`` [GeV] for all candidates vs repeated truth."""
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device) > 0
    else:
        mask = xm.to(device=device) > 0
    mask = mask.reshape(B, N_nu)
    mask_k = mask.unsqueeze(0).expand(K, -1, -1)
    pred_xyz = candidates[..., :3]
    truth_xyz = _truth_invisible_kin_phys(
        batch_d, cartesian=True, device=device, dtype=dtype
    )[:, :N_nu, :].unsqueeze(0).expand(K, -1, -1, -1)
    ppx = pred_xyz[..., 0][mask_k].detach().float().cpu().numpy()
    ppy = pred_xyz[..., 1][mask_k].detach().float().cpu().numpy()
    ppz = pred_xyz[..., 2][mask_k].detach().float().cpu().numpy()
    tpx = truth_xyz[..., 0][mask_k].detach().float().cpu().numpy()
    tpy = truth_xyz[..., 1][mask_k].detach().float().cpu().numpy()
    tpz = truth_xyz[..., 2][mask_k].detach().float().cpu().numpy()
    return ppx, ppy, ppz, tpx, tpy, tpz


def _pc_row_to_4vec_torch(pc_row: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """First four point-cloud features: ``logE, logPt, η, φ`` → ``(E, px, py, pz)`` [GeV]."""
    row = pc_row[..., :4]
    log_e, log_pt, eta, phi = row.unbind(dim=-1)
    pt = torch.expm1(log_pt.clamp(-10.0, 10.0))
    e = torch.expm1(log_e.clamp(-10.0, 10.0))
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return e, px, py, pz


def _nu_kin_to_4vec_torch(kin: Tensor, *, cartesian: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Neutrino slot kinematics → massless four-vector components [GeV]."""
    if cartesian:
        log_pt, eta, phi = cartesian_to_log_pt_eta_phi(kin[..., 0], kin[..., 1], kin[..., 2])
    else:
        log_pt, eta, phi = kin[..., 0], kin[..., 1], kin[..., 2]
    pt = torch.expm1(log_pt.clamp(-10.0, 10.0))
    e = pt * torch.cosh(eta)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return e, px, py, pz


def _add_4vec_torch(
    a: tuple[Tensor, Tensor, Tensor, Tensor],
    b: tuple[Tensor, Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def _mass_from_4vec_torch(
    e: Tensor, px: Tensor, py: Tensor, pz: Tensor
) -> Tensor:
    return torch.sqrt(torch.clamp(e * e - px * px - py * py - pz * pz, min=0.0))


@torch.no_grad()
def _val_mass_reconstruction_masses(
    batch_d: dict[str, Any],
    nu_kin: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """W and top masses [GeV] for valid TT2L events (both tops, flattened).

    Uses ground-truth ``assignments-indices`` to pick b/lepton from the point cloud. The
    point cloud ``x`` and ``nu_kin`` are already physical (log-space) values stored in the
    batch — the model normalizes internally during forward — so NO denormalization is
    applied here. ``nu_kin`` is ``(B, 2, 3)`` physical invisible kinematics (pred or truth).
    """
    assign = batch_d.get("assignments-indices")
    assign_m = batch_d.get("assignments-mask")
    if not isinstance(assign, Tensor) or assign.dim() != 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if not isinstance(assign_m, Tensor):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    B = int(batch_d["x"].shape[0])
    if assign.shape[0] != B or assign.shape[1] < 2 or assign.shape[2] < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    pc = batch_d["x"].to(device=device, dtype=dtype)

    b_idx = torch.arange(B, device=device)
    idx_ok = (assign[..., :2] >= 0).all(dim=-1)
    event_ok = (
        get_event_valid_mask(batch_d, B, device, dtype).reshape(B) > 0
    ) & (assign_m > 0).all(dim=-1) & idx_ok.all(dim=-1)
    if not bool(event_ok.any().item()):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    nu_kin = nu_kin.to(device=device, dtype=dtype)
    nu1 = _nu_kin_to_4vec_torch(nu_kin[:, 0, :], cartesian=cartesian)
    nu2 = _nu_kin_to_4vec_torch(nu_kin[:, 1, :], cartesian=cartesian)

    w_masses: list[Tensor] = []
    top_masses: list[Tensor] = []
    for r in range(2):
        b_pc = _pc_row_to_4vec_torch(pc[b_idx, assign[:, r, 0].long()])
        l_pc = _pc_row_to_4vec_torch(pc[b_idx, assign[:, r, 1].long()])
        nu = nu1 if r == 0 else nu2
        w = _add_4vec_torch(l_pc, nu)
        top = _add_4vec_torch(b_pc, w)
        w_masses.append(_mass_from_4vec_torch(*w))
        top_masses.append(_mass_from_4vec_torch(*top))

    w_all = torch.stack(w_masses, dim=-1)[event_ok].reshape(-1)
    top_all = torch.stack(top_masses, dim=-1)[event_ok].reshape(-1)
    w_np = w_all.detach().float().cpu().numpy()
    top_np = top_all.detach().float().cpu().numpy()
    finite = np.isfinite(w_np) & np.isfinite(top_np)
    return w_np[finite], top_np[finite]


def _val_overlay_kin_figure(
    counts_truth: np.ndarray,
    counts_pred: np.ndarray,
    bin_edges: np.ndarray,
    title: str,
    *,
    pred_label: str = "Pred (val)",
    counts_ref: np.ndarray | None = None,
    ref_label: str = "Ref (frozen)",
    xlabel: str = "Value",
) -> Any:
    """1D density overlay (truth vs current-policy prediction, optionally also reference policy), EveNet-style, as ``wandb.Image``.

    Args:
        counts_truth: Histogram counts for truth.
        counts_pred: Histogram counts for current-policy prediction.
        bin_edges: Bin edges array (length = len(counts_truth) + 1).
        title: Figure title.
        pred_label: Legend label for the current-policy series.
        counts_ref: Optional histogram counts for the frozen reference policy.
        ref_label: Legend label for the reference series.
        xlabel: x-axis label.
    """
    import wandb

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = float(bin_edges[1] - bin_edges[0])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    nt = np.sum(counts_truth) * w + 1e-12
    npred = np.sum(counts_pred) * w + 1e-12
    ax.plot(
        centers,
        counts_truth / nt,
        label="Truth",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax.plot(
        centers,
        counts_pred / npred,
        label=pred_label,
        linewidth=2.0,
        marker="s",
        markersize=4,
    )
    if counts_ref is not None:
        nref = np.sum(counts_ref) * w + 1e-12
        ax.plot(
            centers,
            counts_ref / nref,
            label=ref_label,
            linewidth=2.0,
            marker="^",
            markersize=4,
            linestyle="--",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def _val_selected_delta_arrays(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """Selected-candidate validation residual arrays for profile and response plots."""
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        slot_mask = xm.squeeze(-1).to(device=device) > 0
    else:
        slot_mask = xm.to(device=device) > 0
    slot_mask = slot_mask.reshape(B, N_nu)
    event_valid = get_event_valid_mask(batch_d, B, device, dtype).reshape(B) > 0
    valid_slots = slot_mask & event_valid.unsqueeze(-1)

    b_idx = torch.arange(B, device=device)
    pred = candidates[k_sel, b_idx]
    if _supports_legacy_invisible_kinematics(
        cartesian=cartesian,
        feature_dim=min(int(pred.shape[-1]), int(batch_d["x_invisible"].shape[-1])),
    ):
        if cartesian:
            truth = batch_d["x_invisible_cartesian"].to(device=device, dtype=dtype)
            plp, pred_eta, _pred_phi = cartesian_to_log_pt_eta_phi(
                pred[..., 0], pred[..., 1], pred[..., 2]
            )
            tlp, truth_eta, _truth_phi = cartesian_to_log_pt_eta_phi(
                truth[..., 0], truth[..., 1], truth[..., 2]
            )
            pred_xyz = pred[:, :2, :3].contiguous()
            truth_xyz = truth[:, :2, :3].contiguous()
        else:
            truth = batch_d["x_invisible"].to(device=device, dtype=dtype)
            plp, pred_eta = pred[..., 0], pred[..., 1]
            pred_phi = pred[..., 2]
            tlp, truth_eta = truth[..., 0], truth[..., 1]
            truth_phi = truth[..., 2]
            pred_xyz = log_pt_eta_phi_to_cartesian(
                plp.clamp(-10.0, 10.0), pred_eta, pred_phi
            )[:, :2, :].contiguous()
            truth_xyz = log_pt_eta_phi_to_cartesian(
                tlp.clamp(-10.0, 10.0), truth_eta, truth_phi
            )[:, :2, :].contiguous()

        pred_pt = torch.expm1(plp.clamp(-10.0, 10.0))
        truth_pt = torch.expm1(tlp.clamp(-10.0, 10.0))
        delta_pt = pred_pt - truth_pt
        delta_eta = pred_eta - truth_eta
        delta_xyz = pred_xyz - truth_xyz

        slot_count = valid_slots.sum(dim=-1)
        valid_events_with_slots = slot_count > 0
        pt_delta_event_mean = (
            (delta_pt * valid_slots.to(delta_pt.dtype)).sum(dim=-1)
            / slot_count.clamp(min=1).to(delta_pt.dtype)
        )

        return {
            "pt_truth": truth_pt[valid_slots].detach().float().cpu().numpy(),
            "pt_delta": delta_pt[valid_slots].detach().float().cpu().numpy(),
            "px_delta": delta_xyz[..., 0][valid_slots].detach().float().cpu().numpy(),
            "py_delta": delta_xyz[..., 1][valid_slots].detach().float().cpu().numpy(),
            "pz_delta": delta_xyz[..., 2][valid_slots].detach().float().cpu().numpy(),
            "eta_truth": truth_eta[valid_slots].detach().float().cpu().numpy(),
            "eta_delta": delta_eta[valid_slots].detach().float().cpu().numpy(),
            "pt_delta_event_mean": pt_delta_event_mean[valid_events_with_slots]
            .detach()
            .float()
            .cpu()
            .numpy(),
        }

    truth = batch_d["x_invisible"].to(device=device, dtype=dtype)
    feature_dim = min(int(pred.shape[-1]), int(truth.shape[-1]))
    feature_names = list(_invisible_feature_names())
    if len(feature_names) < feature_dim:
        feature_names.extend(f"feature_{index}" for index in range(len(feature_names), feature_dim))
    periodic_indices = set(
        index for index in _invisible_periodic_feature_indices() if 0 <= index < feature_dim
    )
    pred_sel = pred[..., :feature_dim]
    truth_sel = truth[..., :feature_dim]
    delta = pred_sel - truth_sel
    for index in periodic_indices:
        delta[..., index] = wrapped_delta_phi(pred_sel[..., index], truth_sel[..., index])

    out: dict[str, np.ndarray] = {}
    for index, feature_name in enumerate(feature_names[:feature_dim]):
        out[f"{feature_name}_truth"] = (
            truth_sel[..., index][valid_slots].detach().float().cpu().numpy()
        )
        out[f"{feature_name}_delta"] = (
            delta[..., index][valid_slots].detach().float().cpu().numpy()
        )
    return out


def _concat_np_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate non-empty numpy chunks into one float64 vector."""
    parts = [np.asarray(x, dtype=np.float64).reshape(-1) for x in chunks if x.size > 0]
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float64)


def _gather_val_array_dict(
    local_arrays: dict[str, np.ndarray],
    *,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    """Gather per-rank validation arrays to rank 0 and concatenate matching keys."""
    if world_size <= 1:
        return {
            k: np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in local_arrays.items()
        }
    if rank == 0:
        gathered: list[Any] = [None] * world_size
        dist.gather_object(local_arrays, object_gather_list=gathered, dst=0)
        keys = set(local_arrays)
        for part in gathered:
            if isinstance(part, dict):
                keys.update(part)
        merged: dict[str, np.ndarray] = {}
        for key in keys:
            chunks = [
                np.asarray(part.get(key, np.array([], dtype=np.float64)), dtype=np.float64)
                .reshape(-1)
                for part in gathered
                if isinstance(part, dict)
            ]
            merged[key] = _concat_np_chunks(chunks)
        return merged
    dist.gather_object(local_arrays, dst=0)
    return {}


def _profile_fit_metrics(
    profile_name: str,
    truth_value: np.ndarray,
    delta_value: np.ndarray,
) -> tuple[float, float]:
    """Fit binned mean residual with ``delta_mean = slope * truth + intercept``."""
    bin_edges = _profile_bin_edges(profile_name, [truth_value])
    centers, means, _errors, counts = _binned_delta_profile(
        truth_value, delta_value, bin_edges=bin_edges
    )
    keep = np.isfinite(centers) & np.isfinite(means) & (counts > 0)
    if int(np.sum(keep)) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(centers[keep], means[keep], deg=1)
    slope_f = float(slope)
    intercept_f = float(intercept)
    zero = (
        float(-intercept_f / slope_f)
        if math.isfinite(slope_f) and abs(slope_f) > 1e-12
        else float("nan")
    )
    return slope_f, zero


def _validation_delta_profile_figure(
    truth_current: np.ndarray,
    delta_current: np.ndarray,
    *,
    profile_name: str,
    title: str,
    truth_initial: np.ndarray | None = None,
    delta_initial: np.ndarray | None = None,
) -> Any:
    """Validation profile plot of mean selected-candidate residual versus truth value."""
    import wandb

    initial_arrays = []
    if truth_initial is not None and delta_initial is not None:
        initial_arrays = [truth_initial]
    x_label, y_label, display = _profile_axis_labels(profile_name)
    bin_edges = _profile_bin_edges(profile_name, [truth_current] + initial_arrays)
    centers, mean_cur, err_cur, counts = _binned_delta_profile(
        truth_current, delta_current, bin_edges=bin_edges
    )
    mean_init = err_init = None
    if truth_initial is not None and delta_initial is not None:
        _centers, mean_init, err_init, _counts = _binned_delta_profile(
            truth_initial, delta_initial, bin_edges=bin_edges
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if mean_init is not None and err_init is not None:
        keep_init = np.isfinite(mean_init)
        if np.any(keep_init):
            ax.errorbar(
                centers[keep_init],
                mean_init[keep_init],
                yerr=err_init[keep_init],
                fmt="o--",
                linewidth=1.6,
                markersize=4,
                capsize=2,
                color="#7f7f7f",
                label=f"initial validation {display}",
            )
    keep_cur = np.isfinite(mean_cur)
    if np.any(keep_cur):
        ax.errorbar(
            centers[keep_cur],
            mean_cur[keep_cur],
            yerr=err_cur[keep_cur],
            fmt="s-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            color="#1f77b4",
            label=f"current validation {display}",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_slope_history_figure(
    epochs: list[float],
    slopes: list[float],
    *,
    profile_name: str,
) -> Any:
    """History plot: epoch number versus fitted validation residual-profile slope."""
    import wandb

    _x_label, _y_label, display = _profile_axis_labels(profile_name)
    ep = np.asarray(epochs, dtype=np.float64)
    sl = np.asarray(slopes, dtype=np.float64)
    keep = np.isfinite(ep) & np.isfinite(sl)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if np.any(keep):
        ax.plot(ep[keep], sl[keep], "o-", linewidth=1.8, markersize=4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Fitted {display} residual slope")
    ax.set_title(f"Validation {display} residual-profile slope over epochs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_epoch_history_figure(
    epochs: list[float],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    zero_line: bool = True,
) -> Any:
    """History plot with epoch on the x-axis and one validation diagnostic on y."""
    import wandb

    ep = np.asarray(epochs, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(ep) & np.isfinite(val)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if np.any(keep):
        ax.plot(ep[keep], val[keep], "o-", linewidth=1.8, markersize=4)
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_zero_vs_slope_figure(
    slopes: list[float],
    zero_points: list[float],
    epochs: list[float],
    *,
    profile_name: str,
) -> Any:
    """History plot: fitted slope versus truth value where mean residual crosses zero."""
    import wandb

    x_label, _y_label, display = _profile_axis_labels(profile_name)
    sl = np.asarray(slopes, dtype=np.float64)
    zp = np.asarray(zero_points, dtype=np.float64)
    ep = np.asarray(epochs, dtype=np.float64)
    keep = np.isfinite(sl) & np.isfinite(zp)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if np.any(keep):
        sc = ax.scatter(sl[keep], zp[keep], c=ep[keep], cmap="viridis", s=34)
        ax.plot(sl[keep], zp[keep], "-", linewidth=1.1, alpha=0.55)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Epoch")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(f"Fitted {display} residual slope")
    ax.set_ylabel(f"{x_label} where mean residual = 0")
    ax.set_title(f"Validation {display} zero-crossing versus slope")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _response_matrix_figure(
    initial: np.ndarray,
    current: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Any:
    """2D response matrix comparing event-level initial and current validation values."""
    import wandb

    x = np.asarray(initial, dtype=np.float64).reshape(-1)
    y = np.asarray(current, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)

    stacked = np.concatenate([x, y], axis=0)
    lo, hi = [float(v) for v in np.nanpercentile(stacked, [1.0, 99.0])]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
        lo, hi = center - 1.0, center + 1.0
    pad = max(0.05 * (hi - lo), 1e-6)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hist = ax.hist2d(x, y, bins=50, range=[[lo, hi], [lo, hi]], cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="Events")
    ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(0.0, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.axvline(0.0, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _truth_pred_matrix_figure(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    bin_edges: np.ndarray | None = None,
) -> Any:
    """2D density matrix for truth-vs-pred comparisons."""
    import wandb

    x = np.asarray(truth, dtype=np.float64).reshape(-1)
    y = np.asarray(pred, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)

    if bin_edges is not None and len(bin_edges) >= 2:
        edges = np.asarray(bin_edges, dtype=np.float64)
        lo = float(edges[0])
        hi = float(edges[-1])
    else:
        stacked = np.concatenate([x, y], axis=0)
        lo, hi = [float(v) for v in np.nanpercentile(stacked, [1.0, 99.0])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        pad = max(0.05 * (hi - lo), 1e-6)
        lo -= pad
        hi += pad
        edges = np.linspace(lo, hi, 51)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hist = ax.hist2d(x, y, bins=[edges, edges], cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="Samples")
    ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _dgpo_should_run_validation_epoch(epoch: int, every_n_epochs: int) -> bool:
    """True when end-of-epoch validation should run (matches EveNet ``eval_metrics_every_n_epochs``)."""
    n = max(1, int(every_n_epochs))
    return (int(epoch) + 1) % n == 0


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ema_save: Any | None,
    val_loader: Any | None,
    sampler: DDIMSampler,
    reward_agg: RewardAggregator,
    *,
    val_K: int,
    num_ddim_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    cartesian: bool,
    compute_winrate: bool,
    epoch: int | None = None,
    est_total_batches: int | None = None,
    val_log_batches: bool = True,
    val_rollout_parallel_chains: int = 1,
    val_tqdm_k_chains: bool = True,
    val_tqdm_ddim: bool = False,
    max_batches: int | None = None,
    initial_state: dict[str, np.ndarray] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """One pass over the validation dataset; all tensors under ``torch.no_grad()``.

    Validation uses ``val_K`` candidates per event (default **1** in config), independent of training ``K``.
    Typical setup: **one** current-policy DDIM sample per event; if ``compute_winrate``, **one**
    reference-policy DDIM sample per event for reward-based winrate (no extra multi-candidate rollout for val).

    Under Ray Train, every rank receives its own validation shard via
    ``ray.train.get_dataset_shard("validation")``.  Each rank iterates its own shard;
    a cross-rank ``all_reduce(MIN)`` on the "has-more" flag keeps DDP collectives
    synchronized (loop terminates as soon as **any** rank exhausts its shard).
    Accumulators are all-reduced at the end so every rank returns identical aggregates.

    When ``ema_save`` is set, candidate generation uses the save-EMA shadow; checkpoint
    ``state_dict`` still stores live trainable weights for resume.

    When ``val_log_batches`` is True, logs start/end timing per validation batch so long DDIM runs are
    not silent. ``val_tqdm_k_chains`` wraps the ``val_K`` sequential DDIM calls in a tqdm bar;
    ``val_tqdm_ddim`` adds an inner bar for each DDIM chain (verbose).

    If ``max_batches`` is set, stop after that many local batches per rank (partial val).

    """
    is_rank0 = rank == 0
    model.eval()
    ref_model.eval()
    freeze_reference_model(ref_model)

    core = _unwrap_core_evenet(model)

    ep_str = f"epoch {epoch}" if epoch is not None else "val"
    if is_rank0:
        est_msg = (
            f"≈{est_total_batches} batches (ceil(val_events/batch_size))"
            if est_total_batches is not None and est_total_batches > 0
            else "unknown batch count (streaming)"
        )
        if max_batches is not None and max_batches > 0:
            est_msg = f"cap {max_batches} batches (partial val); full pass would be {est_msg}"
        if val_log_batches:
            _log.info("[DGPO] val: starting pass (%s, %s, %s GPUs).", ep_str, est_msg, world_size)

    t_epoch = time.perf_counter()
    n_val_batches = 0
    sum_r = 0.0
    cnt_r = 0
    sum_win = 0.0
    cnt_win = 0

    local_reward_chunks: list[np.ndarray] = []
    local_reward_event_chunks: list[np.ndarray] = []
    legacy_kinematics = _supports_legacy_invisible_kinematics(
        cartesian=cartesian,
        feature_dim=len(_invisible_feature_names()) or None,
    )
    winrate_enabled = _validation_winrate_enabled(
        compute_winrate=compute_winrate,
        cartesian=cartesian,
        feature_dim=len(_invisible_feature_names()) or None,
    )
    profile_feature_names = tuple(_validation_profile_feature_names(cartesian=cartesian))
    local_pt_delta_event_mean_chunks: list[np.ndarray] = []
    local_profile_chunks: dict[str, list[np.ndarray]] = {
        f"{profile_name}_truth": []
        for profile_name in profile_feature_names
    }
    local_profile_chunks.update({
        f"{profile_name}_delta": []
        for profile_name in profile_feature_names
    })
    all_plot_feature_names = _generation_monitor_feature_names(cartesian=cartesian)
    local_truth_pred_all_chunks: dict[str, list[np.ndarray]] = {
        f"{feature_name}_{suffix}": []
        for feature_name in all_plot_feature_names
        for suffix in ("truth", "pred", "ref")
    }
    # pT in GeV (original physics scale, after expm1 inversion of log1p).
    bin_pt_edges = _diagnostic_bin_edges("pt")
    bin_eta_edges = _diagnostic_bin_edges("eta")
    bin_phi_edges = _diagnostic_bin_edges("phi")
    # px/py track neutrino pT (rarely beyond a few hundred GeV); pz ~ pt*sinh(eta) has a much
    # wider longitudinal spread, so it needs a larger range or the histogram clips to ~empty.
    bin_px_edges = _diagnostic_bin_edges("px")
    bin_py_edges = _diagnostic_bin_edges("py")
    bin_pz_edges = _diagnostic_bin_edges("pz")
    bin_wmass_edges = _diagnostic_bin_edges("wmass")
    bin_topmass_edges = _diagnostic_bin_edges("topmass")
    num_diag_bins = len(bin_pt_edges) - 1
    # _p = current policy, _t = truth, _r = frozen reference policy
    h_pt_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_pt_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_pt_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_e_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_e_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_e_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_p_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_p_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_p_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_x_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_x_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_x_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_y_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_y_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_y_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_z_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_z_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_z_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_wm_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_wm_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_wm_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_tm_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_tm_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_tm_r = np.zeros(num_diag_bins, dtype=np.float64)

    val_iter = iter(val_loader) if val_loader is not None else None
    if val_iter is None:
        if is_rank0 and val_log_batches:
            _log.warning("[DGPO] val: rank=%s has no val shard; returning empty metrics.", rank)
        return {
            "val/reward/mean": float("nan"),
            "val/reward/median": float("nan"),
            "val/reward/p10": float("nan"),
            "val/reward/p30": float("nan"),
            "val/reward/p70": float("nan"),
            "val/reward/p90": float("nan"),
            "val/winrate": float("nan"),
        }

    batch_round = 0
    while True:
        batch_cpu, has_more = _next_batch_synced(
            val_iter, world_size=world_size, device=device
        )
        if not has_more or batch_cpu is None:
            break

        batch_round += 1
        n_val_batches += 1
        batch_d = batch_to_device(batch_cpu, device)
        B = int(batch_d["x"].shape[0])

        if is_rank0 and val_log_batches:
            batch_suffix = (
                f" {batch_round}/{est_total_batches}"
                if est_total_batches is not None and est_total_batches > 0
                else f" {batch_round}"
            )
            _log.info(
                "[DGPO] val round%s: B=%s×%s GPUs | current policy: val_K=%s DDIM chains (%s steps each)...",
                batch_suffix,
                B,
                world_size,
                val_K,
                num_ddim_steps,
            )

        buf: dict[str, Tensor] = {}
        if ema_save is not None:
            buf = _save_trainable_weights(model)
            ema_save.copy_to(core)
        t_gen = time.perf_counter()
        chain_desc = f"val DDIM ({ep_str})"
        try:
            candidates = generate_neutrino_candidates(
                core,
                batch_d,
                sampler,
                K=val_K,
                num_ddim_steps=num_ddim_steps,
                device=device,
                parallel_chains=val_rollout_parallel_chains,
                tqdm_k_chains=val_tqdm_k_chains and is_rank0,
                use_tqdm_ddim=val_tqdm_ddim and is_rank0,
                chain_progress_desc=chain_desc,
            )
        finally:
            if buf:
                _restore_trainable_weights(model, buf)

        t_after_cur = time.perf_counter()
        if is_rank0 and val_log_batches:
            _log.info(
                "[DGPO] val round%s: generation done in %.1fs.",
                batch_suffix,
                t_after_cur - t_gen,
            )

        rewards, _ = reward_agg.compute(candidates, batch_d)

        valid = get_event_valid_mask(batch_d, B, device, dtype).reshape(-1)
        m_sel = valid > 0
        vb = m_sel

        k_sel = _kin_hist_candidate_indices_per_event(
            rewards, candidates, batch_d, cartesian=cartesian
        )

        if bool(vb.any().item()):
            if val_K == 1:
                r_per_event = rewards[0, vb]
            else:
                r_per_event = rewards[:, vb].max(dim=0).values
            sum_r += float(r_per_event.sum().detach().cpu().item())
            cnt_r += int(vb.sum().item())
            r_per_event_np = r_per_event.detach().float().cpu().numpy()
            local_reward_chunks.append(r_per_event_np)
            local_reward_event_chunks.append(r_per_event_np)
        selected_delta_arrays = _val_selected_delta_arrays(
            candidates,
            batch_d,
            k_sel,
            cartesian=cartesian,
            device=device,
            dtype=dtype,
        )
        for key in local_profile_chunks:
            local_profile_chunks[key].append(
                selected_delta_arrays.get(key, np.array([], dtype=np.float64))
            )
        if "pt_delta_event_mean" in selected_delta_arrays:
            local_pt_delta_event_mean_chunks.append(
                selected_delta_arrays["pt_delta_event_mean"]
            )
        if not cartesian:
            feature_arrays = _val_pred_truth_feature_flat_all_candidates(
                candidates,
                batch_d,
                feature_names=all_plot_feature_names,
                device=device,
            )
            for key, values in feature_arrays.items():
                local_truth_pred_all_chunks[key].append(values)

        if legacy_kinematics:
            ppt, peta, pphi, tpt, teta, tphi = _val_pred_truth_kin_flat(
                candidates,
                batch_d,
                k_sel,
                cartesian=cartesian,
                device=device,
            )
            h_pt_p += np.histogram(ppt, bins=bin_pt_edges)[0]
            h_pt_t += np.histogram(tpt, bins=bin_pt_edges)[0]
            h_e_p += np.histogram(peta, bins=bin_eta_edges)[0]
            h_e_t += np.histogram(teta, bins=bin_eta_edges)[0]
            h_p_p += np.histogram(pphi, bins=bin_phi_edges)[0]
            h_p_t += np.histogram(tphi, bins=bin_phi_edges)[0]
            if cartesian:
                all_px_p, all_py_p, all_pz_p, all_px_t, all_py_t, all_pz_t = (
                    _val_pred_truth_cartesian_flat_all_candidates(
                        candidates,
                        batch_d,
                        device=device,
                        dtype=dtype,
                    )
                )
                local_truth_pred_all_chunks["px_truth"].append(all_px_t)
                local_truth_pred_all_chunks["px_pred"].append(all_px_p)
                local_truth_pred_all_chunks["py_truth"].append(all_py_t)
                local_truth_pred_all_chunks["py_pred"].append(all_py_p)
                local_truth_pred_all_chunks["pz_truth"].append(all_pz_t)
                local_truth_pred_all_chunks["pz_pred"].append(all_pz_p)

            ppx, ppy, ppz, tpx, tpy, tpz = _val_pred_truth_cartesian_flat(
                candidates,
                batch_d,
                k_sel,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            h_x_p += np.histogram(ppx, bins=bin_px_edges)[0]
            h_x_t += np.histogram(tpx, bins=bin_px_edges)[0]
            h_y_p += np.histogram(ppy, bins=bin_py_edges)[0]
            h_y_t += np.histogram(tpy, bins=bin_py_edges)[0]
            h_z_p += np.histogram(ppz, bins=bin_pz_edges)[0]
            h_z_t += np.histogram(tpz, bins=bin_pz_edges)[0]

            b_idx = torch.arange(B, device=device)
            pred_nu_kin = candidates[k_sel, b_idx][:, :2, :3]
            truth_nu_kin = _truth_invisible_kin_phys(
                batch_d, cartesian=cartesian, device=device, dtype=dtype
            )[:, :2, :]
            w_p, top_p = _val_mass_reconstruction_masses(
                batch_d,
                pred_nu_kin,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            w_t, top_t = _val_mass_reconstruction_masses(
                batch_d,
                truth_nu_kin,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            if w_p.size:
                h_wm_p += np.histogram(w_p, bins=bin_wmass_edges)[0]
            if top_p.size:
                h_tm_p += np.histogram(top_p, bins=bin_topmass_edges)[0]
            if w_t.size:
                h_wm_t += np.histogram(w_t, bins=bin_wmass_edges)[0]
            if top_t.size:
                h_tm_t += np.histogram(top_t, bins=bin_topmass_edges)[0]

        # Always run one ref-policy DDIM pass (K=1) for val_neutrino overlays; reuse for winrate.
        ref_core = _unwrap_core_evenet(ref_model)
        if is_rank0 and val_log_batches:
            _log.info("[DGPO] val round%s: reference policy DDIM (K=1)...", batch_suffix)
        t_ref = time.perf_counter()
        r_one = generate_neutrino_candidates(
            ref_core,
            batch_d,
            sampler,
            K=1,
            num_ddim_steps=num_ddim_steps,
            device=device,
            parallel_chains=1,
            tqdm_k_chains=False,
            use_tqdm_ddim=val_tqdm_ddim and is_rank0,
            chain_progress_desc=f"val ref DDIM ({ep_str})",
        )
        if is_rank0 and val_log_batches:
            _log.info(
                "[DGPO] val round%s: ref DDIM done in %.1fs.",
                batch_suffix,
                time.perf_counter() - t_ref,
            )
        k_sel_ref = torch.zeros(B, dtype=torch.long, device=device)
        if legacy_kinematics:
            rpt, reta, rphi, _, _, _ = _val_pred_truth_kin_flat(
                r_one, batch_d, k_sel_ref, cartesian=cartesian, device=device
            )
            h_pt_r += np.histogram(rpt, bins=bin_pt_edges)[0]
            h_e_r += np.histogram(reta, bins=bin_eta_edges)[0]
            h_p_r += np.histogram(rphi, bins=bin_phi_edges)[0]

            rpx, rpy, rpz, _, _, _ = _val_pred_truth_cartesian_flat(
                r_one,
                batch_d,
                k_sel_ref,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            h_x_r += np.histogram(rpx, bins=bin_px_edges)[0]
            h_y_r += np.histogram(rpy, bins=bin_py_edges)[0]
            h_z_r += np.histogram(rpz, bins=bin_pz_edges)[0]

        if legacy_kinematics:
            ref_nu_kin = r_one[k_sel_ref, b_idx][:, :2, :3]
            w_r, top_r = _val_mass_reconstruction_masses(
                batch_d,
                ref_nu_kin,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            if w_r.size:
                h_wm_r += np.histogram(w_r, bins=bin_wmass_edges)[0]
            if top_r.size:
                h_tm_r += np.histogram(top_r, bins=bin_topmass_edges)[0]

        if not cartesian:
            ref_feature_arrays = _val_pred_truth_feature_flat_all_candidates(
                r_one,
                batch_d,
                feature_names=all_plot_feature_names,
                device=device,
            )
            for key, values in ref_feature_arrays.items():
                local_truth_pred_all_chunks[key.replace("_pred", "_ref")].append(values)
        elif legacy_kinematics:
            all_px_r, all_py_r, all_pz_r, _, _, _ = _val_pred_truth_cartesian_flat_all_candidates(
                r_one,
                batch_d,
                device=device,
                dtype=dtype,
            )
            local_truth_pred_all_chunks["px_ref"].append(all_px_r)
            local_truth_pred_all_chunks["py_ref"].append(all_py_r)
            local_truth_pred_all_chunks["pz_ref"].append(all_pz_r)

        if winrate_enabled:
            rewards_ref, _ = reward_agg.compute(r_one, batch_d)
            r_cur = rewards.max(dim=0).values
            r_ref = rewards_ref[0]
            wins = (r_cur > r_ref) & m_sel & torch.isfinite(r_cur) & torch.isfinite(r_ref)
            w = wins.float().sum()
            nw = m_sel.sum()
            sum_win += float(w.detach().cpu().item())
            cnt_win += int(nw.detach().cpu().item())

        if max_batches is not None and max_batches > 0 and batch_round >= max_batches:
            if is_rank0 and val_log_batches:
                _log.info(
                    "[DGPO] val: stopping early at validation_max_batches=%s (partial val metrics).",
                    max_batches,
                )
            break

    if is_rank0 and val_log_batches:
        _log.info(
            "[DGPO] val: finished %s batches (%s rounds × %s GPUs) in %.1fs.",
            n_val_batches * world_size,
            n_val_batches,
            world_size,
            time.perf_counter() - t_epoch,
        )

    # All-reduce accumulators so every rank has the global totals.
    if world_size > 1:
        acc = torch.tensor(
            [sum_r, cnt_r, sum_win, cnt_win],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        a = acc.cpu().tolist()
        sum_r, cnt_r = a[0], int(a[1])
        sum_win, cnt_win = a[2], int(a[3])

    hist_stack = np.stack(
        [
            h_pt_p, h_pt_t, h_pt_r,
            h_e_p, h_e_t, h_e_r,
            h_p_p, h_p_t, h_p_r,
            h_x_p, h_x_t, h_x_r,
            h_y_p, h_y_t, h_y_r,
            h_z_p, h_z_t, h_z_r,
            h_wm_p, h_wm_t, h_wm_r,
            h_tm_p, h_tm_t, h_tm_r,
        ]
    )
    t_hist = torch.from_numpy(hist_stack).to(device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(t_hist, op=dist.ReduceOp.SUM)
    hist_merged = t_hist.cpu().numpy()
    (
        h_pt_p, h_pt_t, h_pt_r,
        h_e_p, h_e_t, h_e_r,
        h_p_p, h_p_t, h_p_r,
        h_x_p, h_x_t, h_x_r,
        h_y_p, h_y_t, h_y_r,
        h_z_p, h_z_t, h_z_r,
        h_wm_p, h_wm_t, h_wm_r,
        h_tm_p, h_tm_t, h_tm_r,
    ) = [hist_merged[i] for i in range(24)]


    p10 = p30 = p50 = p70 = p90 = float("nan")
    if world_size > 1:
        if rank == 0:
            gathered: list[Any] = [None] * world_size
            dist.gather_object(
                local_reward_chunks,
                object_gather_list=gathered,
                dst=0,
            )
            merged_list: list[np.ndarray] = []
            for part in gathered:
                if part:
                    merged_list.extend(part)
            merged_r = (
                np.concatenate(merged_list, axis=0)
                if merged_list
                else np.array([], dtype=np.float64)
            )
            if merged_r.size > 0:
                p10, p30, p50, p70, p90 = [
                    float(x) for x in np.nanpercentile(merged_r, [10, 30, 50, 70, 90])
                ]
        else:
            dist.gather_object(local_reward_chunks, dst=0)
        pct_t = torch.tensor(
            [p10, p30, p50, p70, p90], dtype=torch.float64, device=device
        )
        dist.broadcast(pct_t, src=0)
        p10, p30, p50, p70, p90 = [float(x) for x in pct_t.cpu().tolist()]
    else:
        merged_list = local_reward_chunks
        merged_r = (
            np.concatenate(merged_list, axis=0)
            if merged_list
            else np.array([], dtype=np.float64)
        )
        if merged_r.size > 0:
            p10, p30, p50, p70, p90 = [
                float(x) for x in np.nanpercentile(merged_r, [10, 30, 50, 70, 90])
            ]

    def _mean(num: float, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    win_metric = _mean(sum_win, cnt_win) if winrate_enabled else float("nan")

    local_state = {
        "reward": _concat_np_chunks(local_reward_event_chunks),
        "pt_delta_mean": _concat_np_chunks(local_pt_delta_event_mean_chunks),
    }
    for profile_name in profile_feature_names:
        local_state[f"{profile_name}_truth"] = _concat_np_chunks(
            local_profile_chunks[f"{profile_name}_truth"]
        )
        local_state[f"{profile_name}_delta"] = _concat_np_chunks(
            local_profile_chunks[f"{profile_name}_delta"]
        )
    profile_compare_local = {}
    for profile_name in profile_feature_names:
        profile_compare_local[f"{profile_name}_truth"] = local_state[f"{profile_name}_truth"]
        profile_compare_local[f"{profile_name}_delta"] = local_state[f"{profile_name}_delta"]
    if initial_state is not None:
        for profile_name in profile_feature_names:
            for suffix in ("truth", "delta"):
                key = f"{profile_name}_{suffix}"
                profile_compare_local[f"initial_{key}"] = np.asarray(
                    initial_state.get(key, np.array([], dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
    profile_merged = _gather_val_array_dict(
        profile_compare_local, rank=rank, world_size=world_size
    )

    response_initial_state = initial_state
    if response_initial_state is None and epoch == -1:
        # The baseline pass should still create the W&B image series; later epochs
        # then update the same key with initial-vs-current heatmaps.
        response_initial_state = local_state

    response_merged: dict[str, np.ndarray] = {}
    if response_initial_state is not None:
        init_reward = np.asarray(
            response_initial_state.get("reward", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        init_pt_delta = np.asarray(
            response_initial_state.get("pt_delta_mean", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        n_reward = min(init_reward.size, local_state["reward"].size)
        n_pt = min(init_pt_delta.size, local_state["pt_delta_mean"].size)
        response_merged = _gather_val_array_dict(
            {
                "reward_initial": init_reward[:n_reward],
                "reward_current": local_state["reward"][:n_reward],
                "pt_delta_initial": init_pt_delta[:n_pt],
                "pt_delta_current": local_state["pt_delta_mean"][:n_pt],
            },
            rank=rank,
            world_size=world_size,
        )
    truth_pred_all_merged = _gather_val_array_dict(
        {
            key: _concat_np_chunks(chunks)
            for key, chunks in local_truth_pred_all_chunks.items()
        },
        rank=rank,
        world_size=world_size,
    )

    out: dict[str, Any] = {
        "val/reward/mean": _mean(sum_r, cnt_r),
        "val/reward/median": p50,
        "val/reward/p10": p10,
        "val/reward/p30": p30,
        "val/reward/p70": p70,
        "val/reward/p90": p90,
        "val/winrate": win_metric,
        "_val_initial_state": local_state,
    }

    _val_kin_suffix = f"val: {val_K} candidate{'s' if val_K != 1 else ''} vs truth"
    _pred_lbl = "Pred (val)" if val_K == 1 else f"Pred (val, best-of-{val_K})"
    if is_rank0:
        for profile_name in profile_feature_names:
            truth_key = f"{profile_name}_truth"
            delta_key = f"{profile_name}_delta"
            truth_arr = profile_merged.get(truth_key, np.array([], dtype=np.float64))
            delta_arr = profile_merged.get(delta_key, np.array([], dtype=np.float64))
            slope, zero_point = _profile_fit_metrics(
                profile_name, truth_arr, delta_arr
            )
            finite_delta = delta_arr[np.isfinite(delta_arr)]
            delta_mean = float(np.mean(finite_delta)) if finite_delta.size > 0 else float("nan")
            out[f"val_diagnostics/profile/{profile_name}/delta_mean"] = delta_mean
            out[f"val_diagnostics/profile/{profile_name}/slope"] = slope
            out[f"val_diagnostics/profile/{profile_name}/zero_delta_truth"] = zero_point
            out[
                f"val_diagnostics/profile/{profile_name}_delta_vs_truth_{profile_name}"
            ] = _validation_delta_profile_figure(
                truth_arr,
                delta_arr,
                profile_name=profile_name,
                title=f"Validation {profile_name} residual vs truth {profile_name}",
                truth_initial=profile_merged.get(f"initial_{truth_key}"),
                delta_initial=profile_merged.get(f"initial_{delta_key}"),
            )
        if response_initial_state is not None:
            out["val/response/reward_initial_vs_current"] = (
                _response_matrix_figure(
                    response_merged.get("reward_initial", np.array([], dtype=np.float64)),
                    response_merged.get("reward_current", np.array([], dtype=np.float64)),
                    xlabel="Initial validation reward",
                    ylabel="Current validation reward",
                    title="Validation 2D correlation: initial reward vs current reward",
                )
            )
            if (
                response_merged.get("pt_delta_initial", np.array([], dtype=np.float64)).size > 0
                or response_merged.get("pt_delta_current", np.array([], dtype=np.float64)).size > 0
            ):
                out["val/response/pt_delta_mean_initial_vs_current"] = (
                    _response_matrix_figure(
                        response_merged.get("pt_delta_initial", np.array([], dtype=np.float64)),
                        response_merged.get("pt_delta_current", np.array([], dtype=np.float64)),
                        xlabel="Initial event mean delta pT [GeV]",
                        ylabel="Current event mean delta pT [GeV]",
                        title="Validation 2D correlation: initial vs current event mean delta pT",
                    )
                )
        available_truth_pred_features = _available_truth_pred_features(
            truth_pred_all_merged,
            all_plot_feature_names,
        )
        for feature_name in available_truth_pred_features:
            truth_key = f"{feature_name}_truth"
            pred_key = f"{feature_name}_pred"
            out[f"val_neutrino/all/{feature_name}_truth_vs_pred"] = _truth_pred_matrix_figure(
                truth_pred_all_merged.get(truth_key, np.array([], dtype=np.float64)),
                truth_pred_all_merged.get(pred_key, np.array([], dtype=np.float64)),
                xlabel=f"Truth {feature_name}",
                ylabel=f"Pred {feature_name}",
                title=(
                    f"Validation 2D truth vs pred {feature_name} "
                    f"({val_K} candidate{'s' if val_K != 1 else ''}, all)"
                ),
                bin_edges=_generation_special_bin_edges(feature_name),
            )
            for metric_name, metric_value in _truth_pred_scalar_metrics(
                truth_pred_all_merged.get(truth_key, np.array([], dtype=np.float64)),
                truth_pred_all_merged.get(pred_key, np.array([], dtype=np.float64)),
            ).items():
                out[f"val_neutrino/all_metrics/{feature_name}/{metric_name}"] = metric_value
            bin_edges = _generation_special_bin_edges(feature_name)
            out[f"val_neutrino/jsd/current/{feature_name}"] = _array_histogram_jsd(
                truth_pred_all_merged.get(truth_key, np.array([], dtype=np.float64)),
                truth_pred_all_merged.get(pred_key, np.array([], dtype=np.float64)),
                bin_edges=bin_edges,
            )
            out[f"val_neutrino/jsd/ref/{feature_name}"] = _array_histogram_jsd(
                truth_pred_all_merged.get(truth_key, np.array([], dtype=np.float64)),
                truth_pred_all_merged.get(f"{feature_name}_ref", np.array([], dtype=np.float64)),
                bin_edges=bin_edges,
            )
        if legacy_kinematics:
            out["val_neutrino/pt"] = _val_overlay_kin_figure(
                h_pt_t,
                h_pt_p,
                bin_pt_edges,
                f"Neutrino pT [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_pt_r,
                xlabel="pT [GeV]",
            )
            out["val_neutrino/eta"] = _val_overlay_kin_figure(
                h_e_t,
                h_e_p,
                bin_eta_edges,
                f"Neutrino η ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_e_r,
                xlabel="η",
            )
            out["val_neutrino/phi"] = _val_overlay_kin_figure(
                h_p_t,
                h_p_p,
                bin_phi_edges,
                f"Neutrino φ ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_p_r,
                xlabel="φ [rad]",
            )
            out["val_neutrino/px"] = _val_overlay_kin_figure(
                h_x_t,
                h_x_p,
                bin_px_edges,
                f"Neutrino p_x [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_x_r,
                xlabel="p_x [GeV]",
            )
            out["val_neutrino/py"] = _val_overlay_kin_figure(
                h_y_t,
                h_y_p,
                bin_py_edges,
                f"Neutrino p_y [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_y_r,
                xlabel="p_y [GeV]",
            )
            out["val_neutrino/pz"] = _val_overlay_kin_figure(
                h_z_t,
                h_z_p,
                bin_pz_edges,
                f"Neutrino p_z [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_z_r,
                xlabel="p_z [GeV]",
            )
            out["val_neutrino/jsd/current/pt"] = _histogram_jsd(h_pt_t, h_pt_p)
            out["val_neutrino/jsd/current/eta"] = _histogram_jsd(h_e_t, h_e_p)
            out["val_neutrino/jsd/current/phi"] = _histogram_jsd(h_p_t, h_p_p)
            out["val_neutrino/jsd/current/px"] = _histogram_jsd(h_x_t, h_x_p)
            out["val_neutrino/jsd/current/py"] = _histogram_jsd(h_y_t, h_y_p)
            out["val_neutrino/jsd/current/pz"] = _histogram_jsd(h_z_t, h_z_p)
            out["val_neutrino/jsd/ref/pt"] = _histogram_jsd(h_pt_t, h_pt_r)
            out["val_neutrino/jsd/ref/eta"] = _histogram_jsd(h_e_t, h_e_r)
            out["val_neutrino/jsd/ref/phi"] = _histogram_jsd(h_p_t, h_p_r)
            out["val_neutrino/jsd/ref/px"] = _histogram_jsd(h_x_t, h_x_r)
            out["val_neutrino/jsd/ref/py"] = _histogram_jsd(h_y_t, h_y_r)
            out["val_neutrino/jsd/ref/pz"] = _histogram_jsd(h_z_t, h_z_r)
            out["val_mass/w_mass"] = _val_overlay_kin_figure(
                h_wm_t,
                h_wm_p,
                bin_wmass_edges,
                f"W mass reconstruction vs truth resonance ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_wm_r,
                xlabel="W mass [GeV]",
            )
            out["val_mass/top_mass"] = _val_overlay_kin_figure(
                h_tm_t,
                h_tm_p,
                bin_topmass_edges,
                f"Top mass reconstruction vs truth resonance ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_tm_r,
                xlabel="Top mass [GeV]",
            )
            out["val_mass/jsd/current/w_mass"] = _histogram_jsd(h_wm_t, h_wm_p)
            out["val_mass/jsd/current/top_mass"] = _histogram_jsd(h_tm_t, h_tm_p)
            out["val_mass/jsd/ref/w_mass"] = _histogram_jsd(h_wm_t, h_wm_r)
            out["val_mass/jsd/ref/top_mass"] = _histogram_jsd(h_tm_t, h_tm_r)
    return out


def dgpo_train_loop(cfg: dict[str, Any]) -> None:
    """Per-worker DGPO training loop launched by ``ray.train.torch.TorchTrainer``.

    Each Ray Train worker runs this function in its own process.  Ray Train
    initialises the torch distributed process group and gives each worker its
    own per-rank Ray Data shard via ``ray.train.get_dataset_shard``.  This
    function:

    1. Resolves rank / world-size / device from the Ray Train context.
    2. Pulls its own train (and validation) shard.
    3. Builds the EveNet backbone, EMA shadows, reference policy, and DGPO optimizer.
    4. Iterates the DGPO algorithm in lock-step across ranks (``_next_batch_synced``
       all-reduces the ``has-more`` flag so collectives stay aligned).
    5. Runs validation, checkpoint top-K save, and ``last`` checkpoint on rank 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = ray.train.get_context()
    rank = int(ctx.get_world_rank())
    world_size = int(ctx.get_world_size())
    local_rank = int(ctx.get_local_rank())
    is_rank0 = rank == 0
    device = ray.train.torch.get_device()

    # Earliest possible visibility check: emitted by every worker before any I/O or model build,
    # so the user can see the actual world_size immediately and confirm multi-node DDP is live.
    _log.info(
        "[DGPO][boot] rank=%s/%s local_rank=%s host=%s device=%s",
        rank, world_size, local_rank, os.uname().nodename, device,
    )

    config_path = Path(str(cfg["config_path"])).resolve()
    max_steps: int | None = cfg.get("max_steps")
    wandb_flag = bool(cfg.get("wandb", True))
    total_events = int(cfg["total_events"])
    val_events_in = cfg.get("val_events", 0)
    val_events: int | None = int(val_events_in) if val_events_in else None
    config_yaml_text = cfg.get("config_yaml", None)

    if not config_path.is_file():
        if not isinstance(config_yaml_text, str) or not config_yaml_text.strip():
            raise FileNotFoundError(
                f"DGPO worker cannot access config path {config_path} and no config_yaml payload was provided."
            )
        worker_runtime_dir = Path(
            tempfile.mkdtemp(prefix="dgpo_worker_runtime_", dir=os.environ.get("TMPDIR", None))
        )
        config_path = worker_runtime_dir / config_path.name
        config_path.write_text(config_yaml_text)
        _log.info(
            "[DGPO][boot] rank=%s materialized worker-local runtime config at %s",
            rank,
            config_path,
        )

    global_config.load_yaml(config_path)
    _assert_rl_enabled()
    platform_info = global_config.platform

    wandb_active = _start_wandb_run(disable=not wandb_flag) if is_rank0 else False

    # Per-rank Ray Data shards.  Ray Train assigns each worker a disjoint subset.
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("validation") if val_events else None

    batch_size = int(platform_info.batch_size)
    prefetch = int(getattr(platform_info, "prefetch_batches", 1))
    train_loader_cfg = {
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
        # Mirror EveNet pretraining: enable in-shard random shuffling.
        "local_shuffle_buffer_size": batch_size * prefetch,
    }
    # Keep validation order deterministic so pre-DGPO and later response matrices
    # compare the same validation rows in the same per-rank order.
    val_loader_cfg = {
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
    }

    bundle = load_evenet_model_for_dgpo(
        None,
        device,
        checkpoint_path=None,
        config=global_config,
    )
    eve_net = bundle.model

    ckpt_dict = None
    if bundle.checkpoint_path is not None:
        ckpt_dict = torch.load(
            str(bundle.checkpoint_path), map_location=device, weights_only=False
        )

    eve_net.train()
    apply_component_freezes(eve_net, global_config)
    ref_model = make_reference_model(
        eve_net, global_config, bundle.normalization_dict, device, checkpoint=ckpt_dict
    )
    ema_save = make_ema(eve_net, global_config, checkpoint=ckpt_dict, device=device)
    ema_rollout = make_ema_rollout(
        eve_net, global_config, checkpoint=ckpt_dict, device=device
    )

    # Wrap the diffusion-vector forward in a thin nn.Module, then let Ray Train's
    # ``prepare_model`` install DDP with the right device + process-group config.
    fw = _DGPODDPForward(eve_net)
    if world_size > 1:
        model = ray.train.torch.prepare_model(
            fw,
            parallel_strategy_kwargs={"find_unused_parameters": True},
        )
    else:
        model = fw

    dtype = next(eve_net.parameters()).dtype
    # DDIM is the only rollout sampler.
    sampler = DDIMSampler(device=device)
    reward_agg = build_reward_aggregator(
        eve_net, device, normalization_dict=bundle.normalization_dict
    )
    effective_batch = batch_size * world_size
    steps_per_epoch = max(1, math.ceil(total_events / effective_batch))
    train_opt_lr = global_config.options.Training
    warm_up_factor = float(train_opt_lr.get("learning_rate_warm_up_factor", 1.0))
    warmup_steps = max(1, math.ceil(warm_up_factor * steps_per_epoch))
    optimizer = build_optimizer(
        model,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        is_rank0=is_rank0,
    )

    start_epoch, global_step = parse_dgpo_resume_from_checkpoint(ckpt_dict)
    if ckpt_dict is not None and "dgpo_optimizer_state_dict" in ckpt_dict:
        try:
            optimizer.load_state_dict(ckpt_dict["dgpo_optimizer_state_dict"])
            if is_rank0:
                _log.info("[DGPO] Restored optimizer state from checkpoint.")
        except (ValueError, RuntimeError) as ex:
            if is_rank0:
                _log.warning(
                    "[DGPO] Could not load optimizer state (continuing fresh optimizer): %s", ex
                )

    dg = global_config.dgpo
    _vm_raw = dg.get("validation_max_batches", None)
    val_max_batches: int | None = None
    if _vm_raw is not None:
        val_max_batches = int(_vm_raw)
        if val_max_batches <= 0:
            if is_rank0:
                _log.warning(
                    "[DGPO] validation_max_batches=%s is not positive; running full validation.",
                    _vm_raw,
                )
            val_max_batches = None
    K = int(_dgpo_cfg_get(dg, "K", 1))
    val_K = max(1, int(dg.get("validation_K", 1)))
    rollout_parallel_chains = max(1, int(dg.get("rollout_parallel_chains", 1)))
    val_rollout_parallel_chains = max(
        1,
        int(dg.get("validation_rollout_parallel_chains", rollout_parallel_chains)),
    )
    validation_every_n_epochs = max(1, int(dg.get("validation_every_n_epochs", 1)))
    beta = float(_dgpo_cfg_get(dg, "beta", 1.0))
    # Training and validation use independent DDIM rollout-step budgets: training uses
    # num_ddim, validation uses num_ddim_val. The validation-specific key falls back
    # to the training value when unset (null) for backward-compatible behavior.
    num_ddim = int(_dgpo_cfg_get(dg, "num_ddim_steps", 1))
    _val_steps_raw = dg.get("validation_num_ddim_steps", None)
    num_ddim_val = int(_val_steps_raw) if _val_steps_raw is not None else num_ddim
    if is_rank0:
        _log.info(
            "[DGPO] DDIM rollout steps: training=%s, validation=%s. Parallel chains: training=%s, validation=%s.",
            num_ddim,
            num_ddim_val,
            rollout_parallel_chains,
            val_rollout_parallel_chains,
        )
    log_every = max(1, int(dg.get("log_every", 1)))
    diagnostic_plot_names, diagnostic_plot_every = _resolve_diagnostic_plot_settings(dg)
    log_reward_dist_every = max(
        1, int(dg.get("log_reward_dist_every", diagnostic_plot_every))
    )
    diagnostic_profile_accumulate_steps = max(
        1, int(dg.get("diagnostic_profile_accumulate_steps", 1))
    )
    num_train_timesteps = max(1, int(dg.get("num_train_timesteps", 1)))
    _adv_raw = dg.get("adv_clip_max", None)
    adv_clip_max_cfg: float | None = float(_adv_raw) if _adv_raw is not None else None
    grad_clip_norm_cfg = float(dg.get("grad_clip_norm", _GRAD_CLIP_NORM))
    policy_eval_t_min_cfg = float(dg.get("policy_eval_t_min", 0.0))
    policy_eval_t_max_cfg = float(dg.get("policy_eval_t_max", 1.0))
    # Frozen DGPO method (hardwired in train_step): z-score advantages, shared noise,
    # accumulated sub-step gradients into one AdamW update, rollout EMA always on;
    # no PPO clip or velocity KL anchor.

    proj_cfg_startup = resolve_projection_constraint_config(dg)
    constraint_ckpt_blob = (
        (
            ckpt_dict.get(_DGPO_CONSTRAINT_CKPT_KEY)
            or ckpt_dict.get(_DGPO_CONSTRAINT_CKPT_KEY_LEGACY)
        )
        if ckpt_dict is not None
        else None
    )
    constraint_state: ProjectionConstraintState | None = None
    if proj_cfg_startup.active:
        _validate_dgpo_constraint_resume(constraint_ckpt_blob, expected_type="latent_swd")
        latent_cfg = proj_cfg_startup.latent_swd
        if latent_cfg is None or not latent_cfg.checkpoint_file:
            raise ValueError(
                "dgpo.projection_constraint.latent_swd.checkpoint_file is required "
                "(frozen encoder checkpoint)."
            )
        ckpt_file = Path(latent_cfg.checkpoint_file).expanduser()
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"latent_swd.checkpoint_file not found: {ckpt_file} "
                "(frozen latent-constraint encoder)"
            )
        policy_norm = Path(
            str(global_config.options.Dataset.normalization_file)
        ).expanduser()
        latent_norm_raw = latent_cfg.normalization_file.strip()
        if latent_norm_raw:
            latent_norm = Path(latent_norm_raw).expanduser()
            if policy_norm.resolve() != latent_norm.resolve() and is_rank0:
                _log.warning(
                    "[DGPO] latent_swd.normalization_file (%s) differs from policy "
                    "Dataset.normalization_file (%s); latent/policy neutrino spaces may "
                    "diverge.",
                    latent_norm,
                    policy_norm,
                )
        constraint_state = init_latent_swd_state(
            latent_cfg,
            device=device,
            resume_payload=constraint_ckpt_blob,
        )
        broadcast_latent_swd_state(
            constraint_state,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if is_rank0:
            enc = constraint_state.model
            policy_norm_res = policy_norm.resolve()
            latent_norm_cfg = latent_cfg.normalization_file.strip()
            _log.info(
                "[DGPO] DGPO + CPO + latent-SWD (frozen): checkpoint=%s margin=%.4g "
                "num_projections=%s apply_to=%s world_size=%s latent_dim=%s d_model=%s "
                "encoder_params=%.3fM policy_norm=%s latent_swd_norm=%s. "
                "Encoder broadcast from rank 0; no on-policy retrain/finetune.",
                str(ckpt_file),
                float(latent_cfg.margin),
                int(latent_cfg.num_projections),
                latent_cfg.apply_to,
                world_size,
                enc.latent_dim,
                enc.d_model,
                sum(p.numel() for p in enc.parameters()) / 1e6,
                policy_norm_res,
                latent_norm_cfg or "(from checkpoint payload)",
            )
            if constraint_ckpt_blob is not None:
                _log.info(
                    "[DGPO] latent-SWD resume metadata from DGPO checkpoint: %s",
                    {
                        k: constraint_ckpt_blob.get(k)
                        for k in ("constraint_type", "checkpoint_file", "normalization_file")
                    },
                )
    elif is_rank0:
        _log.info("[DGPO] projection_constraint.type=none -> pure DGPO (no CPO / latent-SWD repair).")
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    save_dir_raw = global_config.options.Training.get("model_checkpoint_save_path", None)
    top_k_ckpt = int(global_config.options.Training.get("model_checkpoint_save_top_k", 5))
    ckpt_topk: _DgpoCheckpointTopK | None = None
    if save_dir_raw and is_rank0:
        ckpt_topk = _DgpoCheckpointTopK(
            Path(str(save_dir_raw)).expanduser().resolve(),
            top_k_ckpt,
        )

    epochs = int(global_config.options.Training.epochs)

    if start_epoch > 0 or global_step > 0:
        if is_rank0:
            _log.info(
                "[DGPO] Resuming: start_epoch=%s global_step=%s (total epochs in config=%s).",
                start_epoch,
                global_step,
                epochs,
            )
    if start_epoch >= epochs:
        if is_rank0:
            _log.info(
                "[DGPO] start_epoch=%s >= epochs=%s; nothing to train. Check config or checkpoint.",
                start_epoch,
                epochs,
            )
        _finish_wandb_run(wandb_active)
        return

    if is_rank0:
        _log.info(
            "[DGPO] rank=%s/%s device=%s train_events≈%s val_events≈%s batch=%s train_K=%s val_K=%s "
            "val_every_n_epochs=%s ddim=%s train_timesteps=%s steps/epoch≈%s epochs=%s "
            "(z-score advantages, shared noise, accumulated substeps, rollout EMA on)",
            rank,
            world_size,
            device,
            total_events,
            val_events if val_events is not None else 0,
            batch_size,
            K,
            val_K,
            validation_every_n_epochs,
            num_ddim,
            num_train_timesteps,
            steps_per_epoch,
            epochs,
        )

    wandb_mod = None
    if wandb_active:
        import wandb as wandb_mod

    val_baseline_state: dict[str, np.ndarray] | None = None
    val_profile_history: dict[str, dict[str, list[float]]] = {
        name: {"epoch": [], "delta_mean": [], "slope": [], "zero": []}
        for name in _validation_profile_feature_names(cartesian=_truth_generation_cartesian())
    }

    def _append_validation_history_plots(
        val_metrics: dict[str, Any],
        *,
        epoch_value: int,
    ) -> None:
        if not is_rank0:
            return
        for profile_name, hist in val_profile_history.items():
            delta_mean = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/delta_mean",
                    float("nan"),
                )
            )
            slope = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/slope",
                    float("nan"),
                )
            )
            zero = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/zero_delta_truth",
                    float("nan"),
                )
            )
            if not (math.isfinite(slope) and math.isfinite(zero)):
                continue
            hist["epoch"].append(float(epoch_value))
            hist["delta_mean"].append(delta_mean)
            hist["slope"].append(slope)
            hist["zero"].append(zero)
            if wandb_mod is None:
                continue
            _x_label, y_label, display = _profile_axis_labels(profile_name)
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_delta_mean_vs_epoch"
            ] = _validation_epoch_history_figure(
                hist["epoch"],
                hist["delta_mean"],
                ylabel=y_label.replace("Mean ", "Global mean "),
                title=f"Validation {display} residual mean over epochs",
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_slope_vs_epoch"
            ] = _validation_slope_history_figure(
                hist["epoch"],
                hist["slope"],
                profile_name=profile_name,
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_zero_delta_truth_vs_epoch"
            ] = _validation_epoch_history_figure(
                hist["epoch"],
                hist["zero"],
                ylabel=f"{_x_label} where mean residual = 0",
                title=f"Validation {display} zero-crossing over epochs",
                zero_line=False,
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_zero_delta_vs_slope"
            ] = _validation_zero_vs_slope_figure(
                hist["slope"],
                hist["zero"],
                hist["epoch"],
                profile_name=profile_name,
            )

    profile_accum_suffixes = (
        "truth_all",
        "delta_all",
        "truth_best",
        "delta_best",
        "truth_oracle",
        "delta_oracle",
    )
    profile_accum: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            _diag_profile_raw_key(name, suffix): []
            for suffix in profile_accum_suffixes
        }
        for name in _DIAG_PROFILE_NAMES
    }
    profile_accum_batches: dict[str, int] = {name: 0 for name in _DIAG_PROFILE_NAMES}

    def _append_profile_accum(metrics: dict[str, Any]) -> None:
        if wandb_mod is None:
            return
        for profile_name, key_lists in profile_accum.items():
            have_all = all(
                isinstance(metrics.get(k), np.ndarray) and metrics[k].size > 0
                for k in key_lists
            )
            if not have_all:
                continue
            for k in key_lists:
                key_lists[k].append(metrics[k])
            profile_accum_batches[profile_name] += 1

    def _flush_profile_accum(*, step: int, force: bool = False) -> None:
        nonlocal profile_accum, profile_accum_batches
        if wandb_mod is None:
            return
        if not force and int(step) % int(diagnostic_plot_every) != 0:
            return
        payload: dict[str, Any] = {}
        flushed_names: list[str] = []
        for profile_name, key_lists in profile_accum.items():
            batches = profile_accum_batches[profile_name]
            if batches <= 0:
                continue
            if not force and batches < diagnostic_profile_accumulate_steps:
                continue
            if f"{profile_name}_profile_accumulated" not in diagnostic_plot_names:
                continue
            merged = {
                k: np.concatenate(v, axis=0) if v else np.array([], dtype=np.float64)
                for k, v in key_lists.items()
            }
            payload[_diag_profile_log_key(profile_name, accumulated=True)] = (
                _delta_selection_profiles_figure(
                    merged[_diag_profile_raw_key(profile_name, "truth_all")],
                    merged[_diag_profile_raw_key(profile_name, "delta_all")],
                    merged[_diag_profile_raw_key(profile_name, "truth_best")],
                    merged[_diag_profile_raw_key(profile_name, "delta_best")],
                    merged[_diag_profile_raw_key(profile_name, "truth_oracle")],
                    merged[_diag_profile_raw_key(profile_name, "delta_oracle")],
                    profile_name=profile_name,
                    title=_diag_profile_title(
                        profile_name,
                        accumulated_batches=batches,
                    )
                )
            )
            flushed_names.append(profile_name)
        if not payload:
            return
        try:
            _wandb_log_step(wandb_mod, payload, step=step)
        finally:
            for profile_name in flushed_names:
                profile_accum[profile_name] = {
                    _diag_profile_raw_key(profile_name, suffix): []
                    for suffix in profile_accum_suffixes
                }
                profile_accum_batches[profile_name] = 0

    def _barrier() -> None:
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

    # Following the EveNet ``train.py`` pattern: no rank-0-only synchronous setup
    # before the training loop.  All ranks proceed straight into ``fit``-style
    # iteration and hit the data pipeline simultaneously, avoiding NCCL barriers
    # that would otherwise busy-wait the GPU while rank 0 does cold-start work.
    ve_initial = int(val_events) if val_events is not None else 0
    if start_epoch == 0 and ve_initial > 0 and val_shard is not None:
        if is_rank0:
            _log.info(
                "[DGPO] val: running pre-DGPO baseline validation (epoch=-1) for response diagnostics."
            )
        val_loader = val_shard.iter_torch_batches(**val_loader_cfg)
        est_val_batches = (
            max(1, math.ceil(ve_initial / effective_batch)) if ve_initial > 0 else None
        )
        initial_val_metrics = run_validation_epoch(
            model,
            ref_model,
            ema_save,
            val_loader,
            sampler,
            reward_agg,
            val_K=val_K,
            num_ddim_steps=num_ddim_val,
            device=device,
            dtype=dtype,
            cartesian=_truth_generation_cartesian(),
            compute_winrate=bool(dg.get("validation_compute_winrate", False)),
            epoch=-1,
            est_total_batches=est_val_batches,
            val_log_batches=bool(dg.get("validation_log_batches", True)),
            val_rollout_parallel_chains=val_rollout_parallel_chains,
            val_tqdm_k_chains=bool(dg.get("validation_tqdm_k_chains", True)),
            val_tqdm_ddim=bool(dg.get("validation_tqdm_ddim", False)),
            max_batches=val_max_batches,
            initial_state=None,
            rank=rank,
            world_size=world_size,
        )
        maybe_initial_state = initial_val_metrics.get("_val_initial_state")
        if isinstance(maybe_initial_state, dict):
            val_baseline_state = maybe_initial_state

        if is_rank0:
            _append_validation_history_plots(initial_val_metrics, epoch_value=-1)
            profile_summary = " ".join(
                (
                    f"{profile_name}_slope="
                    f"{initial_val_metrics.get(f'val_diagnostics/profile/{profile_name}/slope', float('nan')):.6g} "
                    f"{profile_name}_zero="
                    f"{initial_val_metrics.get(f'val_diagnostics/profile/{profile_name}/zero_delta_truth', float('nan')):.6g}"
                )
                for profile_name in val_profile_history.keys()
            )
            _log.info(
                "[DGPO] initial val r_mean=%.6f %s",
                initial_val_metrics["val/reward/mean"],
                profile_summary.strip(),
            )
            if wandb_mod is not None:
                _wandb_log_validation(
                    wandb_mod,
                    initial_val_metrics,
                    epoch=-1,
                    wandb_step=0,
                )
        _barrier()
    elif start_epoch > 0 and ve_initial > 0 and is_rank0:
        _log.warning(
            "[DGPO] Response matrices need the pre-DGPO validation baseline; "
            "this run is resuming at start_epoch=%s, so val/response/* will be skipped.",
            start_epoch,
        )

    def constraint_ckpt_payload_for_save() -> dict[str, Any] | None:
        return _dgpo_constraint_checkpoint_payload(constraint_state)

    try:
        legacy_train_kinematics = _supports_legacy_invisible_kinematics(
            cartesian=_truth_generation_cartesian(),
            feature_dim=len(_invisible_feature_names()) or None,
        )

        for epoch in range(start_epoch, epochs):
            # Each call to ``iter_torch_batches`` produces a fresh streaming generator
            # over this rank's shard.  ``local_shuffle_buffer_size`` (set in train_loader_cfg)
            # provides per-shard random shuffling each epoch.
            train_iter = train_shard.iter_torch_batches(**train_loader_cfg)
            train_it = iter(train_iter)
            num_diag_bins = _resolve_diagnostic_num_bins()

            # Epoch-level histogram accumulators for training-distribution plots (all batches).
            td_pt_p = np.zeros(num_diag_bins, dtype=np.float64)
            td_pt_t = np.zeros(num_diag_bins, dtype=np.float64)
            td_e_p = np.zeros(num_diag_bins, dtype=np.float64)
            td_e_t = np.zeros(num_diag_bins, dtype=np.float64)
            td_p_p = np.zeros(num_diag_bins, dtype=np.float64)
            td_p_t = np.zeros(num_diag_bins, dtype=np.float64)
            td_k1_pt_p = np.zeros(num_diag_bins, dtype=np.float64)
            td_k1_pt_t = np.zeros(num_diag_bins, dtype=np.float64)
            td_k1_e_p = np.zeros(num_diag_bins, dtype=np.float64)
            td_k1_e_t = np.zeros(num_diag_bins, dtype=np.float64)
            td_k1_p_p = np.zeros(num_diag_bins, dtype=np.float64)
            td_k1_p_t = np.zeros(num_diag_bins, dtype=np.float64)
            td_all_feature_names = _generation_monitor_feature_names(
                cartesian=_truth_generation_cartesian()
            )
            td_all_chunks: dict[str, list[np.ndarray]] = {
                f"{feature_name}_{suffix}": []
                for feature_name in td_all_feature_names
                for suffix in ("truth", "pred")
            }
            stop_epoch = False
            while True:
                if max_steps is not None and global_step >= max_steps:
                    last_done = epoch - 1 if epoch > 0 else 0
                    if is_rank0:
                        _dgpo_save_last_ckpt(
                            model,
                            ema_save,
                            optimizer,
                            ref_model,
                            last_completed_epoch=last_done,
                            dgpo_next_epoch=epoch,
                            global_step=global_step,
                            ema_rollout=ema_rollout,
                            dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                        )
                        _log.info("[DGPO] max_steps=%s reached; stopping.", max_steps)
                    _barrier()
                    return

                batch_cpu, has_more = _next_batch_synced(
                    train_it, world_size=world_size, device=device
                )
                if not has_more or batch_cpu is None:
                    stop_epoch = True
                    break

                batch_d = batch_to_device(batch_cpu, device)
                reward_dist_step = wandb_active and (
                    global_step % log_reward_dist_every == 0
                )
                diagnostic_dist_step = wandb_active
                metrics = train_step(
                    model,
                    ref_model,
                    ema_rollout,
                    ema_save,
                    batch_d,
                    optimizer,
                    sampler,
                    reward_agg,
                    beta=beta,
                    K=K,
                    num_ddim_steps=num_ddim,
                    rollout_parallel_chains=rollout_parallel_chains,
                    global_step=global_step,
                    epoch=epoch,
                    device=device,
                    dtype=dtype,
                    log_reward_dist=reward_dist_step,
                    log_diagnostic_dist=diagnostic_dist_step,
                    diagnostic_plot_names=diagnostic_plot_names,
                    diagnostic_plot_every=diagnostic_plot_every,
                    num_train_timesteps=num_train_timesteps,
                    adv_clip_max=adv_clip_max_cfg,
                    grad_clip_norm=grad_clip_norm_cfg,
                    policy_eval_t_min=policy_eval_t_min_cfg,
                    policy_eval_t_max=policy_eval_t_max_cfg,
                    constraint_state=constraint_state,
                    world_size=world_size,
                )
                if wandb_mod is not None:
                    payload = _wandb_train_payload(metrics)
                    payload["epoch"] = float(epoch)
                    payload["global_step"] = float(global_step)
                    _wandb_log_step(wandb_mod, payload, step=global_step)
                    _append_profile_accum(metrics)
                    _flush_profile_accum(step=global_step)

                if legacy_train_kinematics:
                    td_pt_p += metrics["_kin_h_pt_p"]
                    td_pt_t += metrics["_kin_h_pt_t"]
                    td_e_p += metrics["_kin_h_e_p"]
                    td_e_t += metrics["_kin_h_e_t"]
                    td_p_p += metrics["_kin_h_p_p"]
                    td_p_t += metrics["_kin_h_p_t"]
                    td_k1_pt_p += metrics["_kin_h_pt_k1_p"]
                    td_k1_pt_t += metrics["_kin_h_pt_k1_t"]
                    td_k1_e_p += metrics["_kin_h_e_k1_p"]
                    td_k1_e_t += metrics["_kin_h_e_k1_t"]
                    td_k1_p_p += metrics["_kin_h_p_k1_p"]
                    td_k1_p_t += metrics["_kin_h_p_k1_t"]
                for feature_name in td_all_feature_names:
                    truth_key = f"_kin_all_{feature_name}_t"
                    pred_key = f"_kin_all_{feature_name}_p"
                    if truth_key not in metrics or pred_key not in metrics:
                        continue
                    td_all_chunks[f"{feature_name}_truth"].append(metrics[truth_key])
                    td_all_chunks[f"{feature_name}_pred"].append(metrics[pred_key])

                if is_rank0 and global_step % log_every == 0:
                    _log.info(
                        "epoch=%s step=%s L_total=%.6f L_dgpo=%.6f "
                        "L_cur=%.4f L_ref=%.4f delta=%.4f "
                        "r_best=%.4f r_med=%.4f gap=%.4f",
                        epoch,
                        global_step,
                        metrics["train/loss/total"],
                        metrics["train/loss/dgpo"],
                        metrics["train/loss/L_cur"],
                        metrics["train/loss/L_ref"],
                        metrics["train/loss/delta"],
                        metrics["reward/monitor/best_of_k"],
                        metrics["reward/monitor/median"],
                        metrics["reward/monitor/mean_gap"],
                    )
                global_step += 1

            # --- Epoch-end: build training-distribution figures from accumulated histograms ---
            if wandb_mod is not None:
                _flush_profile_accum(step=max(global_step - 1, 0), force=True)

            if legacy_train_kinematics and world_size > 1:
                td_stack = np.stack([
                    td_pt_p, td_pt_t, td_e_p, td_e_t, td_p_p, td_p_t,
                    td_k1_pt_p, td_k1_pt_t, td_k1_e_p, td_k1_e_t, td_k1_p_p, td_k1_p_t,
                ])
                td_hist_t = torch.from_numpy(td_stack).to(device=device, dtype=torch.float64)
                dist.all_reduce(td_hist_t, op=dist.ReduceOp.SUM)
                td_merged = td_hist_t.cpu().numpy()
                (
                    td_pt_p, td_pt_t, td_e_p, td_e_t, td_p_p, td_p_t,
                    td_k1_pt_p, td_k1_pt_t, td_k1_e_p, td_k1_e_t, td_k1_p_p, td_k1_p_t,
                ) = [td_merged[i] for i in range(12)]

            td_all_merged: dict[str, np.ndarray] = {}
            td_all_merged = _gather_val_array_dict(
                {
                    key: _concat_np_chunks(chunks)
                    for key, chunks in td_all_chunks.items()
                },
                rank=rank,
                world_size=world_size,
            )

            if is_rank0 and wandb_mod is not None:
                _td_bin_pt = _diagnostic_bin_edges("pt")
                _td_bin_eta = _diagnostic_bin_edges("eta")
                _td_bin_phi = _diagnostic_bin_edges("phi")
                try:
                    td_log: dict[str, Any] = {"epoch": float(epoch)}
                    if legacy_train_kinematics:
                        _td_suffix = "train: reward best-of-K vs truth (all batches)"
                        td_log.update({
                            "train_dist/pt": _val_overlay_kin_figure(
                                td_pt_t, td_pt_p, _td_bin_pt,
                                f"Neutrino pT [GeV] ({_td_suffix})",
                                pred_label="Pred (train)", xlabel="pT [GeV]",
                            ),
                            "train_dist/eta": _val_overlay_kin_figure(
                                td_e_t, td_e_p, _td_bin_eta,
                                f"Neutrino η ({_td_suffix})",
                                pred_label="Pred (train)", xlabel="η",
                            ),
                            "train_dist/phi": _val_overlay_kin_figure(
                                td_p_t, td_p_p, _td_bin_phi,
                                f"Neutrino φ ({_td_suffix})",
                                pred_label="Pred (train)", xlabel="φ [rad]",
                            ),
                            "train_dist_k1/pt": _val_overlay_kin_figure(
                                td_k1_pt_t, td_k1_pt_p, _td_bin_pt,
                                "Neutrino pT [GeV] (train: candidate 0 / K=1 proxy vs truth, all batches)",
                                pred_label="Pred (train K=1 proxy)", xlabel="pT [GeV]",
                            ),
                            "train_dist_k1/eta": _val_overlay_kin_figure(
                                td_k1_e_t, td_k1_e_p, _td_bin_eta,
                                "Neutrino η (train: candidate 0 / K=1 proxy vs truth, all batches)",
                                pred_label="Pred (train K=1 proxy)", xlabel="η",
                            ),
                            "train_dist_k1/phi": _val_overlay_kin_figure(
                                td_k1_p_t, td_k1_p_p, _td_bin_phi,
                                "Neutrino φ (train: candidate 0 / K=1 proxy vs truth, all batches)",
                                pred_label="Pred (train K=1 proxy)", xlabel="φ [rad]",
                            ),
                        })
                    available_td_jsd_features = _available_truth_pred_features(
                        td_all_merged,
                        td_all_feature_names,
                    )
                    for feature_name in available_td_jsd_features:
                        bin_edges = _generation_special_bin_edges(feature_name)
                        td_log[f"train_dist/jsd/current/{feature_name}"] = _array_histogram_jsd(
                            td_all_merged.get(
                                f"{feature_name}_truth",
                                np.array([], dtype=np.float64),
                            ),
                            td_all_merged.get(
                                f"{feature_name}_pred",
                                np.array([], dtype=np.float64),
                            ),
                            bin_edges=bin_edges,
                        )
                    for feature_name in _available_truth_pred_features(
                        td_all_merged,
                        td_all_feature_names,
                    ):
                        td_log[f"train_dist/all/{feature_name}_truth_vs_pred"] = _truth_pred_matrix_figure(
                            td_all_merged.get(f"{feature_name}_truth", np.array([], dtype=np.float64)),
                            td_all_merged.get(f"{feature_name}_pred", np.array([], dtype=np.float64)),
                            xlabel=f"Truth {feature_name}",
                            ylabel=f"Pred {feature_name}",
                            title=f"Train 2D truth vs pred {feature_name} (all candidates, all batches)",
                            bin_edges=_generation_special_bin_edges(feature_name),
                        )
                        for metric_name, metric_value in _truth_pred_scalar_metrics(
                            td_all_merged.get(f"{feature_name}_truth", np.array([], dtype=np.float64)),
                            td_all_merged.get(f"{feature_name}_pred", np.array([], dtype=np.float64)),
                        ).items():
                            td_log[f"train_dist/all_metrics/{feature_name}/{metric_name}"] = metric_value
                    if len(td_log) > 1:
                        _wandb_log_with_step(
                            wandb_mod,
                            td_log,
                            step=_wandb_epoch_end_step(global_step),
                        )
                except Exception as _e:
                    _log.warning("[DGPO] train_dist figures failed at epoch=%s: %s", epoch, _e)

            ve = int(val_events) if val_events is not None else 0
            run_epoch_val = ve > 0 and val_shard is not None and _dgpo_should_run_validation_epoch(
                epoch, validation_every_n_epochs
            )
            if not run_epoch_val and ve > 0 and val_shard is not None and is_rank0:
                if (epoch + 1) % validation_every_n_epochs != 0:
                    _log.info(
                        "[DGPO] val: skipped epoch=%s (validation_every_n_epochs=%s; "
                        "next val at epoch=%s).",
                        epoch,
                        validation_every_n_epochs,
                        epoch + (validation_every_n_epochs - (epoch + 1) % validation_every_n_epochs),
                    )
            if run_epoch_val:
                if is_rank0:
                    _log.info(
                        "[DGPO] val: requesting val iterator (Ray read+preprocess may take a long time; "
                        "first DDIM batch logs after rows arrive).",
                    )
                val_loader = val_shard.iter_torch_batches(**val_loader_cfg)
                est_val_batches = max(1, math.ceil(ve / effective_batch)) if ve > 0 else None
                val_metrics = run_validation_epoch(
                    model,
                    ref_model,
                    ema_save,
                    val_loader,
                    sampler,
                    reward_agg,
                    val_K=val_K,
                    num_ddim_steps=num_ddim_val,
                    device=device,
                    dtype=dtype,
                    cartesian=_truth_generation_cartesian(),
                    compute_winrate=bool(dg.get("validation_compute_winrate", False)),
                    epoch=epoch,
                    est_total_batches=est_val_batches,
                    val_log_batches=bool(dg.get("validation_log_batches", True)),
                    val_rollout_parallel_chains=val_rollout_parallel_chains,
                    val_tqdm_k_chains=bool(dg.get("validation_tqdm_k_chains", True)),
                    val_tqdm_ddim=bool(dg.get("validation_tqdm_ddim", False)),
                    max_batches=val_max_batches,
                    initial_state=val_baseline_state,
                    rank=rank,
                    world_size=world_size,
                )
                if is_rank0:
                    _append_validation_history_plots(val_metrics, epoch_value=epoch)
                    _log.info(
                        "[DGPO] val epoch=%s r_mean=%.6f r_med=%.6f p10=%.6f p90=%.6f winrate=%.4f",
                        epoch,
                        val_metrics["val/reward/mean"],
                        val_metrics["val/reward/median"],
                        val_metrics["val/reward/p10"],
                        val_metrics["val/reward/p90"],
                        val_metrics["val/winrate"],
                    )
                    if wandb_mod is not None:
                        _wandb_log_validation(
                            wandb_mod,
                            val_metrics,
                            epoch=epoch,
                            wandb_step=_wandb_epoch_end_step(global_step),
                        )
                    if ckpt_topk is not None:
                        ckpt_topk.maybe_save(
                            val_reward_mean=val_metrics["val/reward/mean"],
                            last_completed_epoch=epoch,
                            dgpo_next_epoch=epoch + 1,
                            global_step=global_step,
                            model=model,
                            ema_save=ema_save,
                            optimizer=optimizer,
                            ref_model=ref_model,
                            ema_rollout=ema_rollout,
                            dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                        )

            _barrier()

            if is_rank0:
                _dgpo_save_last_ckpt(
                    model,
                    ema_save,
                    optimizer,
                    ref_model,
                    last_completed_epoch=epoch,
                    dgpo_next_epoch=epoch + 1,
                    global_step=global_step,
                    ema_rollout=ema_rollout,
                    dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                )
            _barrier()

            # ``stop_epoch`` is set when this rank's shard ran dry; nothing else to do.
            del stop_epoch

        if is_rank0:
            _log.info("[DGPO] finished %s epochs (%s optimizer steps).", epochs, global_step)
    finally:
        _finish_wandb_run(wandb_active)


def main() -> None:
    """CLI entry point: build a Ray ``TorchTrainer`` and ``fit()`` across the cluster."""
    p = argparse.ArgumentParser(description="DGPO neutrino RL (Ray Train + DGPO loop)")
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "config.yaml",
        help="YAML config (same merge rules as EveNet training)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (smoke test)",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides config)",
    )
    p.add_argument(
        "--ray-dir",
        type=str,
        default="~/ray_results",
        help="Ray Train RunConfig.storage_path",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    global_config.load_yaml(config_path)
    _assert_rl_enabled()
    global_config.display()
    platform_info = global_config.platform
    # Default 0: let Ray set per-worker CUDA_VISIBLE_DEVICES. With ``1`` (full node
    # visibility), Ray Train + ``ScalingConfig(GPU: 1)`` can call ``torch.cuda.set_device``
    # with local_rank 3 while Slurm only exposes fewer GPUs on that node →
    # ``DeferredCudaCallError: device=3, num_gpus=...``. Override with ``export
    # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` only if you need the old
    # Shifter/NCCL workaround from ``NERSC/start_interactive_ray.sh``.
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "0")

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{_REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
            "TORCH_NCCL_TIMEOUT": "180",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": os.environ[
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ],
        },
    }
    if "WANDB_API_KEY" in os.environ:
        runtime_env["env_vars"]["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # ``address="auto"`` forces a connection to the Ray cluster already started by
    # ``NERSC/start-head.sh`` / ``start-worker.sh`` instead of silently spinning up a
    # fresh single-node cluster on the head.  ``RAY_ADDRESS`` (set by the sbatch helper)
    # takes precedence when present.  Outside Slurm we fall back to a local cluster.
    ray_addr_env = os.environ.get("RAY_ADDRESS")
    try:
        ray.init(
            address=ray_addr_env or "auto",
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    except (ConnectionError, ValueError) as ex:
        _log.warning(
            "[DGPO][launch] No existing Ray cluster (%s); falling back to local init.",
            ex,
        )
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    # Wait for the expected number of Ray workers to join.  Worker srun in the sbatch
    # script typically needs 30-90s on NERSC; without this wait, ``trainer.fit()`` may
    # see only the head node and silently run on 1 node.
    expected_workers = int(platform_info.number_of_workers)
    expected_gpus_per_worker = float(dict(platform_info.resources_per_worker).get("GPU", 1))
    expected_gpus = float(expected_workers) * expected_gpus_per_worker
    wait_timeout_s = float(os.environ.get("DGPO_RAY_WAIT_S", "300"))
    poll_every = 5.0
    waited = 0.0
    while waited < wait_timeout_s:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        cur_nodes = len(ray.nodes())
        if cur_gpus >= expected_gpus:
            _log.info(
                "[DGPO][launch] Ray cluster ready: nodes=%s GPUs=%s (expected %s).",
                cur_nodes, cur_gpus, expected_gpus,
            )
            break
        _log.info(
            "[DGPO][launch] waiting for Ray workers... nodes=%s GPUs=%s/%s (%.0fs/%.0fs)",
            cur_nodes, cur_gpus, expected_gpus, waited, wait_timeout_s,
        )
        time.sleep(poll_every)
        waited += poll_every
    else:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        _log.warning(
            "[DGPO][launch] Timed out after %.0fs waiting for cluster: GPUs=%s (expected %s). "
            "Continuing — Ray Train may run with fewer workers or hang.",
            wait_timeout_s, cur_gpus, expected_gpus,
        )

    base_dir = Path(platform_info.data_parquet_dir)
    base_val_dir = (
        Path(platform_info.data_parquet_val_dir)
        if "data_parquet_val_dir" in platform_info
        else None
    )
    process_fn = make_process_fn(base_dir)
    train_ds, val_ds, total_events, val_events = prepare_datasets(
        base_dir=base_dir,
        process_event_batch_partial=process_fn,
        platform_info=platform_info,
        load_all_in_ram=False,
        base_val_dir=base_val_dir,
        predict=False,
    )

    datasets: dict[str, Any] = {"train": train_ds}
    if val_ds is not None and val_events:
        datasets["validation"] = val_ds

    scaling_config = ScalingConfig(
        num_workers=int(platform_info.number_of_workers),
        resources_per_worker=dict(platform_info.resources_per_worker),
        use_gpu=bool(platform_info.get("use_gpu", True)),
    )
    run_config = RunConfig(
        name="DGPO-Training",
        storage_path=args.ray_dir,
    )

    # Driver-side launch banner: visible on the head node before any worker spawns,
    # so a wrong cluster size is caught before the data pipeline starts.
    try:
        cluster_resources = ray.cluster_resources()
    except Exception:
        cluster_resources = {}
    _log.info(
        "[DGPO][launch] num_workers=%s resources_per_worker=%s use_gpu=%s "
        "cluster_GPUs=%s cluster_CPUs=%s nodes=%s",
        scaling_config.num_workers,
        scaling_config.resources_per_worker,
        scaling_config.use_gpu,
        cluster_resources.get("GPU"),
        cluster_resources.get("CPU"),
        len(ray.nodes()) if ray.is_initialized() else "?",
    )
    trainer_config = {
        "config_path": str(config_path),
        "config_yaml": config_path.read_text(),
        "max_steps": args.max_steps,
        "wandb": not args.no_wandb,
        "total_events": int(total_events),
        "val_events": int(val_events) if val_events else 0,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=dgpo_train_loop,
        train_loop_config=trainer_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets=datasets,
    )
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
