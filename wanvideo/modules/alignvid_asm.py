# wanvideo/modules/alignvid_asm.py

import math
import torch
import torch.nn as nn
from typing import Optional, Dict


def sigmoid_rescale(x: torch.Tensor,
                    low: float = 1.0,
                    high: float = 2.0) -> torch.Tensor:
    """
    Monotonic mapping f(x) used for energy-based scaling (Eq. 10).
    Output in [low, high].
    """
    # Normalize x by a soft range; detach to avoid large gradients
    x_norm = torch.tanh(x / 5.0)
    s = torch.sigmoid(x_norm)
    return low + (high - low) * s


def compute_energy(Q: torch.Tensor,
                   K: torch.Tensor) -> torch.Tensor:
    """
    Compute average logit energy:
        E = (1 / (n_q * n_k)) * sum_{i,j} <Q_i, K_j> / sqrt(d_k)
    as in Eq. (10). [web:20]
    """
    n_q, d_k = Q.shape[-2], Q.shape[-1]
    n_k = K.shape[-2]
    scale = 1.0 / math.sqrt(d_k)

    # [n_q, n_k]
    logits = torch.matmul(Q, K.transpose(-1, -2)) * scale
    E = logits.mean()
    return E


class AlignVidScheduler:
    """
    Implements BGS (block-level) and SGS (step-level) scheduling
    for energy-based attention scaling (Algorithm 2, Eq. 11–16). [web:20]

    Usage:
      - Construct once per model.
      - Call `get_scale(l, t)` inside each attention forward.
    """

    def __init__(
        self,
        num_blocks: int,
        bgs_mode: str = "first_half",  # "first_half", "foreground", "all"
        bgs_mask: Optional[torch.Tensor] = None,
        gamma: float = 1.35,
        t_low: int = 0,
        t_high: int = 20,
        total_steps: int = 25,
        scale_queries: bool = False,
        scale_keys: bool = True,
        energy_low: float=1.0,
        energy_high: float=2.0,
    ):
        """
        Args:
            num_blocks: total number of DiT blocks in the Wan 2.1 model.
            bgs_mode: block gating:
              - "foreground": use bgs_mask (1 for foreground-sensitive blocks). [web:20]
              - "first_half": heuristic: first 50% of blocks gated on.
              - "all": apply modulation to all blocks.
            bgs_mask: optional Bool[ num_blocks ] from offline calibration.
            gamma: base scaling coefficient (>1) for ASM.
            t_low, t_high: SGS active interval [t_low, t_high] (0-based indexing).
            total_steps: total diffusion steps T.
            scale_queries / scale_keys: s_Q, s_K in Eq. (15) with s_Q + s_K = 1. [web:20]
            energy_low, energy_high: range for sigmoid rescaling.
        """
        self.num_blocks = num_blocks
        self.bgs_mode = bgs_mode
        self.gamma = gamma
        self.t_low = t_low
        self.t_high = t_high
        self.total_steps = total_steps
        self.scale_queries = bool(scale_queries)
        self.scale_keys = bool(scale_keys)
        assert self.scale_queries ^ self.scale_keys, "Exactly one of Q/K must be scaled."
        self.energy_low = energy_low
        self.energy_high = energy_high

        if bgs_mode == "foreground":
            assert bgs_mask is not None, "Foreground BGS requires a block mask."
            assert bgs_mask.shape[0] == num_blocks
            self.bgs_mask = bgs_mask.bool()
        elif bgs_mode == "first_half":
            mask = torch.zeros(num_blocks, dtype=torch.bool)
            mask[: max(1, num_blocks // 2)] = True
            self.bgs_mask = mask
        elif bgs_mode == "all":
            self.bgs_mask = torch.ones(num_blocks, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown bgs_mode: {bgs_mode}")

    def m_step(self, t: int) -> float:
        """
        SGS mask m(t) in Eq. (13): 1 if t in [t_low, t_high], else 0. [web:20]
        t is assumed 0-based.
        """
        return 1.0 if (t >= self.t_low and t <= self.t_high) else 0.0

    def b_block(self, l: int) -> float:
        """
        Block gate b^(l) in Eq. (14). [web:20]
        """
        return 1.0 if self.bgs_mask[l] else 0.0

    def get_scale(self, l: int, t: int) -> float:
        """
        Returns g^(l,t) = m(t) * b^(l) * (gamma - 1) (Eq. 14). [web:20]
        """
        m = self.m_step(t)
        b = self.b_block(l)
        return m * b * (self.gamma - 1.0)
