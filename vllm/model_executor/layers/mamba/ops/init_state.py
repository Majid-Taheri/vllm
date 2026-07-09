# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused initial-state gather for Mamba2 SSD prefill.

Replaces ``torch.where(mask[:, None, None, None], ssm_state[idx], 0)`` —
which materialises a zero tensor of the broadcast shape via FillFunctor
before the where — with a single Triton kernel that writes either the
gathered state or zero into the output, no intermediate fill.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mamba_init_state_fwd_kernel(
    out_ptr,                # [batch, nheads, headdim, dstate]
    state_ptr,              # [num_blocks, nheads, headdim, dstate]
    mask_ptr,               # [batch] (bool)
    indices_ptr,            # [batch] (int64)
    nheads,
    stride_out_b: tl.int64,
    stride_out_h: tl.int64,
    stride_out_p: tl.int64,
    stride_state_blk: tl.int64,
    stride_state_h: tl.int64,
    stride_state_p: tl.int64,
    INNER: tl.constexpr,    # headdim * dstate (contiguous innermost run)
    BLOCK: tl.constexpr,
):
    """One program per (batch_row, head, inner-tile)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    has_state = tl.load(mask_ptr + pid_b).to(tl.int1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    inner_mask = offs < INNER

    out_off = (
        pid_b * stride_out_b
        + pid_h * stride_out_h
        + offs * stride_out_p
    )

    if has_state:
        blk = tl.load(indices_ptr + pid_b)
        src_off = (
            blk.to(tl.int64) * stride_state_blk
            + pid_h * stride_state_h
            + offs * stride_state_p
        )
        vals = tl.load(state_ptr + src_off, mask=inner_mask, other=0.0)
        tl.store(out_ptr + out_off, vals, mask=inner_mask)
    else:
        zeros = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + out_off, zeros, mask=inner_mask)


def mamba_gather_initial_states(
    ssm_state: torch.Tensor,   # [num_blocks, nheads, headdim, dstate]
    indices: torch.Tensor,     # [batch] int64
    has_initial_states: torch.Tensor,  # [batch] bool
) -> torch.Tensor:
    """Fused replacement for::

        torch.where(
            has_initial_states[:, None, None, None],
            ssm_state[indices],
            0,
        )

    Returns a fresh ``[batch, nheads, headdim, dstate]`` tensor matching
    ``ssm_state.dtype``. One kernel launch, no zero-tensor materialisation.
    """
    assert ssm_state.dim() == 4, "ssm_state must be [num_blocks, nh, hd, ds]"
    assert indices.dim() == 1
    assert has_initial_states.dim() == 1
    assert indices.shape[0] == has_initial_states.shape[0]

    batch = indices.shape[0]
    _, nheads, headdim, dstate = ssm_state.shape

    # Innermost (headdim, dstate) is contiguous in both the source slab and
    # the destination — collapse them into a single linear dim of length
    # headdim * dstate so we can launch one tiled inner axis. We don't
    # require contiguity, just that strides agree on the inner axis.
    inner = headdim * dstate

    out = torch.empty(
        (batch, nheads, headdim, dstate),
        dtype=ssm_state.dtype,
        device=ssm_state.device,
    )

    # Strides in elements (Triton pointer arithmetic is in elements).
    stride_out_b = out.stride(0)
    stride_out_h = out.stride(1)
    stride_out_p = out.stride(3)  # innermost stride (== 1 for contiguous)

    stride_state_blk = ssm_state.stride(0)
    stride_state_h = ssm_state.stride(1)
    stride_state_p = ssm_state.stride(3)

    # Pick a BLOCK that divides 'inner' efficiently. Most Nemotron variants
    # use headdim=64, dstate=128 → inner=8192; smaller heads still hit good
    # occupancy with BLOCK=256.
    if inner >= 1024:
        BLOCK = 1024
    elif inner >= 256:
        BLOCK = 256
    else:
        BLOCK = max(triton.next_power_of_2(inner), 32)
    n_inner_tiles = triton.cdiv(inner, BLOCK)

    grid = (batch, nheads, n_inner_tiles)
    _mamba_init_state_fwd_kernel[grid](
        out,
        ssm_state,
        has_initial_states.to(torch.bool, copy=False).contiguous(),
        indices.to(torch.int64, copy=False).contiguous(),
        nheads,
        stride_out_b,
        stride_out_h,
        stride_out_p,
        stride_state_blk,
        stride_state_h,
        stride_state_p,
        INNER=inner,
        BLOCK=BLOCK,
    )
    return out
