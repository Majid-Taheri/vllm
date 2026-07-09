# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the fused mamba initial-state gather kernel."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.init_state import (
    mamba_gather_initial_states,
)
from vllm.platforms import current_platform


def _reference(ssm_state, indices, has_initial_states):
    """The torch.where formulation we are replacing."""
    return torch.where(
        has_initial_states[:, None, None, None],
        ssm_state[indices],
        0,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="kernel is CUDA-only"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        # (num_blocks, batch, nheads, headdim, dstate)
        # Nemotron-H/Super uses headdim=64, dstate=128, nheads=128 (TP=1).
        (32, 1, 128, 64, 128),
        (32, 4, 128, 64, 128),
        (32, 8, 128, 64, 128),
        # A non-default-shape sanity check (smaller heads).
        (16, 3, 16, 32, 64),
    ],
)
def test_mamba_gather_initial_states_matches_torch_where(dtype, shape):
    num_blocks, batch, nheads, headdim, dstate = shape
    device = torch.device("cuda")
    torch.manual_seed(0)

    ssm_state = torch.randn(
        num_blocks, nheads, headdim, dstate, dtype=dtype, device=device
    )
    indices = torch.randint(0, num_blocks, (batch,), dtype=torch.int64, device=device)
    # Mix of True/False to exercise both branches in a single launch.
    has_initial_states = torch.tensor(
        [bool(i % 2) for i in range(batch)], dtype=torch.bool, device=device
    )

    ref = _reference(ssm_state, indices, has_initial_states)
    fused = mamba_gather_initial_states(ssm_state, indices, has_initial_states)

    assert ref.shape == fused.shape
    assert ref.dtype == fused.dtype
    # Bit-exact: both paths copy or write zero — no FP arithmetic.
    torch.testing.assert_close(fused, ref, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="kernel is CUDA-only"
)
def test_mamba_gather_initial_states_all_false():
    """All has_initial_states=False → fully zero output, ssm_state untouched."""
    device = torch.device("cuda")
    ssm_state = torch.randn(8, 4, 32, 64, dtype=torch.float16, device=device)
    indices = torch.zeros(3, dtype=torch.int64, device=device)
    has_initial_states = torch.zeros(3, dtype=torch.bool, device=device)

    out = mamba_gather_initial_states(ssm_state, indices, has_initial_states)
    assert torch.all(out == 0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="kernel is CUDA-only"
)
def test_mamba_gather_initial_states_all_true():
    """All has_initial_states=True → output equals gathered ssm_state."""
    device = torch.device("cuda")
    ssm_state = torch.randn(8, 4, 32, 64, dtype=torch.float16, device=device)
    indices = torch.tensor([1, 3, 5], dtype=torch.int64, device=device)
    has_initial_states = torch.ones(3, dtype=torch.bool, device=device)

    out = mamba_gather_initial_states(ssm_state, indices, has_initial_states)
    torch.testing.assert_close(out, ssm_state[indices], atol=0, rtol=0)
