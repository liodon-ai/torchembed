"""Triton kernels for accelerated embedding operations."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def fused_rope_bshd_apply(x, cos, sin):
    """Apply adjacent-pairs RoPE to a single tensor in (B, S, H, D) layout.

    Uses a stride-aware Triton kernel that reads and writes the tensor exactly
    once with no intermediate copies, regardless of the input's memory layout.

    This is the right entry-point for frameworks whose attention tensors are
    seq-major (e.g. torchtune), as opposed to ``fused_rope_forward`` which
    expects head-major ``(..., seq, dim)`` layout.

    Args:
        x: Input tensor of shape ``(B, S, H, D)``.
        cos: Cosine values of shape ``(S, D // 2)``.  May be non-contiguous
            (e.g. a stride-2 slice from a stacked cache).
        sin: Sine values of shape ``(S, D // 2)``.  Same constraints as ``cos``.

    Returns:
        Rotated tensor with the same shape and dtype as ``x``.

    Raises:
        ImportError: if triton is not installed.
    """
    if triton is None:
        raise ImportError("triton is required. Install it with: pip install triton")
    return _fused_rope_bshd_impl(x, cos, sin)


def fused_rope_forward(q, k, cos, sin):
    """Apply RoPE to Q and K using a fused triton kernel.

    Falls back to a clear ImportError if triton is not installed.

    Args:
        q: Query tensor of shape (..., seq_len, dim).
        k: Key tensor of shape (..., seq_len, dim).
        cos: Cosine cache of shape (seq_len, dim).
        sin: Sine cache of shape (seq_len, dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with the same shapes as inputs.
    """
    if triton is None:
        raise ImportError(
            "triton is required. Install it with: pip install triton"
        )
    return _fused_rope_forward_impl(q, k, cos, sin)


if triton is not None:

    @triton.jit
    def _fused_rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_x_b,
        stride_x_s,
        stride_x_d,
        stride_cos_s,
        stride_cos_d,
        stride_out_b,
        stride_out_s,
        stride_out_d,
        HALF_D,
        BLOCK_SIZE: tl.constexpr,
    ):
        b = tl.program_id(0)
        s = tl.program_id(1)

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < HALF_D

        x0 = tl.load(
            x_ptr + b * stride_x_b + s * stride_x_s + offsets * stride_x_d,
            mask=mask,
        )
        x1 = tl.load(
            x_ptr + b * stride_x_b + s * stride_x_s
            + (HALF_D + offsets) * stride_x_d,
            mask=mask,
        )

        c = tl.load(
            cos_ptr + s * stride_cos_s + offsets * stride_cos_d,
            mask=mask,
        )
        sn = tl.load(
            sin_ptr + s * stride_cos_s + offsets * stride_cos_d,
            mask=mask,
        )

        out0 = x0 * c - x1 * sn
        out1 = x1 * c + x0 * sn

        base = out_ptr + b * stride_out_b + s * stride_out_s
        tl.store(base + offsets * stride_out_d, out0, mask=mask)
        tl.store(base + (HALF_D + offsets) * stride_out_d, out1, mask=mask)

    class _FusedRoPE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos, sin):
            ctx.save_for_backward(cos, sin)
            return _fused_rope_forward_core(x, cos, sin)

        @staticmethod
        def backward(ctx, grad_output):
            cos, sin = ctx.saved_tensors
            # Forward applies rotation matrix [[c, -s], [s, c]] to x. Its gradient
            # w.r.t. x is the transpose [[c, s], [-s, c]], which is the same rotation
            # kernel run with sin negated (a rotation matrix's transpose is its inverse).
            grad_input = _fused_rope_forward_core(grad_output, cos, -sin)
            return grad_input, None, None

    def _fused_rope_forward_core(x, cos, sin):
        orig_shape = x.shape
        *leading, seq_len, dim = orig_shape
        batch_total = 1
        for d in leading:
            batch_total *= d

        x_2d = x.reshape(batch_total, seq_len, dim)
        out = torch.empty_like(x_2d)

        half_dim = dim // 2
        block_size = triton.next_power_of_2(half_dim)

        grid = (batch_total, seq_len)
        _fused_rope_kernel[grid](
            x_2d,
            cos,
            sin,
            out,
            x_2d.stride(0),
            x_2d.stride(1),
            x_2d.stride(2),
            cos.stride(0),
            cos.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            half_dim,
            BLOCK_SIZE=block_size,
        )
        return out.reshape(orig_shape)

    def _fused_rope_forward_impl(q, k, cos, sin):
        q_out = _FusedRoPE.apply(q, cos, sin)
        k_out = _FusedRoPE.apply(k, cos, sin)
        return q_out, k_out

    # ------------------------------------------------------------------ #
    # BSHD kernel — adjacent-pairs RoPE for (B, S, H, D) layout           #
    # Used by torchtune and any framework whose tensors are seq-major.     #
    # ------------------------------------------------------------------ #

    @triton.jit
    def _fused_rope_bshd_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        H,
        stride_xb,
        stride_xs,
        stride_xh,
        stride_xd,
        stride_cs,
        stride_cd,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        HALF_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Kernel grid: (B * H, S).

        Implements adjacent-pairs rotation: pairs (x[2i], x[2i+1]) for each i.
        Handles any (B, S, H, D) memory layout via explicit strides — no copies.
        """
        bh_id = tl.program_id(0)
        s_id = tl.program_id(1)

        b_id = bh_id // H
        h_id = bh_id % H

        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < HALF_D

        x_base = (
            x_ptr + b_id * stride_xb + s_id * stride_xs + h_id * stride_xh
        )
        x_even = tl.load(x_base + offs * 2 * stride_xd, mask=mask)
        x_odd = tl.load(x_base + (offs * 2 + 1) * stride_xd, mask=mask)

        cs_base = cos_ptr + s_id * stride_cs
        sn_base = sin_ptr + s_id * stride_cs
        c = tl.load(cs_base + offs * stride_cd, mask=mask)
        sn = tl.load(sn_base + offs * stride_cd, mask=mask)

        out_e = x_even * c - x_odd * sn
        out_o = x_odd * c + x_even * sn

        o_base = (
            out_ptr + b_id * stride_ob + s_id * stride_os + h_id * stride_oh
        )
        tl.store(o_base + offs * 2 * stride_od, out_e, mask=mask)
        tl.store(o_base + (offs * 2 + 1) * stride_od, out_o, mask=mask)

    class _FusedRoPEBSHD(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos, sin):
            ctx.save_for_backward(cos, sin)
            return _fused_rope_bshd_core(x, cos, sin)

        @staticmethod
        def backward(ctx, grad_output):
            cos, sin = ctx.saved_tensors
            return _fused_rope_bshd_core(grad_output, cos, -sin), None, None

    def _fused_rope_bshd_core(x, cos, sin):
        assert x.dim() == 4, f"expected (B, S, H, D), got {x.shape}"
        B, S, H, D = x.shape
        out = torch.empty_like(x)
        half_d = D // 2
        block_size = triton.next_power_of_2(half_d)
        _fused_rope_bshd_kernel[(B * H, S)](
            x,
            cos,
            sin,
            out,
            H,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            cos.stride(0),
            cos.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            HALF_D=half_d,
            BLOCK_SIZE=block_size,
        )
        return out

    def _fused_rope_bshd_impl(x, cos, sin):
        return _FusedRoPEBSHD.apply(x, cos, sin)
