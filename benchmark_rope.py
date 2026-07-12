"""
Benchmark: Pure PyTorch vs torch.compile vs view_as_complex vs Triton for RoPE
Addresses: https://github.com/facebookresearch/xformers/pull/1397#issuecomment-4649811732
"""

import torch
import triton
import triton.language as tl
import time

# ---------------------------------------------------------
# 1. Pure PyTorch Reference — rotate_half (chunk/cat)
# ---------------------------------------------------------
def rope_pytorch(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out

rope_compiled = torch.compile(rope_pytorch, mode="reduce-overhead")

# ---------------------------------------------------------
# 2. view_as_complex — avoids chunk/cat allocations
# ---------------------------------------------------------
def rope_complex(q, k, cos, sin):
    # cos/sin shape: (S, D) — treat as complex (S, D//2)
    def apply(x, cos, sin):
        B, H, S, D = x.shape
        xc = torch.view_as_complex(x.reshape(B, H, S, D // 2, 2).float())
        rc = torch.view_as_complex(torch.stack([cos, sin], dim=-1).float())  # (S, D//2)
        out = torch.view_as_real(xc * rc).flatten(-2).to(x.dtype)
        return out
    # cos/sin passed as (S, D) full-dim; take first D//2 pairs
    cos_h = cos[..., :cos.shape[-1] // 2]
    sin_h = sin[..., :sin.shape[-1] // 2]
    return apply(q, cos_h, sin_h), apply(k, cos_h, sin_h)

rope_complex_compiled = torch.compile(rope_complex, mode="reduce-overhead")

# ---------------------------------------------------------
# 2. Triton Fused Kernel
# ---------------------------------------------------------
@triton.jit
def _rope_fwd_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    Q_out_ptr, K_out_ptr,
    seq_len: tl.constexpr, dim: tl.constexpr, rotary_dim: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_cs, stride_cd,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # Base offsets
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_s * stride_qs
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh + pid_s * stride_ks
    cos_base = Cos_ptr + pid_s * stride_cs
    sin_base = Sin_ptr + pid_s * stride_cs
    q_out_base = Q_out_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_s * stride_qs
    k_out_base = K_out_ptr + pid_b * stride_kb + pid_h * stride_kh + pid_s * stride_ks
    
    # Load Q, K
    q = tl.load(q_base + offs_d, mask=offs_d < dim)
    k = tl.load(k_base + offs_d, mask=offs_d < dim)
    
    # Load cos, sin
    cos = tl.load(cos_base + offs_d, mask=offs_d < rotary_dim)
    sin = tl.load(sin_base + offs_d, mask=offs_d < rotary_dim)
    
    # Compute rotated indices: index i maps to (i + half) % rotary_dim
    half = rotary_dim // 2
    offs_rot = (offs_d + half) % rotary_dim
    
    # Load rotated parts
    q_rot = tl.load(q_base + offs_rot, mask=offs_rot < rotary_dim)
    k_rot = tl.load(k_base + offs_rot, mask=offs_rot < rotary_dim)
    
    # Apply sign flip for second half
    q_rot = tl.where(offs_rot >= half, -q_rot, q_rot)
    k_rot = tl.where(offs_rot >= half, -k_rot, k_rot)
    
    # Apply rotation only within rotary_dim
    mask_rot = offs_d < rotary_dim
    q_out = tl.where(mask_rot, q * cos + q_rot * sin, q)
    k_out = tl.where(mask_rot, k * cos + k_rot * sin, k)
    
    # Store
    tl.store(q_out_base + offs_d, q_out, mask=offs_d < dim)
    tl.store(k_out_base + offs_d, k_out, mask=offs_d < dim)

def rope_triton(q, k, cos, sin):
    B, H, S, D = q.shape
    rotary_dim = cos.shape[-1]
    assert rotary_dim <= D
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    grid = (B, H, S)
    _rope_fwd_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        S, D, rotary_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        BLOCK_D=triton.next_power_of_2(D)
    )
    return q_out, k_out

# ---------------------------------------------------------
# 3. Benchmark Runner
# ---------------------------------------------------------
def benchmark_fn(fn, q, k, cos, sin, warmup=10, runs=50):
    for _ in range(warmup):
        fn(q, k, cos, sin)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(runs):
        fn(q, k, cos, sin)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000

def run_benchmarks():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    B, H, D, rot = 1, 32, 128, 128
    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    print(f"B={B} H={H} D={D} rot_dim={rot}  dtype=float16  device=cuda")
    print(f"{'S':<8} | {'PyTorch':>10} | {'compile':>10} | {'complex':>10} | {'cx+cmp':>10} | {'Triton':>10} | {'vs PT':>7} | {'vs cx':>7}")
    print("-" * 90)

    for S in seq_lens:
        q   = torch.randn(B, H, S, D,  device="cuda", dtype=torch.float16)
        k   = torch.randn(B, H, S, D,  device="cuda", dtype=torch.float16)
        cos = torch.randn(S, D,         device="cuda", dtype=torch.float16)
        sin = torch.randn(S, D,         device="cuda", dtype=torch.float16)

        t_py  = benchmark_fn(rope_pytorch,          q, k, cos, sin)
        t_co  = benchmark_fn(rope_compiled,          q, k, cos, sin)
        t_cx  = benchmark_fn(rope_complex,           q, k, cos, sin)
        t_cxc = benchmark_fn(rope_complex_compiled,  q, k, cos, sin)
        t_tr  = benchmark_fn(rope_triton,            q, k, cos, sin)

        print(f"{S:<8} | {t_py:>9.3f}ms | {t_co:>9.3f}ms | {t_cx:>9.3f}ms | {t_cxc:>9.3f}ms | {t_tr:>9.3f}ms | {t_py/t_tr:>6.2f}x | {t_cx/t_tr:>6.2f}x")

if __name__ == "__main__":
    run_benchmarks()
