# Positional embeddings

Four positional encoding strategies for transformer models:

| Class | Description | Used in |
|---|---|---|
| `RotaryEmbedding` | Rotate Q/K vectors in 2D subspaces | LLaMA, Mistral, Falcon |
| `ALiBiEmbedding` | Linear distance bias on attention scores | BLOOM, MPT |
| `SinusoidalEmbedding` | Fixed sin/cos position encoding | Original Transformer |
| `LearnedPositionalEmbedding` | Trainable position lookup table | BERT, GPT-2 |

---

## RotaryEmbedding

Rotary Position Embedding (RoPE) encodes position by rotating query and key vectors in 2D subspaces. Unlike additive embeddings, RoPE is applied directly to Q and K inside the attention layer, not to the input sequence.

**Reference:** Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" ([arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864))

### Constructor

```python
RotaryEmbedding(
    dim: int,
    max_seq_len: int = 2048,
    base: int = 10000,
    use_fused: bool = False,
    device: Optional[torch.device] = None,
)
```

| Param | Description |
|---|---|
| `dim` | Head dimension (must be even). Typically `d_model // num_heads`. |
| `max_seq_len` | Maximum sequence length to precompute. Longer sequences computed on the fly. |
| `base` | Base for frequency geometric progression. Default 10000. LLaMA 3 uses 500000. |
| `use_fused` | If True, uses a fused triton kernel on GPU (requires `torchembed[triton]`). |

### Forward

```python
forward(q, k, seq_dim=-2) -> tuple[Tensor, Tensor]
```

| Param | Shape | Description |
|---|---|---|
| `q` | `(..., seq_len, dim)` | Query tensor |
| `k` | `(..., seq_len, dim)` | Key tensor |
| Returns | `(..., seq_len, dim)` each | Rotated Q and K |

### Properties

- No trainable parameters (pure rotation)
- Preserves vector norms
- Automatically extends cache for sequences longer than `max_seq_len`

### Examples

**Basic usage (LLaMA-style):**

```python
from torchembed.positional import RotaryEmbedding

rope = RotaryEmbedding(dim=128, base=500000)
q = torch.randn(2, 8, 64, 128)   # (batch, heads, seq, dim)
k = torch.randn(2, 8, 64, 128)
q, k = rope(q, k)
```

**With fused triton kernel (GPU):**

```python
rope = RotaryEmbedding(dim=128, use_fused=True).to("cuda")
q, k = rope(q.cuda(), k.cuda())
```

---

## ALiBiEmbedding

Attention with Linear Biases adds a fixed, non-learned bias to attention scores that penalizes distance between tokens linearly. This allows strong extrapolation to longer sequences than seen during training.

**Reference:** Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" ([arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409))

### Constructor

```python
ALiBiEmbedding(
    num_heads: int,
    max_seq_len: int = 2048,
)
```

| Param | Description |
|---|---|
| `num_heads` | Number of attention heads. Each head gets a different slope. |
| `max_seq_len` | Maximum sequence length to precompute biases for. |

### Forward

```python
forward(attn_scores) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `attn_scores` | `(batch, heads, seq_q, seq_k)` | Raw attention logits |
| Returns | same as input | Scores with ALiBi bias added |

### Properties

- No trainable parameters
- Handles non-power-of-2 head counts (interleaves slopes)
- Bias is non-positive (penalizes, never rewards distance)
- Zero bias at diagonal (self-attention, distance = 0)

### Example

```python
from torchembed.positional import ALiBiEmbedding

alibi = ALiBiEmbedding(num_heads=8)
scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
scores = alibi(scores)
weights = scores.softmax(-1)
```

---

## SinusoidalEmbedding

Fixed sinusoidal positional encoding from "Attention Is All You Need." Adds a non-learned, frequency-based positional signal to input embeddings. The encoding is deterministic and can generalize slightly beyond the training sequence length.

**Reference:** Vaswani et al., "Attention Is All You Need" ([arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))

### Constructor

```python
SinusoidalEmbedding(
    dim: int,
    max_seq_len: int = 4096,
    dropout: float = 0.0,
    learned_scale: bool = False,
)
```

| Param | Description |
|---|---|
| `dim` | Embedding dimension (must be even). |
| `max_seq_len` | Maximum supported sequence length. |
| `dropout` | Dropout applied after adding the embedding. |
| `learned_scale` | If True, adds a single learned scalar to scale the signal. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(batch, seq_len, dim)` | Input token embeddings |
| Returns | same as input | Token embeddings + positional encoding |

---

## LearnedPositionalEmbedding

Standard learned positional embedding — a simple lookup table mapping each position index to a learnable vector. Used in BERT, GPT-2, and many other models. Simpler than sinusoidal but cannot extrapolate beyond the training sequence length.

### Constructor

```python
LearnedPositionalEmbedding(
    max_seq_len: int,
    dim: int,
    dropout: float = 0.0,
    padding_idx: Optional[int] = None,
)
```

| Param | Description |
|---|---|
| `max_seq_len` | Maximum sequence length (vocabulary size for positions). |
| `dim` | Embedding dimension. |
| `dropout` | Dropout rate. |
| `padding_idx` | If set, the embedding at this index is not updated. |

### Forward

```python
forward(x, offset=0) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(batch, seq_len, dim)` | Input token embeddings |
| `offset` | int | Starting position index (useful for KV-cache) |
| Returns | same as input | Token embeddings + positional embeddings |

### Example (with KV-cache offset)

```python
emb = LearnedPositionalEmbedding(max_seq_len=4096, dim=768)

# Training: full sequence
x = torch.randn(2, 128, 768)
out = emb(x)   # positions 0..127

# Inference with KV-cache: one token at a time
for step in range(prompt_len, total_len):
    x = torch.randn(2, 1, 768)
    out = emb(x, offset=step)   # position = step
```
