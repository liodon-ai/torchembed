# Categorical embeddings

Two classes for tabular and structured data:

| Class | Description |
|---|---|
| `EntityEmbedding` | Single categorical feature with auto-sized dimension |
| `MultiCategoricalEmbedding` | Multiple categorical columns at once |

---

## EntityEmbedding

Learned embedding for a single categorical feature. Wraps `nn.Embedding` with automatic dimension sizing and optional dropout.

The auto-sizing heuristic (fast.ai rule) empirically works well for tabular data without manual tuning: `min(600, round(1.6 * n_cats ** 0.56))`.

**Reference:** Howard & Gugger, "Deep Learning for Coders with fastai and PyTorch"

### Constructor

```python
EntityEmbedding(
    num_categories: int,
    embed_dim: Optional[int] = None,
    dropout: float = 0.0,
    padding_idx: Optional[int] = None,
)
```

| Param | Description |
|---|---|
| `num_categories` | Vocabulary size. Indices must be in [0, num_categories - 1]. |
| `embed_dim` | Embedding dimension. If None, uses auto-sizing heuristic. |
| `dropout` | Dropout rate applied to embeddings. |
| `padding_idx` | Index that produces a zero embedding. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(...)` | Category indices (long) |
| Returns | `(..., embed_dim)` | Embeddings |

### Example

```python
from torchembed.categorical import EntityEmbedding

emb = EntityEmbedding(num_categories=1000)
x = torch.randint(0, 1000, (32,))
out = emb(x)   # (32, auto_sized_dim)
```

---

## MultiCategoricalEmbedding

Joint embedding for multiple categorical columns. Intended for tabular data where each row has several categorical features (e.g. country, product category, day of week). Creates one `EntityEmbedding` per column and concatenates the results.

### Constructor

```python
MultiCategoricalEmbedding(
    cardinalities: Sequence[int],
    embed_dims: Optional[Sequence[int]] = None,
    dropout: float = 0.0,
    padding_idx: Optional[int] = None,
)
```

| Param | Description |
|---|---|
| `cardinalities` | Vocabulary sizes, one per column. E.g. `[50, 7, 12]`. |
| `embed_dims` | Per-column dimensions. If None, uses auto-sizing. |
| `dropout` | Dropout applied to each embedding independently. |
| `padding_idx` | Shared padding index for all columns. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(batch, num_columns)` | Category indices (long). Column order must match `cardinalities`. |
| Returns | `(batch, output_dim)` | Concatenated embeddings |

### Properties

- `output_dim`: total output dimension (sum of all column embed dims)
- `embedding_dims()`: returns `[(num_cats, embed_dim), ...]` for each column

### Example

```python
from torchembed.categorical import MultiCategoricalEmbedding

# 3 categorical columns: country (50), weekday (7), month (12)
emb = MultiCategoricalEmbedding(cardinalities=[50, 7, 12])
print(emb.output_dim)

x = torch.stack([
    torch.randint(0, 50, (32,)),
    torch.randint(0, 7, (32,)),
    torch.randint(0, 12, (32,)),
], dim=1)
out = emb(x)   # (32, output_dim)
```
