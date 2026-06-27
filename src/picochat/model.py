import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, dim: int, hidden_ff_dim: int, num_heads: int):
    super().__init__()

    self.attention = nn.Sequential(
      nn.RMSNorm(dim),
      CausalSelfAttention(dim, num_heads)
    )

    self.ffn = nn.Sequential(
      nn.RMSNorm(dim),
      SwiGLU(dim, hidden_ff_dim)
    )

  def forward(self, x):
    x = x + self.attention(x)
    x = x + self.ffn(x)
    return x

class SwiGLU(nn.Module):

  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.hidden_dim = hidden_dim

    self.w_in = nn.Linear(dim, hidden_dim * 2, bias=False)
    self.w_out = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x):
    w, v = self.w_in(x).chunk(2, dim=-1)
    return self.w_out(nn.functional.silu(w) * v)


class CausalSelfAttention(nn.Module):

  def __init__(self, dim: int, num_heads: int, rope_module: nn.Module = None):
    super().__init__()
    self.num_heads = num_heads
    self.head_size = dim // num_heads
    assert self.head_size * self.num_heads == dim, "num_heads must be a factor of dim"

    self.w_qkv = nn.Linear(dim, dim*3, bias=False)
    self.w_output = nn.Linear(dim, dim, bias=False)

    self.rope = rope_module if rope_module is not None else RoPE(self.head_size)

  def forward(self, x):
    q, k, v = self.w_qkv(x).chunk(3, dim=-1)

    q = q.view((x.shape[0], x.shape[1], self.num_heads, self.head_size)).transpose(1, 2)
    k = k.view((x.shape[0], x.shape[1], self.num_heads, self.head_size)).transpose(1, 2)
    v = v.view((x.shape[0], x.shape[1], self.num_heads, self.head_size)).transpose(1, 2)

    q, k = self.rope(q), self.rope(k)
    attn = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).reshape(x.shape)

    return self.w_output(attn)


class RoPE(nn.Module):

  def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
    super().__init__()
    self.half_d = dim // 2

    with torch.no_grad():
      inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
      freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)

      self.register_buffer("cos_table", freqs.cos(), persistent=False)
      self.register_buffer("sin_table", freqs.sin(), persistent=False)

  def forward(self, x):
    """(batch, num_heads, seq_len, head_dim)"""
    seq_len = x.shape[-2]
    return torch.cat([
      x[..., :self.half_d] * self.cos_table[:seq_len, :] - x[..., self.half_d:] * self.sin_table[:seq_len, :],
      x[..., :self.half_d] * self.sin_table[:seq_len, :] + x[..., self.half_d:] * self.cos_table[:seq_len, :],
    ], dim=-1)


"""
class TestRoPEModule(unittest.TestCase):
  def test_magnitude_preservation(self, dim=512, max_seq_len=1024):
    rope = RoPE(dim, max_seq_len)

    x = torch.randn((6, 3, max_seq_len, dim))
    x_rotated = rope(x)

    self.assertEqual(x_rotated.shape, x.shape, "Shape mismatch!")
    self.assertTrue(torch.allclose(torch.norm(x, dim=-1), torch.norm(x_rotated, dim=-1)), "Tensor norm was not preserved")

  def test_relative_positioning(self, dim=512):
    rope = RoPE(dim, 20)
    r = torch.randn((2, dim))
    x = torch.empty((20, dim))
    x[:10, :] = r[0, :]
    x[10:, :] = r[1, :]
    x = rope(x)

    dots = [torch.dot(x[i, :], x[i+10, :]) for i in range(10)]
    for i in range(9):
      self.assertTrue(torch.allclose(dots[i], dots[i+1]))
"""
