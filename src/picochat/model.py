import torch.nn as nn
import torch
import numpy as np

class SwiGLU(nn.Module):
    # TODO: is it worth switching to einsum?

    def __init__(self, data_dim: int, hidden_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._silu = nn.SiLU()
        self._w1 = nn.Parameter(torch.randn((hidden_dim, data_dim)))
        self._w3 = nn.Parameter(torch.randn((hidden_dim, data_dim)))
        self._w2 = nn.Parameter(torch.randn((data_dim, hidden_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        internal = self._silu(x @ self._w1.T) * (x @ self._w3.T)
        return internal @ self._w2.T

class RoPE(nn.Module):
    def __init__(self, theta: float, input_dim: int, max_seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cos_table, sin_table = self._calculate_rfuncs(theta, input_dim, max_seq_len)
        self.register_buffer('_cos_table', cos_table)
        self.register_buffer('_sin_table', sin_table)

    @staticmethod
    @torch.no_grad
    def _calculate_rfuncs(theta: float, input_dim: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns arrays cos(t), sin(t), 
        where t[i, j] = i / theta^(2*j / input_dim), i < max_seq_len, j < input_dim/2
        """
        t = theta ** (torch.arange(0, input_dim, 2).float() / input_dim)
        t.unsqueeze_(0)
        t = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1) @ t.reciprocal()
        
        return torch.cos(t), torch.sin(t)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Todo: rotate across halfway boundary instead of across pair??
        x_rotated = x.clone()
        x_rotated[..., :, ::2] = self._cos_table[token_positions] * x[..., ::2] - self._sin_table[token_positions] * x[..., 1::2]
        x_rotated[..., :, 1::2] = self._sin_table[token_positions] * x[..., ::2] + self._cos_table[token_positions] * x[..., 1::2]

        return x_rotated

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, feedforward_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._p2 = nn.Sequential(
            nn.RMSNorm(hidden_dim),
            SwiGLU()
        )

        self._p1 = nn.Sequential(
            nn.RMSNorm(hidden_dim),

        )


def test_rope():
    r = RoPE(10000.0, 18, 15)
    rotated_x = r(torch.rand((2, 5, 10, 15, 18)), torch.randint(15, (2, 5, 10, 15)))
    print(rotated_x.shape)

test_rope()

@torch.no_grad
def test_swiglu(d_model=64, d_ff=128):
    torch.manual_seed(4)
    in_embeddings = torch.randn(4, 12, d_model)
    ts_state_dict = torch.load("model.pt", map_location="cpu")
    w1_weight, w2_weight, w3_weight = [ts_state_dict[f"_orig_mod.layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    swiglu = SwiGLU(d_model, d_ff)
    swiglu._w1 = nn.Parameter(w1_weight)
    swiglu._w2 = nn.Parameter(w2_weight)
    swiglu._w3.copy_(w3_weight)
    actual_output = swiglu(in_embeddings)
    

    expected_array = np.load('test_swiglu.npz')['array']

    print(expected_array.shape, actual_output.shape)
    print(np.abs(expected_array - actual_output.numpy()).max())

# test_swiglu()