from typing import Optional
import torch
from torch import nn
import math
from .flash_attention import _scaled_dot_product_flash_attention


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Construct a linear transformation module."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # initialize by kaiming
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = 2.0 / math.sqrt(in_features + out_features)
        torch.nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # init embedding matrix
        embed = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        std = 1.0
        torch.nn.init.trunc_normal_(embed, std=std, a=-3 * std, b=3 * std)
        self.embed = nn.Parameter(embed)

        # book marking
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Check if tensor contains integer values
        # Validate input tensor
        assert token_ids.dtype in [
            torch.int32,
            torch.int64,
            torch.long,
        ], f"token_ids must be integer tensor, got {token_ids.dtype}"
        shape = token_ids.shape

        token_ids = token_ids.reshape(-1)  # [N,]
        lookup = self.embed[token_ids]  # [N, d]

        shape = list(shape) + [self.embedding_dim]
        lookup = lookup.reshape(shape)

        return lookup


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """layer wised rms norm

        RMSNorm(a) = a / RMS(a) * scale where scale is learnable
        RMS(a) = sqrt(eps + avg(a_i^2))

        upcast input to torch.float32 to prevent overflow when you square the input
        """
        super().__init__()

        # parameters
        self._scale = nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))

        # bookmarking
        self.dim = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): shape: (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: same shape with x
        """
        assert x.ndim == 3 and x.shape[-1] == self.dim

        # compute average of square on last dim (using float32)
        oldtype = x.dtype
        x = x.float()
        square_avg = (x * x).mean(dim=-1, keepdim=True)
        dividor = torch.sqrt(self.eps + square_avg)
        dividor = dividor.to(oldtype)
        x = x.to(oldtype)
        
        out = self._scale * (x / dividor)

        return out


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_filter: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) * W3x)
        shape of each tensor:
            x: d_model
            W1, W3: (d_filter, d_model)
            W2: (d_model, d_filter)
        """
        super().__init__()
        if d_filter is None:
            assert d_model % 3 == 0
            d_filter = 8 * d_model // 3

        self.linear1 = Linear(d_model, d_filter, device, dtype)
        self.linear2 = Linear(d_filter, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_filter, device, dtype)

        # book marking
        self.d_model = d_model
        self.d_filter = d_filter
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = self.linear1(x)
        silu = silu * torch.sigmoid(silu)
        out = self.linear2(silu * self.linear3(x))
        return out
    
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Formular:
            R^i_k = [[cos(theta_{i,k}), -sin(theta_{i,k})],[cos(theta_{i,k}), sin(theta_{i,k})]]
        where
            theta_{i,k} = (i+1) / (Theta)^{2(k+1)/d} where k in [0, d/2), i in [0, max_len)

        for a vector x of size dim
            R^i @ d = [d0 * cos0 - d1 * sin0, d0 * sin0 + d1 * cos0 ... ]
        args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        # when i == 0, thetas = [1 / theta^{2/d}, 1 / theta^{4/d}, ... 1 / theta^{d/d}]
        # theta = (1 / theta) ** [2/d, 4/d, ... d/d]
        assert d_k % 2 == 0
        num_cos = d_k // 2
        assert num_cos > 0
        base = torch.tensor(1.0 / theta, dtype=torch.float64, device=device)
        powers = (
            2.0
            * torch.arange(0, num_cos, device=device, dtype=torch.float64)
            / float(d_k)
        )
        thetas = torch.float_power(base, powers)
        thetas = thetas.to(torch.float32)

        # store cos and sin in a tensor of shape [max_seq_len, d_k // 2]
        # cosines[i, k] = cos(theta_{i, k})
        # cosines = torch.empty(max_seq_len, num_cos, device=device, dtype=torch.float32)
        # sines = torch.empty(max_seq_len, num_cos, device=device, dtype=torch.float32)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        Thetas = positions.reshape((-1, 1)) * thetas.reshape(1, num_cos)
        cosines = torch.cos(Thetas)
        sines = torch.sin(Thetas)

        # store coses and sines
        self.register_buffer("cosines", cosines, persistent=False)
        self.register_buffer("sines", sines, persistent=False)

        # bookmarking
        self.num_cos = num_cos
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): token positions are a tensor of shape (..., seq_len)

        Returns:
            torch.Tensor: _description_
        """
        num_batch, seq_len, d_k = x.shape
        if token_positions.ndim == 1:
            token_positions = token_positions.reshape((1, -1))
            token_positions = torch.tile(token_positions, (num_batch, 1))

        assert token_positions.shape == (num_batch, seq_len)
        assert d_k == self.d_k
        assert torch.all(token_positions < self.max_seq_len)

        # make both x of shape [:, d_k] and token_positions of shape [:]
        x = x.reshape((-1, d_k))
        token_positions = token_positions.reshape(-1)
        coses = self.cosines[token_positions]  # shape [N, d_k // 2]
        sines = self.sines[token_positions]

        # make coses and sines same type with x
        coses = coses.to(x.dtype)
        sines = sines.to(x.dtype)

        # compute even lines each of form: [x0 c0 - x1 s0 ... ]
        x_even = x[:, 0:d_k:2]
        x_odd = x[:, 1:d_k:2]
        out_even = x_even * coses - x_odd * sines  # shape: [N, d_k // 2]

        # compute odd lines each of form: [x0 s0 + x1 c0 ...]
        out_odd = x_even * sines + x_odd * coses  # shape: [N, d_k // 2]

        # concate even and odd
        out = torch.stack([out_even, out_odd], dim=1)  # [N, 2, d_k // 2]
        out = out.transpose(-1, -2)
        out = out.reshape((-1, d_k))
        # equivalent to:
        # out = torch.zeros_like(x)
        # out[:, 0::2] = out_even
        # out[:, 1::2] = out_odd

        # reshape back
        out = out.reshape(num_batch, seq_len, d_k)
        return out


def _scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """scaled dot-product attention function

    Args:
        Q (torch.Tensor): shape (batch_size, ..., seq_len_q, d_k)
        K (torch.Tensor): shape (batch_size, ..., seq_len_k, d_k)
        V (torch.Tensor): shape (batch_size, ..., seq_len_k, d_v)
        mask (torch.Tensor): shape (batch_size, ..., seq_len, seq_len), when set to true attention applied

    Returns:
        torch.Tensor: _description_
    """
    # make input tensor being 4-d
    assert Q.ndim == K.ndim and K.ndim == V.ndim
    if mask is not None:
        assert mask.ndim == Q.ndim

    old_dim = Q.ndim
    if Q.ndim == 3:
        Q = Q.unsqueeze(dim=1)
        K = K.unsqueeze(dim=1)
        V = V.unsqueeze(dim=1)

        if mask is not None:
            mask = mask.unsqueeze(dim=1)

    # compute attention score
    seqlen_q, d_k = Q.shape[-2], Q.shape[-1]
    seqlen_v = V.shape[-2]
    score = (
        Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    )  # [bsize, heads, seqlen_q seqlen_k]
    assert score.shape == (Q.shape[0], Q.shape[1], seqlen_q, seqlen_v)

    # mask score, mask == false fill with '-inf'
    if mask != None:
        # mask = mask.view(1, 1, seqlen_q, seqlen_v)
        assert mask.shape == score.shape
        score = score.masked_fill(mask == False, float("-inf"))

    # compute attention prob
    prob = torch.softmax(score, dim=-1)  # [bsize, heads, seq_len seq_len]

    # apply to v
    out = prob @ V
    if old_dim == 3:
        out = out.squeeze(dim=1)

    return out

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_flash_attention: bool = False,
) -> torch.Tensor:
    """scaled dot-product attention function

    Args:
        Q (torch.Tensor): shape (batch_size, ..., seq_len_q, d_k)
        K (torch.Tensor): shape (batch_size, ..., seq_len_k, d_k)
        V (torch.Tensor): shape (batch_size, ..., seq_len_k, d_v)
        mask (torch.Tensor): shape (batch_size, ..., seq_len, seq_len), when set to true attention applied

    Returns:
        torch.Tensor: _description_
    """
    if not use_flash_attention:
        out = _scaled_dot_product_attention(Q, K, V, mask)
    
    else:
        is_casual = mask is not None
        out = _scaled_dot_product_flash_attention(Q, K, V, is_casual)

    return out


# TODO: add dropout
class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        rope_module: Optional[RotaryPositionalEmbedding] = None,
        use_flash_attention: bool = False,
    ):
        assert d_model % num_heads == 0

        super().__init__()
        # proj maps
        self.proj_q = Linear(d_model, d_model, device, dtype)
        self.proj_k = Linear(d_model, d_model, device, dtype)
        self.proj_v = Linear(d_model, d_model, device, dtype)
        self.proj_o = Linear(d_model, d_model, device, dtype)

        # global mask
        casual_mask = torch.tril(
            torch.ones(size=(max_seq_len, max_seq_len), dtype=torch.bool)
        )
        casual_mask = casual_mask.view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("casual_mask", casual_mask)

        # bookmark
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.rope_module = rope_module
        self.use_flash_attention = use_flash_attention

    def forward(
        self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): shape [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        assert seq_len <= self.max_seq_len and d_model == self.d_model

        # get Q, K, V
        num_heads = self.num_heads
        d_hidden = self.d_hidden
        Q = self.proj_q(x).reshape(batch_size, seq_len, num_heads, d_hidden)
        K = self.proj_k(x).reshape(batch_size, seq_len, num_heads, d_hidden)
        V = self.proj_v(x).reshape(batch_size, seq_len, num_heads, d_hidden)

        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_hidden]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # apply rope
        if self.rope_module is not None:
            if token_positions is None:
                token_positions = torch.arange(0, seq_len, dtype=torch.int64)
                token_positions = token_positions.reshape((1, seq_len))

            assert token_positions.ndim == 2 and token_positions.shape[0] == 1
            Q = Q.reshape(batch_size * num_heads, seq_len, d_hidden)
            K = K.reshape(batch_size * num_heads, seq_len, d_hidden)
            token_positions = token_positions.broadcast_to(
                batch_size * num_heads, seq_len
            )
            Q = self.rope_module.forward(Q, token_positions)
            K = self.rope_module.forward(K, token_positions)
            Q = Q.reshape(batch_size, num_heads, seq_len, d_hidden)
            K = K.reshape(batch_size, num_heads, seq_len, d_hidden)

        # get casual mask
        mask: torch.Tensor = self.casual_mask[:, :, :seq_len, :seq_len]
        mask = mask.broadcast_to(batch_size, num_heads, seq_len, seq_len)

        # apply attention
        O = scaled_dot_product_attention(
            Q, K, V, mask, use_flash_attention=self.use_flash_attention
        )  # [batch_size, num_heads, seq_len, d_hidden]

        # apply projection
        O = O.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.proj_o(O)

        return out


class TransformerBlock(nn.Module):
    """
    A Transformer block contains two ‘sublayers’, one for the multihead self attention, and another for the feed-forward network.
    In each sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in the
    residual connection.
    First Half:
        y = x + MultiHeadSelfAttention(RMSNorm(x)).

    Second Half
        y = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        eps: float = 0.00001,
        rope_module: Optional[RotaryPositionalEmbedding] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model, eps, device, dtype)
        self.attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, device=device, dtype=dtype, rope_module=rope_module, use_flash_attention=use_flash_attention)
        self.rmsnorm2 = RMSNorm(d_model, eps, device, dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)

        # bookmark
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): shape [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        y = x + self.attention(self.rmsnorm1(x))
        y = y + self.swiglu(self.rmsnorm2(y))

        return y


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 0.00001,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = False
    ):
        """

        Args:
            vocab_size (int): The size of the vocabulary
            context_length (int): The maximum context length, necessary for determining the dimensionality of the position embedding matrix
            num_layers (int): The number of Transformer blocks to use
            d_model (int): dimension of the model
            num_heads (int): number of attention heads
            d_ff (int): filter of swiglu
            rope_theta (float):  
            eps (float, optional): eps of rmsnorm. Defaults to 0.00001.
            device (Optional[torch.device], optional): _description_. Defaults to None.
            dtype (Optional[torch.dtype], optional): _description_. Defaults to None.
        """
        super().__init__()

        self.embed = Embedding(vocab_size, d_model, device, dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device)
        
        # store blocks in list
        blocks = [TransformerBlock(d_model, num_heads, d_ff, context_length, eps, self.rope, device, dtype, use_flash_attention) for i in range(num_layers)]
        self.blocks = nn.ModuleList(blocks)

        self.head_norm = RMSNorm(d_model, eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        # bookmarking
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length)

        Returns:
            torch.Tensor: probs
        """
        bsize, seqlen = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.head_norm(x)
        x = self.lm_head(x)

        assert x.shape == (bsize, seqlen, self.vocab_size)
        # probs = torch.softmax(x, dim=-1)

        return x
    

def my_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        inputs (torch.Tensor): Float[Tensor, " batch_size vocab_size"]
        targets (torch.Tensor): targets: Int[Tensor, " batch_size"]

    Returns:
        torch.Tensor: Float[Tensor, ""]

    """
    # To handle stability:
    # 1. remove max from logits
    # 2. use logits - log sum exp (logits) instead of log (exp(logits) / exp_sum)
 
    bsize, num_vocab = inputs.shape
    # remove max for stability
    inputs_stable = inputs - inputs.max(dim=-1, keepdim=True).values
    exp_input = torch.exp(inputs_stable)
    divider = exp_input.sum(dim=-1, keepdim=False)
    # probs = exp_input[torch.arange(0, bsize), targets] / divider
    # return -(torch.log(probs).mean())

    logit_diff = inputs_stable[torch.arange(0, bsize), targets] - torch.log(divider)
    return -(logit_diff.mean())

def my_softmax(inputs: torch.Tensor, dim: int):
    inputs_stable = inputs - inputs.max(dim=dim, keepdim=True).values
    exp_inputs = torch.exp(inputs_stable)
    smx = exp_inputs / exp_inputs.sum(dim=dim, keepdim=True)
    return smx