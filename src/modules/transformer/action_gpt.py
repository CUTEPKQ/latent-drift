"""
Refer to 
https://github.com/FoundationVision/LlamaGen
https://github.com/kakaobrain/rq-vae-transformer
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
import math
from einops import rearrange,repeat


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def q_schedule(bs, low, high, device=None):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low)) + low

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k anclass GPT(nn.Mo        self.config = GPTConfig(vocab_size=vocab_size, grid_thw=grid_thw,
                    embd_pdrop=embd_pdrop, resid_dropout_p=resid_dropout_p, attn_dropout_p=attn_dropout_p, 
                    spatial_n_layer=spatial_n_layer, factorized_n_layer=factorized_n_layer, n_head=n_head, dim=dim, 
                    ffn_dropout_p=ffn_dropout_p, drop_path_rate=drop_path_rate, n_unmasked=n_unmasked,
                    class_num=class_num, token_drop=token_drop, cls_token_num=cls_token_num, rope_base=rope_base, norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, initializer_range=initalizer_range, multiple_of=multiple_of,
                    max_batch_size=max_batch_size, max_seq_len=max_seq_len, n_kv_head=n_kv_head, factorized_k=factorized_k,
                    factorized_bits=factorized_bits, mrope_section=mrope_section)   def __init__(self, vocab_size, grid_thw, spatial_n_layer=12, n_head=8, dim=256, factorized_n_layer=2,
                 embd_pdrop=0., resid_dropout_p=0., attn_dropout_p=0., ffn_dropout_p=0.1, drop_path_rate=0.0, 
                 n_unmasked=0, max_batch_size=32, max_seq_len=2048,
                 class_num=1000, token_drop=0.1, cls_token_num=1, rope_base=10000,
                 norm_eps=1e-5, ffn_dim_multiplier=None, initalizer_range=0.02, multiple_of=256, n_kv_head=None, 
                 factorized_k=2, factorized_bits=[9, 9], mrope_section=[16, 24, 24]):cleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x




def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dict__.keys()}

    def get(self, key, default=None):
        return getattr(self, key, default)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

##Modified from https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size) # 1001
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # labels = torch.where(drop_ids, torch.tensor(self.num_classes, dtype=labels.dtype, device=labels.device), labels)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        labels = labels.squeeze(-1) # [Batch]
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)
        
        # mrope section for multimodal rotary embedding
        self.mrope_section = getattr(config, 'mrope_section', [self.head_dim // 2, self.head_dim // 4, self.head_dim // 4])
        
        # rope type: 'multimodal' or 'simple'
        self.rope_type = getattr(config, 'rope_type', 'multimodal')

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: Optional[torch.Tensor], 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        # apply rotary position embedding based on rope_type
        if self.rope_type == 'simple':
            # Use original RoPE (for intra blocks)
            # cos can be: [seqlen, head_dim//2, 2] (freqs_cis format) or [seqlen, head_dim//2]
            # If sin is None, cos should be in freqs_cis format [seqlen, head_dim//2, 2]
            if sin is None:
                # cos is actually freqs_cis in [seqlen, head_dim//2, 2] format
                if cos.dim() != 3 or cos.shape[-1] != 2:
                    raise ValueError(f"When sin is None, cos should be freqs_cis with shape [seqlen, head_dim//2, 2], got {cos.shape}")
                freqs_cis = cos
            elif cos.dim() == 3 and cos.shape[-1] == 2:  # [seqlen, head_dim//2, 2]
                # Already in freqs_cis format, use as is
                freqs_cis = cos
            elif cos.dim() == 2:  # [seqlen, head_dim//2]
                # Create freqs_cis from cos/sin
                freqs_cis = torch.stack([cos, sin], dim=-1)  # [seqlen, head_dim//2, 2]
            else:
                # Unexpected format, raise error
                raise ValueError(f"Unexpected cos/sin shape for simple RoPE: cos.shape={cos.shape}, expected [seqlen, head_dim//2] or [seqlen, head_dim//2, 2]")
            
            # Ensure freqs_cis is contiguous before passing to apply_simple_rotary_emb
            if not freqs_cis.is_contiguous():
                freqs_cis = freqs_cis.contiguous()
            
            xq, xk = apply_simple_rotary_emb(xq, xk, freqs_cis)
        else:
            # Use multimodal RoPE (default, for spatial blocks)
            # Ensure cos/sin have the correct shape [3, 1, seqlen, head_dim]
            if cos.dim() != 4 or cos.shape[0] != 3:
                raise ValueError(f"Unexpected cos/sin shape for multimodal RoPE: cos.shape={cos.shape}, expected [3, 1, seqlen, head_dim]")
            xq, xk = apply_m_modify_multimodal_rotary_pos_emb(xq, xk, cos, sin, self.mrope_section, unsqueeze_dim=2)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class Block(nn.Module):
    def __init__(self, config, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: Optional[torch.Tensor], start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), cos, sin, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(v_out.dtype)

        return k_out, v_out


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_hid: int, max_len: int = 16):
        """
        :param d_hid: embedding 维度
        :param base: 频率基数，默认 10000
        :param max_len: 最大支持的位置数
        """
        super().__init__()
        self.d_hid = d_hid
        base = 10000.0

        # [max_len, d_hid] 的查找表
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)     # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * -(math.log(base) / d_hid))  # [d_hid/2]

        pe = torch.zeros(max_len, d_hid)                                    # [max_len, d_hid]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # 注册为 buffer，不训练，但可随模型移动
        self.register_buffer("pe_table", pe)

    @torch.no_grad()
    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """
        :param index: [B] 的位置索引
        :return: [B, d_hid] 的位置编码
        """
        return self.pe_table[index]


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def precompute_freqs_cis_1d(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=0):
    """
    Precompute 1D rotary position embeddings for intra blocks
    Args:
        seq_len: sequence length (e.g., 5 for [h, emb[0], emb[1], emb[2], emb[3]])
        n_elem: head dimension
        base: RoPE base frequency
        cls_token_num: number of cls tokens (default 0, will use 1 for intra blocks)
    Returns:
        freqs_cis: [cls_token_num + seq_len, n_elem // 2, 2] with cos/sin embeddings
    """
    half_dim = n_elem // 2
    # Compute frequencies
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    
    # Create position indices
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # [seq_len, head_dim // 4]
    
    # Duplicate for full half_dim (simple 1D encoding)
    freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim // 2]
    
    # Compute cos and sin
    freqs_cos = torch.cos(freqs)  # [seq_len, head_dim // 2]
    freqs_sin = torch.sin(freqs)  # [seq_len, head_dim // 2]
    
    # Stack to get [seq_len, head_dim // 2, 2]
    cache = torch.stack([freqs_cos, freqs_sin], dim=-1)  # [seq_len, head_dim // 2, 2]
    
    # Add cls_token at position 0 (cos(0)=1, sin(0)=0 - identity transform that preserves Q/K)
    if cls_token_num > 0:
        cls_cache = torch.zeros(cls_token_num, half_dim, 2)
        cls_cache[..., 0] = 1.0  # cos(0) = 1
        cls_cache[..., 1] = 0.0  # sin(0) = 0
        cond_cache = torch.cat([cls_cache, cache], dim=0)  # [cls_token_num + seq_len, head_dim // 2, 2]
    else:
        cond_cache = cache

    return cond_cache

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary position embeddings (from gpt.py)
    Args:
        x: (bs, seq_len, n_head, head_dim)
        freqs_cis: (seq_len, head_dim // 2, 2)
    """
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    original_dtype = x.dtype
    
    # Ensure x is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Convert x to float32 for computation
    xshaped = x.float().view(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    
    # Ensure freqs_cis is on the same device as x and is contiguous
    if freqs_cis.device != x.device:
        freqs_cis = freqs_cis.to(x.device)
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    
    # Reshape freqs_cis - use reshape instead of view for safety
    freqs_cis = freqs_cis.reshape(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    
    # Convert freqs_cis to float32 if needed
    if freqs_cis.dtype != torch.float32:
        freqs_cis = freqs_cis.float()
    
    # Apply rotation
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    # Convert back to original dtype
    return x_out2.type(original_dtype)

def apply_simple_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply simple 1D rotary position embeddings (original RoPE from gpt.py)
    Args:
        xq: query tensor [bsz, seqlen, n_head, head_dim]
        xk: key tensor [bsz, seqlen, n_kv_head, head_dim]
        freqs_cis: precomputed cos/sin [seqlen, head_dim//2, 2]
    Returns:
        xq_out, xk_out: rotated query and key tensors
    """
    xq_out = apply_rotary_emb(xq, freqs_cis)
    xk_out = apply_rotary_emb(xk, freqs_cis)
    return xq_out, xk_out

def compute_default_rope_parameters(
    rope_base: float = 10000.0,
    head_dim: int = 128,
    device: Optional["torch.device"] = None) -> torch.Tensor:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = rope_base
    partial_rotary_factor = 1.0
    dim = int(head_dim * partial_rotary_factor)

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=128,
        base=10000,
        device=None,
    ):
        super().__init__()
   
        inv_freq = compute_default_rope_parameters(base, dim, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # import torch; torch.set_printoptions(profile='full')
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def get_t_scale_rope_index(
    seq_len: int,
    grid_thw,   # (T, H, W) for the video grid
    scale_factor: float = 1.0,
    device: Optional[torch.device] = None,
    condition_token_num: int = 1,
) -> torch.Tensor:
    """
    仅根据固定长度和单一视频块 grid 生成三通道 mRoPE 位置:
    - 序列第 0 位: condition token（t/h/w = 0）
    - 序列 1..L-1: 视频 patch tokens

    返回:
        position_ids: [3, 1, L] (float)
    """

    T, H, W = grid_thw
    merge = 1
    # merge = getattr(self.config.vision_config, "spatial_merge_size", 1)
    Hm, Wm = H // merge, W // merge
    if Hm <= 0 or Wm <= 0:
        raise ValueError(f"Invalid merged grid: H//merge={Hm}, W//merge={Wm}, "
                            f"got H={H}, W={W}, merge={merge}")

    total_patches = T * Hm * Wm
    num_visual_tokens = seq_len - condition_token_num
    if num_visual_tokens < 0:
        raise ValueError(f"seq_len must be >= 1 (got {seq_len})")

    if num_visual_tokens > total_patches:
        raise ValueError(
            f"seq_len-1 ({num_visual_tokens}) exceeds available patches T*H'*W'={total_patches} "
            f"(T={T}, H'={Hm}, W'={Wm})."
        )

    # 分配输出: [3, 1, L]
    position_ids = torch.zeros((3, 1, seq_len), dtype=torch.float32, device=device)

    if num_visual_tokens == 0:
        # 只有 condition，一个 0 就够了
        return position_ids

    # --- 构造体素网格索引（与原实现一致） ---
    # 展平顺序: 先 t，再 h，再 w
    t_index = torch.arange(T, device=device).view(-1, 1).expand(-1, Hm * Wm).reshape(-1)
    h_index = (
        torch.arange(Hm, device=device)
        .view(1, -1, 1)
        .expand(T, -1, Wm)
        .reshape(-1)
        - (Hm - 1) // 2
    )
    w_index = (
        torch.arange(Wm, device=device)
        .view(1, 1, -1)
        .expand(T, Hm, -1)
        .reshape(-1)
        - (Wm - 1) // 2
    )

    # 时间步长缩放
    t_index = t_index.to(torch.float32) * scale_factor

    # 文本长度=1（第 0 位是 condition），视觉块从 t=1 开始
    # “体对角线”推进: 把 h/w 的偏移加到相同的 t 基线
    t_base = 1.0
    t_index = t_index + t_base
    h_index = h_index.to(torch.float32) + t_index
    w_index = w_index.to(torch.float32) + t_index

    # 只取前 num_visual_tokens 个 patch（若 L-1 < T*H'*W'）
    t_index = t_index[:num_visual_tokens]
    h_index = h_index[:num_visual_tokens]
    w_index = w_index[:num_visual_tokens]

    # 写入到 position_ids[:, 0, :]
    # idx=0: condition -> (0,0,0) 已是默认
    position_ids[0, 0, condition_token_num:] = t_index
    position_ids[1, 0, condition_token_num:] = h_index
    position_ids[2, 0, condition_token_num:] = w_index

    return position_ids

def apply_m_modify_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # cos:(3,1,seq_len,head_dim)
    # Ensure cos and sin have the correct shape
    if cos.dim() != 4 or cos.shape[0] != 3:
        raise ValueError(f"apply_m_modify_multimodal_rotary_pos_emb expects cos/sin with shape [3, 1, seq_len, head_dim], got {cos.shape}")
    
    # add x, y dim -> (16, 48)
    mrope_section = [mrope_section[0], mrope_section[1] + mrope_section[2]]
    mrope_section = mrope_section * 2
    # adjust t last -> (48, 16, 48, 16)
    mrope_section = mrope_section[::-1]
    index = 0
    result_cos = []
    result_sin = []
    # get x1, y1, x2, y2, ..., t1, t2, ...
    for i, section in enumerate(mrope_section):
        if i % 2 == 0:
            for j in range(section):
                # import pdb; pdb.set_trace()
                row = 1 if j % 2 == 0 else 2
                
                result_cos.append(cos[row, ..., index: index + 1])
                result_sin.append(sin[row, ..., index: index + 1])
                index += 1
        else:
            result_cos.append(cos[0, ..., index:index + section])
            result_sin.append(sin[0, ..., index:index + section])
            index += section
    cos, sin = torch.cat(result_cos, dim=-1).unsqueeze(dim=unsqueeze_dim), torch.cat(result_sin, dim=-1).unsqueeze(dim=unsqueeze_dim)
    # cos : (1,seq_len,1,head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GatedConditionEmbedding(nn.Module):
    def __init__(self, input_dim=8, dim=2048):
        super().__init__()
        
        # 1. 定义门控参数 (Learnable Gates)
        # 初始化为 1.0，表示初始状态下所有特征都同等重要且保留原值
        self.feature_gates = nn.Parameter(torch.ones(input_dim))
        
        # 2. 投影层 (MLP)
        self.projector = MLP(in_features=input_dim, hidden_features=dim//2, out_features=dim)

    def forward(self, manual_features):
        # manual_features: [B, 8]
        
        # --- 关键步骤 ---
        # 利用广播机制：[B, 8] * [8] -> [B, 8]
        # 模型会自己调整 feature_gates 的值
        weighted_features = manual_features * self.feature_gates
        
        # 投影: [B, 8] -> [B, 2048]
        feat_emb = self.projector(weighted_features).unsqueeze(1) # -> [B, 1, 2048]
        
        return feat_emb

class ActionGPT(nn.Module):
    def __init__(self, indim,vocab_size, grid_thw, spatial_n_layer=12, n_head=8, dim=256, factorized_n_layer=2,
                 embd_pdrop=0., resid_dropout_p=0., attn_dropout_p=0., ffn_dropout_p=0.1, drop_path_rate=0.0, 
                 n_unmasked=0, max_batch_size=32, max_seq_len=2048, use_pool = False,
                 class_num=1000, token_drop=0.1, cls_token_num=1, rope_base=10000,cond_drop_prob=0.1,use_handcraft=False,
                 norm_eps=1e-5, ffn_dim_multiplier=None, initalizer_range=0.02, multiple_of=256, n_kv_head=None, 
                 factorized_k=2, mrope_section=[16, 24, 24], t_scale_factor=2.0, target_hw=[5,4],use_cfg=False):
        super().__init__()

        self.config = GPTConfig(indim=indim, vocab_size=vocab_size, grid_thw=grid_thw,
                    embd_pdrop=embd_pdrop, resid_dropout_p=resid_dropout_p, attn_dropout_p=attn_dropout_p, 
                    spatial_n_layer=spatial_n_layer, factorized_n_layer=factorized_n_layer, n_head=n_head, dim=dim, 
                    ffn_dropout_p=ffn_dropout_p, drop_path_rate=drop_path_rate, n_unmasked=n_unmasked, use_pool=use_pool,
                    class_num=class_num, token_drop=token_drop, cls_token_num=cls_token_num, rope_base=rope_base, norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, initializer_range=initalizer_range, multiple_of=multiple_of,
                    max_batch_size=max_batch_size, max_seq_len=max_seq_len, n_kv_head=n_kv_head, factorized_k=factorized_k,use_handcraft=use_handcraft,
                    mrope_section=mrope_section, t_scale_factor=t_scale_factor, target_hw=target_hw,use_cfg=use_cfg, cond_drop_prob=cond_drop_prob)        ## Embedding Layer
        
        self.class_emb = SinusoidalEmbedding(self.config.dim,self.config.class_num) #for class conditional
        if self.config.use_cfg:
            self.null_emb = nn.Parameter(torch.randn(1, self.config.dim) * 0.02) 
            self.cond_drop_prob = self.config.cond_drop_prob
        else:
            self.cond_drop_prob = 0.0

        if self.config.use_handcraft:
            self.handcraft_emb = GatedConditionEmbedding(input_dim=8, dim=self.config.dim)
        
        self.token_drop = nn.Dropout(self.config.token_drop)
        spatial_dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, self.config.spatial_n_layer)]

        # transformer
        self.spatial_blocks = nn.ModuleList()
        for idx in range(self.config.spatial_n_layer):
            self.spatial_blocks.append(Block(self.config, spatial_dpr[idx]))

        # output layer
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        
        self.project_in = nn.Linear(self.config.indim, self.config.dim, bias=False)
        # Simple output head without token factorization
        self.head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

        # video rotary pos embedding
        self.video_rotary_emb = Qwen2VLRotaryEmbedding(dim=self.config.dim // self.config.n_head, base=self.config.rope_base)
        
        # Pre-compute position indices and cos/sin for the given grid
        t, h, w = self.config.grid_thw
        self.cond_token_num = 1 if not self.config.use_handcraft else 2
        seq_len = t * h * w + self.cond_token_num  # +1 for condition token
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.position_ids = get_t_scale_rope_index(seq_len, 
                                                   self.config.grid_thw, 
                                                   scale_factor=self.config.t_scale_factor,
                                                   condition_token_num=self.cond_token_num)  # [3, 1, L]
        
        # Pre-compute cos and sin
        dummy_x = torch.randn(1, seq_len, self.config.dim // self.config.n_head)
        cos, sin = self.video_rotary_emb(dummy_x, self.position_ids)
        # Register as buffers so they move with the model
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initalize_weights() ## initalize the weight

    def initalize_weights(self):
        ## initalize the weight of linear and embedding
        self.apply(self._init_weights)
        
        ### Zero-out output layer
        nn.init.constant_(self.head.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.spatial_blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
    
    def forward(
        self, feature, idx_cls=None,input_pos=None,mask=None, targets=None,cond_drop_prob=0, **kwargs, 
    ):
  
        token_embeddings = self.project_in(feature)
        cls_token_embeddings = self.class_emb(idx_cls) 
        if self.config.use_cfg and cond_drop_prob > 0:
            batch_size = cls_token_embeddings.shape[0]
            cond_drop = torch.rand(batch_size, device=cls_token_embeddings.device) < cond_drop_prob
            cond_drop = cond_drop.unsqueeze(-1)  # [B, 1]
            null_emb_expanded = self.null_emb.expand(batch_size, -1)  # [B, dim]
            cls_token_embeddings = torch.where(cond_drop, null_emb_expanded, cls_token_embeddings)

        cls_token_embeddings = rearrange(cls_token_embeddings, 'b d -> b 1 d')
        if self.config.use_handcraft:
            handcraft_features = kwargs.get('handcraft_f', None)
            if handcraft_features is None:
                raise ValueError("handcraft_features must be provided when use_handcraft is True")
            handcraft_emb = self.handcraft_emb(handcraft_features)  # [B, 1, dim]
            cls_token_embeddings = torch.concat([cls_token_embeddings, handcraft_emb], dim=1)  # [B, 2, dim]

        token_embeddings = torch.concat([cls_token_embeddings, token_embeddings], dim=1)
        h = self.token_drop(token_embeddings)
        
        # use pre-computed cos and sin (already on correct device as buffers)
   
        cos, sin = self.cos, self.sin
        for block in self.spatial_blocks:
            h = block(h, cos, sin, input_pos, mask) #[B N C] 

        if self.config.use_pool:
            cls_f = h[:, :self.cond_token_num, :]  
            h = h[:, self.cond_token_num:, :]  
            h = rearrange(h, 'b (t h w) c -> (b t) c h w', t=self.config.grid_thw[0], h=self.config.grid_thw[1], w=self.config.grid_thw[2])
            target_h, target_w = self.config.target_hw
            h = F.adaptive_avg_pool2d(h, (target_h, target_w))
            h = rearrange(h, '(b t) c h w -> b (t h w) c', b=feature.shape[0])
            h = torch.cat([cls_f, h], dim=1)

      
        h = self.norm(h)

        logits = self.head(h)
        return logits[:,self.cond_token_num:]
    

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        
        return null_logits + (logits - null_logits) * cond_scale



