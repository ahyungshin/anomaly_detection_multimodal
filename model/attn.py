import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets


class DoubleAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(DoubleAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.relative_attention_bias = nn.Embedding(32, 8)

    def compute_bias(self, queries, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=queries.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=queries.device
        )[None, :]

        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = _relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        position_bias = self.compute_bias(queries, query_length=L, key_length=L)
        attn += position_bias



        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        return V.contiguous()

class DoubleAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(DoubleAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.head_projection = nn.Linear(n_heads * 2, n_heads)

        self.n_heads = n_heads

    def forward(self, x_s, x_f, attn_mask):
        B, L, _ = x_s.shape
        _, S, _ = x_s.shape
        H = self.n_heads

        # SFF
        queries = x_s.view(B, L, H, -1)
        keys = x_f.view(B, S, H, -1)
        values = x_f.view(B, S, H, -1)
        out1 = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        # FSS
        queries2 = x_f.view(B, L, H, -1)
        keys2 = x_s.view(B, S, H, -1)
        values2 = x_s.view(B, S, H, -1)
        out2 = self.inner_attention(
            queries2,
            keys2,
            values2,
            attn_mask,
        )
        out = torch.cat((out1, out2), dim=2) # [B,L,2H,E]
        out = self.head_projection(out.permute(0,1,3,2)).permute(0,1,3,2)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size+2 # 2: CLSs
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

        self.relative_attention_bias = nn.Embedding(32, 8)

    def compute_bias(self, queries, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=queries.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=queries.device
        )[None, :]

        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = _relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, queries, keys, values, sigma, attn_mask, use_prior=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores


        position_bias = self.compute_bias(queries, query_length=L, key_length=L)
        attn += position_bias

        if use_prior is True:
            sigma = sigma.transpose(1, 2)  # B L H ->  B H L
            window_size = attn.shape[-1]
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1
            sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
            prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        else:
            sigma=None
            prior=None

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, use_prior=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        
        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask,
            use_prior
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), series, prior, sigma

