import math
import torch
import copy
from torch import nn

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class PSP(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=[(1, 1), (3, 3), (6, 6), (8, 8)], dimension=2):
        super(PSP, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size[0], size[1]))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size[0], size[1], size[2]))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center
    

class scTransformerLayer(nn.Module):
    def __init__(self,  d_model, nheads, dim_feedforward, dropout, is_ffn=True, qk_cat=True, q_low=False, activation=nn.ReLU(inplace=True), mode='linear'):
        super().__init__()
        self.is_ffn = is_ffn
        self.qk_cat = qk_cat
        self.q_low = q_low
        self.dim = d_model // nheads
        self.nheads = nheads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mode = mode

        assert self.mode in ('linear', 'full', 'conv1'), "scTransformerLayer mode error (set linear, full or conv1)"

        # self-cross attention
        self.sc_attn = LinearAttention() if mode == 'linear' else FullAttention()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = activation

        
        # feed-forward network
        # print('mode', mode)
        if self.is_ffn:
            if mode == 'linear' or 'full':
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, dim_feedforward, bias=False),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model, bias=False)
                )
            elif mode == 'conv1':
                t1 = int(abs((math.log(d_model, 2) + 1) / 2))
                k1_size = t1 if t1 % 2 else t1 + 1

                t2 = int(abs((math.log(dim_feedforward, 2) + 1) / 2))
                k2_size = t2 if t2 % 2 else t2 + 1
                self.mlp = nn.Sequential(
                    nn.Conv1d(d_model, dim_feedforward, kernel_size=k1_size, padding=(k1_size - 1) // 2, bias=False),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Conv1d(dim_feedforward, d_model, kernel_size=k2_size, padding=(k2_size - 1) // 2, bias=False),
                )
            
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        self._reset_parameters()

    
    def _reset_parameters(self):
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)

        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.) 

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)


    def forward(self, q_src, kv_src, q_mask=None, kv_mask=None):
        """
        Args:
            q_src (torch.Tensor): [N, L, C]
            k_src (torch.Tensor): [N, S, C]
            q_mask (torch.Tensor): [N, L] (optional)
            kv_mask (torch.Tensor): [N, S] (optional)
        """
        b = q_src.shape[0]
        query = self.query_proj(q_src).view(b, -1, self.nheads, self.dim) # [N, L, (H, D)]

        if self.qk_cat:
            if not self.q_low:
                key = value = torch.cat([kv_src, q_src], dim=1)
            else:
                key = value = torch.cat([q_src, kv_src], dim=1)
        else:
           key = value = kv_src
        key = self.key_proj(key).view(b, -1, self.nheads, self.dim)
        value = self.value_proj(value).view(b, -1, self.nheads, self.dim)
        
        att = self.sc_attn(query, key, value, q_mask, kv_mask)   # [N, L, (H, D)]
        att = self.merge(att.view(b, -1, self.nheads * self.dim))  # [N, L, C]
 
        att = self.norm1(q_src + self.dropout1(att))

        # ffn
        if self.is_ffn:
            if self.mode == 'conv1':
                att = att.permute(0, 2, 1)
            att = att + self.dropout2(self.mlp(att))
            if self.mode == 'conv1':
                att = att.permute(0, 2, 1)
            att = self.norm2(att)

        return att, kv_src


class scTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, q_src, kv_src, q_mask=None, kv_mask=None):
        for layer_block in self.layers:
            q_src, kv_src = layer_block(q_src, kv_src, q_mask, kv_mask)

        return q_src, kv_src


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

# from LoFTR
class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

# from LoFTR
class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


if __name__ == '__main__':
    # queries, keys, values = None
    # model = LinearAttention()

    sat = torch.randn(7, 3, 256, 256)
    # sat = torch.randn(7, 3, 256, 256)
    grd = torch.randn(7, 3, 122, 671)