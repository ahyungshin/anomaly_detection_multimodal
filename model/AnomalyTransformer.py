import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer, DoubleAttention, DoubleAttentionLayer


class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_d, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_d = attention_d
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        d_keys = d_model // n_heads
        self.d_model = d_model

        self.s_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.f_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.sf_projection = nn.Linear(d_model,
                                          d_keys * n_heads)

    def forward(self, x_s, x_f, x_sf, attn_mask=None):

        # # QKV linear
        x_s = self.s_projection(x_s)
        x_f = self.f_projection(x_f)
        x_sf = self.sf_projection(x_sf)

        # self-attn (s, s, s)
        new_xs, attn, mask, sigma = self.attention(
            x_s, x_s, x_s,
            attn_mask=attn_mask,
            use_prior=True
        )
        new_xs = new_xs[:,2:,:]
        x1 = x_s[:,2:,:] + self.dropout(new_xs)
        y1 = x1 = self.norm1(x1)
        y1 = self.dropout(self.activation(self.conv1(y1.transpose(-1, 1))))
        y1 = self.dropout(self.conv2(y1).transpose(-1, 1))
        y1 = self.norm2(x1 + y1)

        # self-attn (f,f,f)
        new_xf, attn_f, mask_f, sigma_f = self.attention(
            x_f, x_f, x_f,
            attn_mask=attn_mask,
            use_prior=True
        )
        new_xf = new_xf[:,2:,:]
        x2 = x_f[:,2:,:] + self.dropout(new_xf)
        y2 = x2 = self.norm1(x2)
        y2 = self.dropout(self.activation(self.conv1(y2.transpose(-1, 1))))
        y2 = self.dropout(self.conv2(y2).transpose(-1, 1))
        y2 = self.norm2(x2 + y2)

        # ===================================

        # double cross-attn (s x f)
        new_x = self.attention_d(
            x_s, x_f,
            attn_mask=attn_mask
        )
        new_x = new_x[:,2:,:]
        x3 = x_sf[:,2:,:] + self.dropout(new_x)
        y3 = x3 = self.norm1(x3)
        y3 = self.dropout(self.activation(self.conv1(y3.transpose(-1, 1))))
        y3 = self.dropout(self.conv2(y3).transpose(-1, 1))
        y3 = self.norm2(x3 + y3)

        return y1, y2, y3, [attn[:,:,2:,2:], attn_f[:,:,2:,2:]], [mask[:,:,2:,2:], mask_f[:,:,2:,2:]], [sigma[:,:,2:,2:], sigma_f[:,:,2:,2:]]


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def token_comb(self, s, f, sf): # [B,L,D]
        cls_s = torch.mean(s, dim=1, keepdim=True) # [B,1,D]
        feat_s = s                                 # [B,L,D]
        cls_f = torch.mean(f, dim=1, keepdim=True)
        feat_f = f
        cls_sf =torch.mean(sf, dim=1, keepdim=True)
        feat_sf = sf

        # calculate similarity
        sim_s_to_sf = torch.bmm(cls_s, feat_sf.transpose(-2,-1))  # [B,1,L]
        sim_f_to_sf = torch.bmm(cls_f, feat_sf.transpose(-2,-1))  
        sim_s_to_s = torch.bmm(cls_s, feat_s.transpose(-2,-1))  
        sim_f_to_f = torch.bmm(cls_f, feat_f.transpose(-2,-1))  
        
        # select top 70%
        ntoken = s.shape[1] # except cls token
        top_k = int(ntoken*0.1) 
        left_k = ntoken - top_k

        # token combination
        idx_s_to_sf = torch.topk(sim_s_to_sf, k=top_k, dim=2)[1].permute(0,2,1).repeat(1,1,s.shape[-1]) #[B,30,D]
        s_to_sf = torch.gather(feat_sf, 1, idx_s_to_sf)
        idx_s_to_s = torch.topk(sim_s_to_s, k=left_k, dim=2)[1].permute(0,2,1).repeat(1,1,s.shape[-1])
        s_to_s = torch.gather(feat_s, 1, idx_s_to_s)
        fused_s = torch.cat((s_to_s, s_to_sf), dim=1)

        idx_f_to_sf = torch.topk(sim_f_to_sf, k=top_k, dim=2)[1].permute(0,2,1).repeat(1,1,s.shape[-1])
        f_to_sf = torch.gather(feat_sf, 1, idx_f_to_sf)
        idx_f_to_f = torch.topk(sim_f_to_f, k=left_k, dim=2)[1].permute(0,2,1).repeat(1,1,s.shape[-1])
        f_to_f = torch.gather(feat_f, 1, idx_f_to_f)
        fused_f = torch.cat(( f_to_f, f_to_sf), dim=1)

        return fused_s, fused_f

    def token_prior(self, x_s, x_f, x_sf):
        cls_s = torch.mean(x_s, dim=1, keepdim=True) # [B,1,D]
        cls_f = torch.mean(x_f, dim=1, keepdim=True)
        cls_sf =torch.mean(x_sf, dim=1, keepdim=True)

        x_s = torch.cat((cls_f, cls_sf, x_s), dim=1)
        x_f = torch.cat((cls_s, cls_sf, x_f), dim=1)
        x_sf = torch.cat((cls_s, cls_f, x_sf), dim=1)

        return x_s, x_f, x_sf


    def forward(self, x_s, x_f, attn_mask=None):
        # [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for layer_i, attn_layer in enumerate(self.attn_layers):
            if layer_i==0:
                x_sf = x_s

            x_s, x_f, x_sf = self.token_prior(x_s, x_f, x_sf)
            x_s, x_f, x_sf, series, prior, sigma = attn_layer(x_s, x_f, x_sf, attn_mask=attn_mask)
            x_s, x_f = self.token_comb(x_s, x_f, x_sf)

            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x_s = self.norm(x_s)
            x_f = self.norm(x_f)
            x_sf = self.norm(x_sf)
        
        x = (x_s + x_f + x_sf) / 3.

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding_s = DataEmbedding(enc_in,  d_model, dropout)
        self.embedding_f = DataEmbedding(1025,  d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    DoubleAttentionLayer(
                        DoubleAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_s, x_f):
        enc_out_s = self.embedding_s(x_s) # [B,L D]
        enc_out_f = self.embedding_f(x_f) # [B,L D]
        enc_out, series, prior, sigmas = self.encoder(enc_out_s, enc_out_f)

        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out # [B, L, C]
