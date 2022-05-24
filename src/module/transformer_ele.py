import torch
import torch.nn as nn
from src.module.mask import get_attn_pad_mask, get_attn_subsequence_mask
from src.module.embed import nn_embedder, PositionalEncoder

class MultiHeadAttention(nn.Module):
    """
    qkv
    attention
    dropout
    z
    """
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None,attention_dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.q = nn.Linear(self.embed_dim, self.all_head_dim,
                             bias=False if qkv_bias==False else None)
        self.k = nn.Linear(self.embed_dim, self.all_head_dim,
                             bias=False if qkv_bias == False else None)
        self.v = nn.Linear(self.embed_dim, self.all_head_dim,
                             bias=False if qkv_bias == False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(self.all_head_dim, self.embed_dim)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.reshape(new_shape) # [b, num_patches, num_heads, head_dim]
        x = x.transpose(1, 2) # [b, num_heads, num_patches, head_dim]
        return x

    def forward(self, input_Q, input_K, input_V, mask=None): # qkv
        # input_Q [b, s, e]
        q = self.q(input_Q) # [b, seq_len, num_heads * head_dim] # num_patches = seq_len
        k = self.k(input_K)
        v = self.v(input_V)
        # print(q.shape, k.shape, v.shape)
        q, k, v = map(self.transpose_multihead, [q, k, v]) # [b, num_heads, seq_len, head_dim]
        attention = torch.matmul(q, k.transpose(2, 3)) # [b, num_heads, num_patches, num_patches]
        attention = attention * self.scale
        # print(attention.shape, mask.shape)
        # maks fill
        if mask is not None:
            attention.masked_fill_(mask, -1e9)
        #
        attention = self.softmax(attention)
        # dropout
        attention = self.dropout(attention)
        # print(attn.shape)
        z = torch.matmul(attention, v) # [b, num_heads, num_patches, head_dim]
        z = z.transpose(1, 2)

        z = self.proj(z.flatten(2))
        # dropout
        return z, attention

class FeedForward(nn.Module):
    """
    feed forward transformer
    """
    def __init__(self, embed_dim, mlp_ratio = 1, dropout=0,act='relu'):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(nn.Linear(embed_dim, int(embed_dim*mlp_ratio)),
                                 nn.ReLU(inplace=True) if act=='relu' else nn.GELU(),
                                 nn.Dropout(dropout) if act=='relu' else nn.Identity(),
                                 nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
    def forward(self, x):

        return self.ffn(x)

class EncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads, qkv_bias = False, qk_scale = None,
                 attention_dropout = 0., dropout=0, mlp_ratio = 4,
                 act='relu', norm_pos='pre'):
        super(EncoderLayer, self).__init__()
        self.pre_norm = True if norm_pos=='pre' else False
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.self_att = MultiHeadAttention(embed_dim, num_heads,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attention_dropout=attention_dropout)
        self.ffn = FeedForward(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout,act=act)

    def forward(self, enc_inputs):
        shortcut = enc_inputs

        if self.pre_norm:
            enc_inputs = self.norm1(enc_inputs)
        z, attention = self.self_att(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = shortcut + z
        if not self.pre_norm:
            enc_outputs = self.norm1(enc_outputs)

        shortcut = enc_outputs
        if self.pre_norm:
            enc_outputs = self.norm2(enc_outputs)
        enc_outputs = shortcut + self.ffn(enc_outputs)
        if not self.pre_norm:
            enc_outputs = self.norm2(enc_outputs)

        return enc_outputs, attention

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim_tar, embed_dim_src, num_heads, qkv_bias = False, qk_scale = None,
                 attention_dropout = 0., dropout=0, mlp_ratio = 4,
                 act='relu', norm_pos='pre'):
        super(DecoderLayer, self).__init__()
        self.pre_norm = True if norm_pos == 'pre' else False
        self.norm1 = nn.LayerNorm(embed_dim_tar)
        self.norm2 = nn.LayerNorm(embed_dim_tar)
        self.norm3 = nn.LayerNorm(embed_dim_tar)
        self.src_tar_dim = nn.Linear(embed_dim_src, embed_dim_tar)

        self.dec_self_attn = MultiHeadAttention(embed_dim_tar, num_heads,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attention_dropout=attention_dropout)
        self.dec_enc_attn = MultiHeadAttention(embed_dim_tar, num_heads,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attention_dropout=attention_dropout)
        self.ffn = FeedForward(embed_dim_tar, mlp_ratio=mlp_ratio, dropout=dropout, act=act)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''

        shortcut = dec_inputs
        if self.pre_norm:
            dec_inputs = self.norm1(dec_inputs)
        # print(dec_self_attn_mask.shape)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = shortcut + dec_outputs
        if not self.pre_norm:
            dec_outputs = self.norm1(dec_outputs)


        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]

        shortcut = dec_outputs
        if self.pre_norm:
            dec_outputs = self.norm2(dec_outputs)
        enc_outputs = self.src_tar_dim(enc_outputs)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = shortcut + dec_outputs
        if not self.pre_norm:
            dec_outputs = self.norm2(dec_outputs)

        shortcut = dec_outputs
        if self.pre_norm:
            dec_outputs = self.norm3(dec_outputs)
        dec_outputs = self.ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = shortcut + dec_outputs
        if not self.pre_norm:
            dec_outputs = self.norm3(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn



class Encoder(nn.Module):
    def __init__(self, n_layers, embed_dim, num_heads, max_seq_len, dropout=0):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoder(embed_dim, max_seq_len=max_seq_len, dropout= dropout )
        self.layers = nn.ModuleList([EncoderLayer(embed_dim,num_heads,dropout=dropout,attention_dropout=dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self, n_layers, embed_dim_tar, embed_dim_src, num_heads, max_seq_len, dropout=0):

        super(Decoder, self).__init__()

        self.pos_emb = PositionalEncoder(embed_dim_tar, max_seq_len=max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim_tar, embed_dim_src, num_heads,dropout=dropout,attention_dropout=dropout)
             for _ in range(n_layers)])

    def forward(self, dec_inputs,  enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''

        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt(dec_self_attn_subsequence_mask,0).cuda()

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, n_layers, embed_dim_src, embed_dim_tar, num_heads,
                 max_seq_len_src, max_seq_len_tar, dropout):
        super().__init__()
        # n_layers, embed_dim, num_heads, max_seq_len, dropout=0
        self.encoder = Encoder(n_layers, embed_dim_src, num_heads, max_seq_len_src, dropout=dropout)

        self.decoder = Decoder(n_layers, embed_dim_tar, embed_dim_src, num_heads, max_seq_len_tar, dropout=dropout)

        self.projection = nn.Linear(embed_dim_tar, embed_dim_tar)
    def forward(self, enc_inputs, dec_inputs): # , src_mask, trg_mask

        enc_outputs, enc_self_attns = self.encoder(enc_inputs) # , src_mask
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs) # , src_mask, trg_mask

        dec_outputs = self.projection(dec_outputs) # [batch_size, max_seq_len_tar, vocab_size_tar]

        return dec_outputs, enc_self_attns, dec_self_attns, dec_enc_attns

if __name__ == '__main__':
    # n_layers, embed_dim_src, embed_dim_tar, num_heads,
    #                  max_seq_len_src, max_seq_len_tar, dropout
    model = Transformer(n_layers=6, embed_dim_src=2000, embed_dim_tar=150, num_heads=4,
                 max_seq_len_src=3, max_seq_len_tar=5, dropout=0.1).cuda()
    data1 = torch.randn(size=[2, 3, 2000]).cuda()
    data2 = torch.randn(size=[2, 5, 150]).cuda()
    data3 = torch.randn(size=[2, 5, 150]).cuda()
    out = model(data1, data2)
    criterion = nn.L1Loss()
    from torch import optim
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    for i in range(100):
        out, enc_self_attns, dec_self_attns, dec_enc_attns = model(data1, data2)
        loss = criterion(out, data3)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
