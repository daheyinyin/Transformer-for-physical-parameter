import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class nn_embedder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        # print(position.shape, div_term.shape,(position * div_term).shape)
        # print(pe.shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('x.shape',x.shape,)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    batch, seq_len, embed_size = 2, 3, 2000
    input = torch.zeros([batch, seq_len, embed_size])
    pos_emb = PositionalEncoder(embed_size, max_seq_len=seq_len)

    out = pos_emb(input.transpose(0, 1)).transpose(0, 1)
    print(out.shape)