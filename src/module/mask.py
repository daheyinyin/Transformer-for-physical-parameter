import torch
import numpy as np


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    # print('musk', seq_q.shape, seq_k.shape, pad_attn_mask.shape)
    return pad_attn_mask.expand(batch_size, len_q, len_k).unsqueeze(1)  # [batch_size,head, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.unsqueeze(1)  # [batch_size,head, tgt_len, tgt_len]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    get_attn_pad_mask()