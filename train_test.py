import numpy as np
import os
from src.module.transformer_ele import Transformer, Transformer_ele
from src.data import ElectricalDataSet, DataLoader, load_txt
import torch.nn as nn
from torch import optim
import torch
import matplotlib.pyplot as plt

from config import config
n_layers = config.n_layers
num_heads = config.num_heads
# embed_dim_src = config.embed_dim_src
# embed_dim_tar = config.embed_dim_tar
embed_dim_src = 5100
embed_dim_tar = 1400
max_seq_len_src = config.max_seq_len_src
max_seq_len_tar = config.max_seq_len_tar
positive_index = config.positive_index
dropout=config.dropout
data_dict = config.data_dict
batch_size = config.batch_size
ckpt = config.ckpt
device = config.device
device = torch.device("cuda:{}".format(config.device)) if config.device in [0,1,2,3,4,5,6,7] else torch.device('cpu')

model = Transformer_ele(n_layers, embed_dim_src, embed_dim_tar, num_heads,
                 max_seq_len_src, max_seq_len_tar, dropout, positive_index).to(device)
if ckpt is not None:
    model.load_state_dict(torch.load(ckpt,map_location=device)) # 设置路径就进行加载
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


def main():
    try:
        train()
    except KeyboardInterrupt:
        torch.save(model.state_dict(),'./Interrupt_model.ckpt')


    # do_predict()


def numpy_txt(array, name):
    np.savetxt(array, name)


def train():
    # source_file, decoder_input, decoder_output
    for epoch in range(500):
        for i in range(20):
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            # print(enc_inputs, dec_inputs, dec_outputs)
            # import sys
            # sys.exit(0)

            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # print(enc_inputs.shape, dec_inputs.shape, dec_outputs.shape) # [32, 3, 2000] ,[32, 5, 150]
            # print(enc_inputs.dtype, dec_inputs.dtype, dec_outputs.dtype)
            enc_inputs = torch.randn(size=(batch_size, max_seq_len_src, embed_dim_src), device=device)
            dec_inputs = torch.randn(size=(batch_size, max_seq_len_tar, embed_dim_tar), device=device)
            dec_outputs = torch.randn(size=(batch_size, max_seq_len_tar, embed_dim_tar), device=device)
            # enc_inputs, dec_inputs, dec_outputs
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # print(outputs.shape, dec_outputs.shape) # [32, 5, 150]
            loss = criterion(outputs, dec_outputs)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
