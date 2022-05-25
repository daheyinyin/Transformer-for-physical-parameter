import numpy as np
import os
from src.module.transformer_ele import Transformer
from src.data import ElectricalDataSet, DataLoader, load_txt
import torch.nn as nn
from torch import optim
import torch
import matplotlib.pyplot as plt
n_layers = 6
num_heads = 8
embed_dim_src = 2000
embed_dim_tar = 150
max_seq_len_src = 3
max_seq_len_tar = 4+1
dropout=0.1
data_dict = dict(source_mean=[200.0, 166.675, 1086.1372078947368],source_std=[1,128.41554355980094, 2064.9014864208366],
                 target_mean=[56.605398270569715, 36.603037880718155, 156.93791664264134, -133.1116516062647],
                 target_std=[141.5582450571235, 13.412551904219523, 282.0334557891527, 7.86867042633041])
batch_size = 32
ckpt = None
# ckpt = r"/home/em/weiyangliao/transformer1/model.ckpt"
# ckpt = r"/home/em/weiyangliao/transformer1/Interrupt_model.ckpt"

dataset = ElectricalDataSet(root=r"./data",mode='train',
                            max_seq_len_src = max_seq_len_src, max_seq_len_tar = max_seq_len_tar,
                            embed_dim_src=embed_dim_src, embed_dim_tar = embed_dim_tar,
                            data_dict=data_dict)
loader = DataLoader(dataset, batch_size, True)



model = Transformer(n_layers, embed_dim_src, embed_dim_tar, num_heads,
                 max_seq_len_src, max_seq_len_tar, dropout).cuda()
if ckpt is not None:
    model.load_state_dict(torch.load(ckpt)) # 设置路径就进行加载
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


def main():
    # try:
    #     train()
    # except KeyboardInterrupt:
    #     torch.save(model.state_dict(),'./Interrupt_model.ckpt')


    do_predict()


def do_predict():
    # read data
    path = r'./data/train/source/101.txt'
    file_name = os.path.basename(path)
    save_dir = r'./data/predict'
    save_path = os.path.join(save_dir, file_name)
    input = load_txt(path=path, header=None, sep=',').transpose()
    print(input.shape)

    # data_trans
    data_trans = dataset.data_trans
    enc_input = data_trans.source_trans(input)
    enc_input = torch.from_numpy(enc_input)
    enc_input = enc_input.unsqueeze(dim=0).cuda()
    # predict
    # print(enc_input.shape)
    output = predict(enc_input).detach().cpu().squeeze()
    output = output[:max_seq_len_tar - 1]
    # decoder_out trans
    output = data_trans.target_inv_trans(output)
    output = output.numpy()
    print(output.shape)  # (4, 150)
    # plot_out(output[0])
    # plot_input_out(input[0], output[0])
    numpy_txt(output.T, save_path)

def do_batch_predict():
    # save_dir
    save_dir = r'./data/predict'
    # dataset
    dataset = ElectricalDataSet(root=r"./data", mode='infer',
                                max_seq_len_src=max_seq_len_src, max_seq_len_tar=max_seq_len_tar,
                                embed_dim_src=embed_dim_src, embed_dim_tar=embed_dim_tar,
                                data_dict=data_dict)
    loader_eval = DataLoader(dataset, batch_size, True)
    data_trans = dataset.data_trans # data_trans

    for enc_inputs, source_file_path in loader_eval:
        enc_inputs = enc_inputs.cuda()
        dec_outputs = predict(enc_inputs).detach().cpu()
        dec_outputs = dec_outputs[:,: max_seq_len_tar - 1]

        outputs = dec_outputs.numpy()
        outputs = outputs[:, max_seq_len_tar - 1]

        # decoder_out trans
        outputs = data_trans.target_inv_trans(outputs)

        # save
        file_names = [os.path.basename(path) for path in source_file_path]
        save_paths = [os.path.join(save_dir, name) for name in file_names]
        [numpy_txt(output.T, save_path) for output, save_path in zip(outputs, save_paths)]


def numpy_txt(array, name):
    np.savetxt(array, name)


def plot_out(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    plt.figure()
    plt.plot(data)

def plot_input_out(input, output):
    plt.figure()
    p1 = plt.plot(input, label='input')
    p2 = plt.plot(output, label='output')
    plt.legend(handles=[p1, p2])


def train():
    # source_file, decoder_input, decoder_output
    for epoch in range(500):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            # print(enc_inputs, dec_inputs, dec_outputs)
            # import sys
            # sys.exit(0)

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # print(enc_inputs.shape, dec_inputs.shape, dec_outputs.shape) # [32, 3, 2000]
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # print(outputs.shape, dec_outputs.shape) # [32, 5, 150]
            loss = criterion(outputs, dec_outputs)
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save
    torch.save(model.state_dict(),'./model.ckpt')


def predict(enc_input):

    enc_outputs, enc_self_attns = model.encoder(enc_input)

    dec_input = torch.zeros(size=(enc_input.shape[0], 1, embed_dim_tar)).type_as(enc_input.data)
    # print('enc_outputs',enc_outputs.shape) # [1, 3, 2000]
    # print('dec_input',dec_input.shape) # [1, 1, 150]
    for i in range(max_seq_len_tar-1):
        dec_outputs, _, _ = model.decoder(dec_input, enc_outputs)
        dec_outputs = model.projection(dec_outputs)
        # print('dec_outputs',dec_outputs.shape) # [1, 5, 150]
        next_data = dec_outputs[:,i].unsqueeze(dim=1)
        print('dec_input:', dec_input.shape, 'next_data:', next_data.shape)
        dec_input = torch.cat([dec_input, next_data],dim=1) # ,dtype=enc_input.dtype, 1
        # if i < max_seq_len_tar:
        #     dec_input = torch.cat([dec_input, torch.zeros(size=(enc_input.shape[0], max_seq_len_tar-i , embed_dim_tar)).type_as(enc_input.data)],dim=1)
    return dec_input


if __name__ == '__main__':
    main()
    plt.show()