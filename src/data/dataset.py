import torch
from torch.utils.data import Dataset, DataLoader
import os
from functools import partial
import pandas as pd
import numpy as np

def load_txt(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    return df.to_numpy(dtype='float32')#


class Data_transformer():
    def __init__(self, data_dict):
        self.source_mean = data_dict['source_mean']
        self.source_std = data_dict['source_std']
        self.target_mean = data_dict['target_mean']
        self.target_std = data_dict['target_std']
        self.source_mean = np.array(self.source_mean, dtype='float32')
        self.source_std = np.array(self.source_std, dtype='float32')
        self.target_mean = np.array(self.target_mean, dtype='float32')
        self.target_std = np.array(self.target_std, dtype='float32')
        self.source_mean = np.expand_dims(self.source_mean,axis=1)
        self.source_std = np.expand_dims(self.source_std,axis=1)
        self.target_mean = np.expand_dims(self.target_mean,axis=1)
        self.target_std = np.expand_dims(self.target_std,axis=1)

    def __call__(self, source_file, target_file):
        source_file = (source_file - self.source_mean)/self.source_std
        target_file = (target_file - self.target_mean) / self.target_std
        return source_file, target_file

    def source_trans(self, source_file):
        source_file = (source_file - self.source_mean) / self.source_std
        return source_file

    def target_inv_trans(self, encoder_out):
        encoder_out = encoder_out * self.target_std + self.target_mean
        return encoder_out

# 自定义数据集函数
class MyDataSet(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs,
                 vocab_size_src, vocab_size_tar, max_seq_len_src, max_seq_len_tar):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tar = vocab_size_tar
        self.max_seq_len_src = max_seq_len_src
        self.max_seq_len_tar = max_seq_len_tar

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

class ElectricalDataSet(Dataset):
    def __init__(self, root, mode='train',
                 embed_dim_src = 2000, embed_dim_tar = 150, max_seq_len_src = 3, max_seq_len_tar = 4,
                 data_dict=None):
        super(ElectricalDataSet, self).__init__()
        self.mode = mode
        self.data_path = os.path.join(root, mode)
        self.source_path = os.path.join(root, mode, 'source')
        self.target_path = os.path.join(root, mode, 'target')
        self.file_list = os.listdir(self.source_path)
        self.file_num = len(self.file_list)
        self.load_tool = partial(load_txt, header=None, sep=',')#
        self.data_dict = data_dict
        self.data_trans = Data_transformer(self.data_dict)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        source_file_path = os.path.join(self.source_path, self.file_list[idx])
        source_file = self.load_tool(source_file_path).transpose()

        if self.mode in ['train', 'eval']:
            target_file_path = os.path.join(self.target_path, self.file_list[idx])
            target_file = self.load_tool(target_file_path).transpose()
            # transformer
            if self.data_trans is not None:
                source_file, target_file = self.data_trans(source_file, target_file)

            # S, E
            if self.mode == 'train':
                S = np.zeros((1, target_file.shape[1]), dtype='float32')
                decoder_input = np.concatenate([S, target_file])

            E = np.ones((1, target_file.shape[1]), dtype='float32')
            decoder_output = np.concatenate([target_file, E])
        else:
            source_file = self.data_trans.source_trans(source_file)

        if self.mode == 'train':
            return source_file, decoder_input, decoder_output
        elif self.mode == 'eval':
            return source_file, decoder_output
        elif self.mode == 'infer':
            return source_file

if __name__ == '__main__':
    # from src.data.demo_data import sentences,make_data
    # enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    # dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs)
    # loader = DataLoader(dataset, 1, True)
    #
    # for data in loader:
    #     print(data)

    # source_file = (source_file - data_dict['source_mean']) / data_dict['source_std']
    # target_file = (target_file - data_dict['target_mean']) / data_dict['target_std']
    # r"/home/em/weiyangliao/transformer1/data"
    dataset = ElectricalDataSet(root=r"../../data",mode='train',
                                data_dict = dict(source_mean=[200, 162.5, 110.0125], source_std=[1, 96.01432, 100.73362],
                     target_mean=[77.022865, 45.173065, 103.761536, -134.91313],
                     target_std=[26.337313, 8.723892, 28.81657, 2.0859916])

                                )

    print(len(dataset[0]), dataset[0][0].shape, type(dataset[0][0] ))
    # for data in dataset[0]:
    #     print(data.shape,data.dtype)
    #     print(data.mean(axis=1), data.std(axis=1))

    # path = r"E:\liaozy\pycharm\DeepLearnling\transformer\data\train\target\1.txt"
    # da = load_txt("/home/em/weiyangliao/transformer1/data/train/source/1.txt")
    # print(da)