import pandas as pd
import numpy as np

def load_txt(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    return df.to_numpy()


if __name__ == '__main__':
    path = r"E:\liaozy\pycharm\DeepLearnling\transformer\data\train\source\1.txt"
    data = pd.read_csv(path,header=None,sep='\t')
    print(data)