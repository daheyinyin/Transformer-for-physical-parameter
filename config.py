from addict import Dict

config = Dict(n_layers = 6,
              num_heads = 8,
              embed_dim_src = 2000,
              embed_dim_tar = 150,
              max_seq_len_src = 3,
              max_seq_len_tar = 4+1,
              positive_index = [1-1, 3-1],
              dropout=0.1,
              data_dict = dict(source_mean=[200.0, 166.675, 1086.1372078947368],source_std=[1,128.41554355980094, 2064.9014864208366],
                 target_mean=[56.605398270569715, 36.603037880718155, 156.93791664264134, -133.1116516062647],
                 target_std=[141.5582450571235, 13.412551904219523, 282.0334557891527, 7.86867042633041]),
              batch_size = 32,
              ckpt = None,
              device = 0 #'CPU'
)

