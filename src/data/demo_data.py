import torch

#自制数据集
             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0


src_word2idx = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
src_idx2word = {src_word2idx[key]: key for key in src_word2idx}
src_vocab_size = len(src_idx2word)                 # 字典字的个数


tgt_word2idx = {'S':0, 'E':1, 'P':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}
tgt_idx2word = {tgt_word2idx[key]: key for key in tgt_word2idx}                               # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_word2idx)                                                     # 目标字典尺寸

src_len = len(sentences[0][0].split(" "))                                           # Encoder输入的最大长度 5
tgt_len = len(sentences[0][1].split(" "))                                           # Decoder输入输出最大长度 5
# print(src_len,tgt_len)

# 把sentences 转换成字典索引
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
      enc_input = [[src_word2idx[n] for n in sentences[i][0].split()]] # ids
      dec_input = [[tgt_word2idx[n] for n in sentences[i][1].split()]]
      dec_output = [[tgt_word2idx[n] for n in sentences[i][2].split()]]
      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)



if __name__ == '__main__':
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    print('src_vocab_size, tgt_vocab_size',src_vocab_size, tgt_vocab_size)

    print(enc_inputs)
    print(dec_inputs)
    print(dec_outputs)
