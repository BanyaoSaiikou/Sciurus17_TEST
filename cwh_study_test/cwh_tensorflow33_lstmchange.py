# -*- coding:utf-8 -*-




from __future__ import absolute_import, division, print_function, unicode_literals
from operator import irshift
from re import X
from tkinter.ttk import Style
from turtle import shape
# 安装tfds pip install tfds-nightly==1.0.2.dev201904090105
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.layers as layers
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
import os

from visdom import Visdom
from random import randint
##

#导入dataset，区分train和val

dataset_dir = os.path.join(os.path.dirname(__file__),  "/home/cwh/デスクトップ/Tuan/imitation_old/dataset")
train_dataset_file = os.path.join(dataset_dir, "lstm_dataset_train.txt")
test_dataset_file = os.path.join(dataset_dir, "lstm_dataset_test.txt")#wei
val_dataset_file = os.path.join(dataset_dir, "lstm_dataset_test.txt")#wei
with open(train_dataset_file, "rb") as fp:
    samples = pickle.load(fp)
x = []
y = []
z = []
for sample in samples:
    scenario = np.array(sample[0])#目标状态
    input_states = np.array(sample[1])
    actions = np.array(sample[3])#获取scenario，input_states，    ，actions

    scenario_list=scenario.tolist()
    input_states_list=input_states.tolist()
    actions_list=actions.tolist()

    scenario_list1 = [str(x) for x in scenario_list]
    input_states_list1 = [str(x) for x in input_states_list]
    actions_list1 = [str(x) for x in actions_list]

    x.append(input_states_list1)
    z.append(scenario_list1)
    y.append(actions_list1)

# print("!!!!!!!!!!",len(x))
# print("!!!!!!!!!!",len(y))
# print("type",type(y[0][0]))


for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j]=x[i][j]+"|"+z[i][-1]
#！！！！！！！！！！！！！！
# for i in range(len(x)):
#     for j in range(len(x[i])):
#         x[i][j]=x[i][j]

x1 = []
for i in range(len(x)):
    s=""
    for j in range(len(x[i])):
        if j !=(len(x[i])-1):
            s=s+x[i][j]+" "
        if j ==(len(x[i])-1):
            s=s+x[i][j]
    x1.append(s)
    

y1 = []
for i in range(len(y)):
    s=""

    for j in range(len(y[i])):
       

            if j !=(len(y[i])-1):
                s=s+"a"+y[i][j]+" "
            if j ==(len(y[i])-1):
                 s=s+"a"+y[i][j]


    y1.append(s)

###########

with open(test_dataset_file, "rb") as fp2:
    samples2 = pickle.load(fp2)
x2 = []
y2 = []
z2 = []
for sample in samples2:
    scenario = np.array(sample[0])#目标状态
    input_states = np.array(sample[1])
    actions = np.array(sample[3])#获取scenario，input_states，    ，actions

    scenario_list=scenario.tolist()
    input_states_list=input_states.tolist()
    actions_list=actions.tolist()

    scenario_list1 = [str(x) for x in scenario_list]
    input_states_list1 = [str(x) for x in input_states_list]
    actions_list1 = [str(x) for x in actions_list]

    x2.append(input_states_list1)
    z2.append(scenario_list1)
    y2.append(actions_list1)

# print("!!!!!!!!!!x",(x[0]))
# print("!!!!!!!!!!len",len(x[0]))
# print("!!!!!!!!!!type",type(x[0]))
# print("!!!!!!!!!!x",(x[0][0]))
# print("!!!!!!!!!!type",type(x[0][0]))

# !!!!!!!!!!x ['021', '4321', '2121']
# !!!!!!!!!!len 3
# !!!!!!!!!!type <class 'list'>
# !!!!!!!!!!x 021
# !!!!!!!!!!type <class 'str'>

# print("!!!!!!!!!!",len(y))
# print("type",type(y[0][0]))

for i in range(len(x2)):
    for j in range(len(x2[i])):
        x2[i][j]=x2[i][j]+"|"+z2[i][-1]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# for i in range(len(x2)):
#     for j in range(len(x2[i])):
#         x2[i][j]=x2[i][j]

x22 = []
for i in range(len(x2)):
    s2=""
    for j in range(len(x2[i])):
        if j !=(len(x2[i])-1):
            s2=s2+x2[i][j]+" "
        if j ==(len(x2[i])-1):
            s2=s2+x2[i][j]
    x22.append(s2)

y22 = []
for i in range(len(y2)):
    s2=""
    for j in range(len(y2[i])):
        if j !=(len(y2[i])-1):
            s2=s2+"a"+y2[i][j]+" "
        if j ==(len(y2[i])-1):
            s2=s2+"a"+y2[i][j]
    y22.append(s2)



##
##
with open(test_dataset_file, "rb") as fp3:
    samples3 = pickle.load(fp3)
x3 = []
y3 = []
z3 = []
for sample in samples3:
    scenario = np.array(sample[0])#目标状态
    input_states = np.array(sample[1])
    actions = np.array(sample[3])#获取scenario，input_states，    ，actions

    scenario_list=scenario.tolist()
    input_states_list=input_states.tolist()
    actions_list=actions.tolist()

    scenario_list1 = [str(x) for x in scenario_list]
    input_states_list1 = [str(x) for x in input_states_list]
    actions_list1 = [str(x) for x in actions_list]

    x3.append(input_states_list1)
    z3.append(scenario_list1)
    y3.append(actions_list1)


# print("!!!!!!!!!!",len(x))
# print("!!!!!!!!!!",len(y))
# print("type",type(y[0][0]))

for i in range(1000):
    for j in range(len(x3[i])):
        x3[i][j]=x3[i][j]+z3[i][-1]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# for i in range(len(x2)):
#     for j in range(len(x2[i])):
#         x2[i][j]=x2[i][j]

x33 = []
for i in range(1000):
    s3=""
    for j in range(len(x3[i])):
        if j !=(len(x3[i])-1):
            s3=s3+x3[i][j]+" "
        if j ==(len(x3[i])-1):
            s3=s3+x3[i][j]
    x33.append(s3)

y33 = []
for i in range(1000):
    s3=""
    for j in range(len(y3[i])):
        if j !=(len(y3[i])-1):
            s3=s3+y3[i][j]+" "
        if j ==(len(y3[i])-1):
            s3=s3+y3[i][j]
    y33.append(s3)


print("!!!!!!!x1!!!!!!!",len(x1))
print("!!!!!!!!y1!!!!!!",len(y1))
print("!!!!!!!x1!!!!!!!",(x1[0]))
print("!!!!!!!!y1!!!!!!",(y1[0]))


xy = tf.data.Dataset.from_tensor_slices((x1, y1))#合并x1，y1，创建新的数据集
xy22 = tf.data.Dataset.from_tensor_slices((x22, y22))#合并x1，y1，创建新的数据集
xy33 = tf.data.Dataset.from_tensor_slices((x33, y33))#合并x1，y1，创建新的数据集
print(x1[0])
print(x22[0])
print(y1[0])
print(y22[0])
# 05 35 335 345 55 
# 020 1520 1020 1520 1020 1820 2020 
#---
# print(y1[0])#action对应数据
# print(xy)
# print(type(xy))
##
#导入dataset，区分train和val
##
#   3、将文本编码成数字形式

tokenizer_pt = tfds.features.text.Tokenizer()#实例化一个分词器
tokenizer_en = tfds.features.text.Tokenizer()#实例化一个分词器

vocabulary_set_pt = set()
vocabulary_set_en = set()
#============================================无拼接
for pt, en in xy:
    some_tokens_en = tokenizer_en.tokenize(en.numpy())# 可以将一句话分成多个单词
    vocabulary_set_en.update(some_tokens_en)
    some_tokens_pt = tokenizer_pt.tokenize(pt.numpy())# 可以将一句话分成多个单词
    vocabulary_set_pt.update(some_tokens_pt)

#===========================有拼接
# for i in range( len(x1)):

#     some_tokens_pt = tokenizer_pt.tokenize(  bytes(x1[i], encoding = "utf8"))# 可以将一句话分成多个单词
#     vocabulary_set_pt.update(some_tokens_pt)
#     # print("!!!!!!!111",len(vocabulary_set_pt))#!!!!!!!111 1326

#     some_tokens_en = tokenizer_en.tokenize(  bytes(y1[i], encoding = "utf8"))# 可以将一句话分成多个单词
#     vocabulary_set_en.update(some_tokens_en)
#===============================================
#x1 list(str,str)
#x1 bytes : b,"  str str"
#建立编码器
encoder_en = tfds.features.text.TokenTextEncoder(vocabulary_set_en)
encoder_pt = tfds.features.text.TokenTextEncoder(vocabulary_set_pt)


# sample_string =   bytes(x1[-1], encoding = "utf8")
# print ('The sample string: {}'.format(sample_string))

# tokenized_string = encoder_pt.encode(sample_string)
# print ('Tokenized string is {}'.format(tokenized_string))

# original_string = encoder_pt.decode(tokenized_string)
# print ('The original string: {}'.format(original_string))
#-------------------------------
# sample_string =   bytes(y1[-1], encoding = "utf8")
# print ('The sample string: {}'.format(sample_string))

# tokenized_string = encoder_en.encode(sample_string)
# print ('Tokenized string is {}'.format(tokenized_string))

# original_string = encoder_en.decode(tokenized_string)
# print ('The original string: {}'.format(original_string))
#==============> print:
# The sample string: b"astronomers now believe that every star in the galaxy has a planet , and they speculate that up to one fifth of them have an earth-like planet that might be able to harbor life , but we have n't seen any of them ."
# Tokenized string is [5319, 11336, 10262, 13270, 10967, 14283, 6577, 215, 20858, 780, 11675, 631, 14097, 9829, 24792, 13270, 12727, 17133, 21179, 18750, 5674, 12003, 6870, 20855, 6259, 25462, 631, 13270, 14480, 19721, 20610, 17133, 26110, 17347, 15186, 25, 6870, 13132, 12541, 13984, 14641, 5674, 12003]
# The original string: astronomers now believe that every star in the galaxy has a planet and they speculate that up to one fifth of them have an earth like planet that might be able to harbor life but we have n t seen any of them

#每条语句向量的最前面和最后面添加star和end的token 方便模型知道什么时候开始什么时候结束。
def encode(lang1, lang2):

    lang1 = [encoder_pt.vocab_size] + encoder_pt.encode(
          lang1.numpy()) + [encoder_pt.vocab_size+1]
          #.encode( )以 encoding 指定的编码格式编码字符串


    lang2 = [encoder_en.vocab_size] + encoder_en.encode(
          lang2.numpy()) + [encoder_en.vocab_size+1]

    return lang1, lang2



#删除过长的样本
MAX_LENGTH = 40

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                           tf.size(y) <= max_length)

#编码函数
def tf_encode(pt, en):

    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en

#将样本打乱、分批
BUFFER_SIZE = 20000
BATCH_SIZE = 10            #[10,37,128]
#BATCH_SIZE = 64
#shan


train_dataset = xy.map(tf_encode)#在前後加上開始與結束

train_dataset = train_dataset.filter(filter_max_length)#刪除過長樣本
# 将数据集缓存到内存中以加快读取速度。
train_dataset = train_dataset.cache()#使用非序列化的方式将RDD中的数据全部尝试持久化到内存中
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, ((None, ), (None, )))#有一个固定大小的buffer，每次按均匀分布从buffer中取出下一个元素。
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)#将数据生成时间与数据消耗时间分开


test_dataset = xy22.map(tf_encode)
test_dataset = test_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, ((None, ), (None, )))

val_dataset = xy33.map(tf_encode)
val_dataset =val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, ((None, ), (None, )))


def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)

    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]#两行输出内容一样


    return tf.cast(pos_encoding, dtype=tf.float32)#比如读入的图片如果是int8类型,数据格式转换为float32
#填充mask
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    #前瞻mask
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

#生成所有遮挡
def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

#Scaled dot-product attention
#self attention
def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
    q: 请求的形状 == (..., seq_len_q, depth)
    k: 主键的形状 == (..., seq_len_k, depth)
    v: 数值的形状 == (..., seq_len_v, depth_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
    输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
# shape=(None, 8, None, None)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
#shape=(None, 8, None, 16)

    return output, attention_weights

#Multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads=8, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads=8, seq_len_q, seq_len_k=16)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)/(None, None, 8, 16)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)/shape=(None, None, 128)四维变三维

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)/ shape=(None, None, 128)

        return output, attention_weights# shape=(None, None, 128)，shape=(None, 8, None, None),



#Position-wise feed-forward networks（点式前馈网络）
#FFN/FC
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


#编码器层
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)/shape=(None, None, 128)

    attn_output = self.dropout1(attn_output, training=training)
    # print("!!!!!",attn_output)     shape=(None, None, 128)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model) /shape=(None, None, 128)

    return out2
    # out2.shape(64, 14, 128)

#解码器层
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    
    def call(self, x, enc_output, training, 
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        # print("!!!!!1",attn_weights_block1) shape=(None, 8, None, None),
        # print("!!!!!1",attn_weights_block2) shape=(None, 8, None, None),

        return out3, attn_weights_block1, attn_weights_block2


#Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model

        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
       # print("x.shape!!!!!!!!!",x.shape)    (64, 14)
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model) shape=(None, None, 128),

#Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model#128
        self.num_layers = num_layers#4
        # print("!!!!!!!!!!!shan",maximum_position_encoding)  =29

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]# shape=(2,)

        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)#shape=(None, None, 128)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1#shape=(None, 8, None, None)
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2#shape=(None, 8, None, None)
    
        # x.shape == (batch_size, target_seq_len, d_model)
        # print("!!!!!1",x) shape=(None, None, 128)
        # print("!!!!!2",attention_weights) (None, 8, None, None)
        return x, attention_weights

#创建 Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # print("!!!!!1",final_output) shape=(None, None, 29)
        # print("!!!!!2",attention_weights) (None, 8, None, None) 
        return final_output, attention_weights

#配置超参数
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
# #官网超参数
# num_layers = 6
# d_model = 512
# dff = 2048
# num_heads = 8


input_vocab_size = encoder_pt.vocab_size +1291#1338
target_vocab_size = encoder_en.vocab_size+28 #1317
print("!!!!!!!11","input_vocab_size,",input_vocab_size,"target_vocab_size",target_vocab_size)
# input_vocab_size = encoder_pt.vocab_size + 10
# target_vocab_size = encoder_en.vocab_size + 10

max_seq_len = 40
dropout_rate = 0.1
#优化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show() 

#损失函数与指标
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))#在您的批次中，您会有不同长度的句子，并且您会进行0填充以使所有句子长度相等
  loss_ = loss_object(real, pred)
  #但是，将填充部分包含在损失计算中没有意义0- 因此，它首先查看您拥有 a 的索引，0然后使用乘法使它们的损失贡献为 0。
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  #    損失関数にはSparseCategoricalCrossentropyを使用します。
    # 予測した単語確率と正解単語間の損失を求めます。
    # また、損失を計算する際はPADの0を無視するためにマスクします。
  return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')#total 除以 count 。
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')#total 除以 count 。
#训练
#检查点

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./cwh_study_test/checkpoint/train303"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)#checkpoint.restore(tf.train.latest_checkpoint('./save'))  
  #例如，调用 checkpoint.restore('./save/model.ckpt-1') 就可以载入前缀为 model.ckpt ，序号为 1 的文件来恢复模型。
  print ('Latest checkpoint restored!!')

#梯度下降
#teacher-forcing 
# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]#竖折的全部,横着的最后一号位 en
  tar_real = tar[:, 1:]#竖折全部,横着的1号位
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)


  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

def train_step2(inp, tar):
  tar_inp = tar[:, :-1]#竖折的全部,横着的最后一号位 en
  tar_real = tar[:, 1:]#竖折全部,横着的1号位
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)


  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  val_loss(loss)
  val_accuracy(tar_real, predictions)
# print("pt",pt)
# print("vocabulary_set_pt",vocabulary_set_pt)#诸多单词 type set

# print("encoder_pt",encoder_pt)#<TokenTextEncoder vocab_size=37503>
# print("encoder_pt.vocab_size",encoder_pt.vocab_size)



#---------
#--------------------------------
#  instantiate the window class （实例化监听窗口类）
viz = Visdom() 
#  create a window and initialize it （创建监听窗口）

#viz.line([[0.,0.,0.,0.]], [0], win='train', opts=dict(title='train_loss&train_acc&val_loss&val_acc', legend=['train_loss', 'train_acc','val_loss','val_acc']))

#viz.line([[0.,0.]], [0], win='train', opts=dict(title='train_loss&val_loss', legend=['train_loss','val_loss']))
#viz.line([[0.,0.]], [0], win='train', opts=dict(title='val_loss&val_acc', legend=['val_loss','val_acc']))
#-------------------------------
list1 = [0.32323232323232326, 0.43, 0.6122448979591837, 0.62, 0.65,0.65,0.7021063189568706  ,0.687 , 0.686 ,0.696,
0.7067067067067067, 0.696,0.7024048096192385,0.7137137137137137,0.705,0.7218875502008032,0.704,0.7244488977955912,0.7107107107107107,0.7017017017017017,
0.708,0.7097097097097097,0.702,0.7097097097097097,0.716

,0.32323232323232326, 0.43, 0.6122448979591837, 0.62, 0.65,0.65,0.7021063189568706  ,0.687 , 0.686 ,0.696,
0.7067067067067067, 0.696,0.7024048096192385,0.7137137137137137,0.705,0.7218875502008032,0.704,0.7244488977955912,0.7107107107107107,0.7017017017017017,
0.708,0.7097097097097097,0.702,0.7097097097097097,0.716]
list2 = [0.434,0.637,0.682,0.747,0.735,0.762,0.794,0.811,0.803,0.806,
0.837,0.833,0.843, 0.832, 0.805,0.833,0.843,0.851,0.8648648648648649, 0.849

,0.434,0.637,0.682,0.747,0.735,0.762,0.794,0.811,0.803,0.806,
0.837,0.833,0.843, 0.832, 0.805,0.833,0.843,0.851,0.8648648648648649, 0.849]
listval = [ 0.004,0.063,0.155,0.305,0.346, 0.369, 0.429,0.428,0.432,0.45,0.471,0.494,0.494,0.584,0.584,0.581,0.605,0.633,0.648,0.658]
#---------
# #训练
EPOCHS = 20
for epoch in range(EPOCHS):
  start = time.time()
  
#   train_loss.reset_states()
#   train_accuracy.reset_states()#no reset is ok

  # inp -> portuguese, tar -> english
#1  0.004
#0.063
#0.155
#0.305
#5    0.346
#6 0.369
#7 0.429
#0.428
#0.432
#10    0.45
#11 0.471
# 0.494
#0.494
#15    0.584
#16  0.581
#
#
#
#20    0.658
#####val
#   for (batch2, (inp2, tar2)) in enumerate(val_dataset):
#     train_step2(inp2, tar2)
    
#     if batch2 % 50 == 0:
#       print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#           epoch + 1, batch2, val_loss.result(), val_accuracy.result()))

#####
  for (batch, (inp, tar)) in enumerate(train_dataset):
    train_step(inp, tar)

    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))





      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
#   print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
#                                                 train_loss.result(), 
#                                                 train_accuracy.result()))
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))


    #  randomly get loss and acc （设置显示数据）

    #  update window image （传递数据到监听窗口进行画图）
 # viz.line([[train_loss.result(), listval[epoch],val_loss.result(),list2[epoch]]], [epoch], win='train', update='append')
  #   viz.line([[ val_loss.result(),list2[epoch]]], [epoch], win='train', update='append')
  #viz.line([[train_loss.result(), val_loss.result()]], [epoch], win='train', update='append')
    #-----------------------------

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


  #评估函数
def evaluate(inp_sentence):
    start_token = [encoder_pt.vocab_size]#pt开始到结束时候的位置
    end_token = [encoder_pt.vocab_size + 1]

    # 输入语句是葡萄牙语，增加开始和结束标记
    inp_sentence = start_token + encoder_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)#增加一个维度

    # 因为目标是英语，输入 transformer 的第一个词应该是
    # 英语的开始标记。
    decoder_input = [encoder_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
  
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
    
        # 从 seq_len 维度选择最后一个词
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        #返回的是predictions中的最大值的索引号，如果predictions是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量

        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == encoder_en.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights
    
        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)#star +s1+s2+s3+end

    return tf.squeeze(output, axis=0), attention_weights

# #注意力权重图
# def plot_attention_weights(attention, sentence, result, layer):
#     fig = plt.figure(figsize=(16, 8))

#     sentence = tokenizer_pt.encode(sentence)

#     attention = tf.squeeze(attention[layer], axis=0)

#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(2, 4, head+1)

#         # 画出注意力权重
#         ax.matshow(attention[head][:-1, :], cmap='viridis')

#         fontdict = {'fontsize': 10}

#         ax.set_xticks(range(len(sentence)+2))
#         ax.set_yticks(range(len(result)))

#         ax.set_ylim(len(result)-1.5, -0.5)

#         ax.set_xticklabels(
#             ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
#             fontdict=fontdict, rotation=90)

#         ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
#                             if i < tokenizer_en.vocab_size], 
#                            fontdict=fontdict)

#         ax.set_xlabel('Head {}'.format(head+1))
  
#     plt.tight_layout()
#     plt.show()
#翻译函数
def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = encoder_en.decode([i for i in result 
                                            if i < encoder_en.vocab_size])  
    # print("   ")
    # print('Input: {}'.format(sentence))
    # print('Predicted translation: {}'.format(predicted_sentence))

    # if plot:
    #     plot_attention_weights(attention_weights, sentence, result, plot)
    return predicted_sentence


def change11(a):
    change = list(a)
    change.pop()
    str4 = "".join(change)
    return str4




#正常序列
# print("1111111111111")
# print("入力: ","0|4")
# print("予測：",translate("0|4 4|4"))
# print("実際：",change11('a0 a10 a20 a24 '))
# print("----------------------")
# print("入力: ","0|4 1|4 ")
# print("予測：",translate("0|4 1|4 4|4"))
# print("実際：",change11('a0 a10 a20 a24 '))
# print("----------------------")
# print("入力: ","0|4 1|4 9|4 ")
# print("予測：",translate("0|4 1|4 9|4 4|4"))
# print("実際：",change11('a0 a10 a20 a24 '))
# print("----------------------")
# print("入力: ","0|4 1|4 9|4 4|4 ")
# print("予測：",translate("0|4 1|4 9|4 4|4"))
# print("実際：",change11('a0 a10 a20 a24 '))

# #正常序列
# print("222222222222222")
# print("入力: ","0|4")
# print("予測：",translate("0|4 4|4"))
# print("実際：",change11('a0 a10 a1 a20 a24 '))
# print("----------------------")
# print("入力: ","0|4 1|4 ")
# print("予測：",translate("0|4 1|4 4|4"))
# print("実際：",change11('a0 a10 a1 a20 a24'))
# print("----------------------")
# print("入力: ","0|4 1|4 27|4 ")
# print("予測：",translate("0|4 1|4 27|4 4|4"))
# print("実際：",change11('a0 a10 a1 a20 a24 '))
# print("----------------------")
# print("入力: ","0|4 1|4 27|4 9|4 ")
# print("予測：",translate("0|4 1|4 27|4 9|4 4|4"))
# print("実際：",change11('a0 a10 a1 a20 a24'))
# print("----------------------")
# print("入力: ","0|4 1|4 27|4 9|4 4|4")
# print("予測：",translate("0|4 1|4 27|4 9|4 4|4"))
# print("実際：",change11('a0 a10 a1 a20 a24 '))
# print("22222222222222222222")
# print("入力: ","0|20 20|20")
# print("予測：",translate("020 2020"))
# print("実際：",change11('0 10 17 24 '))
# print("----------------------")
# print("入力: ","0|20 29|20 20|20")
# print("予測：",translate("020 2920 2020"))
# print("実際：",change11('0 10 17 24 '))
# print("----------------------")
# print("入力: ","0|20 29|20 24|20 20|20")
# print("予測：",translate("020 2920 2420 2020"))
# print("実際：",change11('0 10 17 24 '))
good_luck_num="0"
while good_luck_num!="q":
 good_luck_num = input('请输入码：')
 print("入力:",good_luck_num)
 a=translate(good_luck_num)
 print( "出力all:",a)
 if a[-6]!="a":
      b1=int(a[-6]+a[-5])
      print( "出力1:",b1)
 if a[-6]=="a":
     b1=int(a[-5])
     print("出力2:",b1)

# print("----------test3------------")
# print("入力: ","0|30 38|30 30|30 2|30 30|30")
# print("予測：",translate("030 3830 3030 230 3030"))
# 入力:  0|30 38|30 30|30 2|30 30|30
# 予測： 8 8 8 5 24


#55555555555
# print("入力: ","0|37 37|37")
# print("予測：",translate("037 3737"))
# print(" ")
# print("入力: ","0|37 34|37 37|37")
# print("予測：",translate("037 3437 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 37|37")
# print("予測：",translate("037 3437 2737 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 37|37")
# print("予測：",translate("037 3437 2737 137 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 27|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 2737 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 27|37 1|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 2737 137 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 27|37 1|37 1|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 2737 137 137 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 27|37 1|37 1|37 10|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 2737 137 137 1037 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 27|37 1|37 1|37 10|37 1|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 2737 137 137 1037 137 3737"))
# print(" ")
# print("入力: ","0|37 34|37 27|37 1|37 34|37 0|37 27|37 1|37 1|37 10|37 1|37 1|37 37|37")
# print("予測：",translate("037 3437 2737 137 3437 037 2737 137 137 1037 137 137 3737"))


# print(" 6666666")
# print("入力: ","0|37")
# print("予測：",translate("0|37 37|37"))
# print(" ")

# print("入力: ","0|37 2|37")
# print("予測：",translate("0|37 2|37 37|37"))
# print(" ")

# print("入力: ","0|37 2|37 34|37")
# print("予測：",translate("0|37 2|37 34|37 37|37"))
# print(" ")

# print("入力: ","0|37 2|37 34|37 30|37 ")
# print("予測：",translate("0|37 2|37 34|37 30|37 37|37"))
# print(" ")

# print("入力: ","0|37 2|37 34|37 30|37 27|37")
# print("予測：",translate("0|37 2|37 34|37 30|37 27|37 37|37"))
# print(" ")

# print("入力: ","0|37 2|37 34|37 30|37 27|37 1|37")
# print("予測：",translate("0|37 2|37 34|37 30|37 27|37 1|37 37|37"))
# print(" ")

# print("入力: ","0|37 2|37 34|37 30|37 27|37 1|37 34|37")
# print("予測：",translate("0|37 2|37 34|37 30|37 27|37 1|37 34|37 37|37"))
# print(" ")




# check_i=0
# check_j=0
# for j in range(1000):
#    try:
#         jj=j+1000
#         print("入力: ",x22[jj])
#         print("data: ")
#         print("予測：",translate(str(x22[jj])))
#         print("実際：",(y22[jj]))
#         print("结果: ",translate(str(x22[jj]))==(y22[jj]))
#         print(" ")
#         check_j+=1
#         print("check_j",check_j)
#         if (translate(str(x22[jj]))==(y22[jj])):
#             check_i+=1
#            # print("成功数：",check_i)
#             #print("確率：",check_i/check_j)
#    except Exception as e:
#         pass
#    continue
# print("確率：",check_i/check_j)


# check_i=0
# check_j=0
# for j in range(1000):
#    try:
#         print("入力: ",x1[j])
#         print("data: ")
#         print("予測：",translate(x1[j]))
#         print("実際：",(y1[j]))
#         print("结果: ",translate(x1[j])==(y1[j]))
#         print(" ")
#         check_j+=1
#         print("check_j",check_j)
#         if (translate(x1[j])==(y1[j])):
#             check_i+=1
#             print("成功数：",check_i)
#             print("確率：",check_i/check_j)
#    except Exception as e:
#         pass
#    continue
# print("確率：",check_i/check_j)


#epo=1；0.32323232323232326                  0.434
#    2         0.43                                                         0.637
#   3         0.6122448979591837                        0.682
#    4          0.62                                                      0.747
#    5      0.65                                                            0.735
#  6  0.65                                                                    0.762
# 7 0.7021063189568706                                         0.794
# 8      0.687                                                                     0.811
# 9      0.686                                                                 0.803
# 10     0.696                                                                          0.806
#11    0.7067067067067067                                         0.837
#12             0.696                                                                    0.833
#13               0.7024048096192385                                      0.843
#14    0.7137137137137137                                               0.832
#15                0.705                                                                    0.805
#16               0.7218875502008032                                      0.768                                                    10%error 0.547
#17                                  0.704                                                            0.703
#18  0。732                 0.7244488977955912                               0.851
#19     0.71          0.7107107107107107                                    0.8648648648648649
#20    0.732             0.7017017017017017                                   0.849
#21 0.708
#22     0.7097097097097097
#23    0.702
#24   0.7097097097097097                 
#25    0.715/0.716
#26
#27
#28
#29
#30
#31
#32
#33
#34
#35 0.89
#36
#37
#38
#39
#40                0.89          0.685                                                                                        


#0.2230   epoch=40   5%      0.698   
#0.2147            0.683     10%
#20%        loss    0.2261            0.701
#15%                                             0.694



#1    0.007
#2     0.062
#3        0.222
#4        0.269
#5           0.342
#6       0.337
#7     0.367
#8   0.423
#  9     0.446
#10    0.467
#11   0.456
#12   0.492s
#13  0.485
#14    0.491
#15
#16
#17
#18
#19
#20
#21
#22
#23

#39         0.691
#40         0.701



#50        0.855    跑了val