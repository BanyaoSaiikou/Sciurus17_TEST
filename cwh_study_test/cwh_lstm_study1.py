# -*- coding:utf-8 -*-


from tensorflow import keras
#from tensorflow.keras import layers##造成这个错误CUDNN_STATUS_EXECUTION_FAILED
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt


num_words = 30000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)#num_words/skip_top ===》大于/小于某数的数值变成2
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
# print(x_train, ' ', y_train)
# print(x_test, ' ', y_test)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')#往后填充
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
# print(x_train, ' ', y_train)
# print(x_test, ' ', y_test)
def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),#嵌入层
        layers.LSTM(32, return_sequences=True),#循环层
        layers.LSTM(1, activation='sigmoid', return_sequences=False)#return_sequences=True，则输出全部序列，即每一个 Cell 的 $h_t$。
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model
model = lstm_model()
model.summary()#输出结构

history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)

model.save('/home/cwh/デスクトップ/cwh_study_test/lstm_test01.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()
