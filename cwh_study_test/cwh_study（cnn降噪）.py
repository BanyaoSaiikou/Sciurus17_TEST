# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import mnist
import numpy as np
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model
(x_train, _), (x_test, _) = mnist.load_data()

import matplotlib.pyplot as plt






noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


#model.save('/home/cwh/デスクトップ/cwh_study_test/keras_model_hdf5_version3.h5')n = 10

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#  # 显示原始图像
#  ax = plt.subplot(2, n, i + 1)
#  plt.imshow(x_test_noisy[i].reshape(28, 28))
#  plt.gray()
#  ax.get_xaxis().set_visible(False)
#  ax.get_yaxis().set_visible(False)

#  # 显示重构后的图像
#  ax = plt.subplot(2, n, i+1+n)
#  plt.imshow(x_test[i].reshape(28, 28))
#  plt.gray()
#  ax.get_xaxis().set_visible(False)
#  ax.get_yaxis().set_visible(False)
# plt.show()

# 编码过程
input_img = Input(shape=(28, 28, 1)) 

############
# 编码 #
############

# Conv1 #
x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

# Conv2 #
x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 

# Conv 3 #
x = Conv2D(filters = 8,kernel_size=(3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)


############
# 解码 #
############

# DeConv1
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

# DeConv2
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

# Deconv3
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)




autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')#3 loss -500 用不了
#autoencoder.compile(optimizer='adadelta', loss='mse')#5
#autoencoder.compile(optimizer='adam', loss='mse')#4效果最好
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#6效果最好
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')#7
autoencoder.fit(x_train_noisy, x_train,
 epochs=100,
 batch_size=128,
 shuffle=True,
 validation_data=(x_test_noisy, x_test)
 )

autoencoder.save('/home/cwh/デスクトップ/cwh_study_test/keras_model_hdf5_version7.h5')