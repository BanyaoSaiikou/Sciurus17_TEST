#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt
from glob import glob


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#通过通过通过通过通过  
# ----====https://blog.csdn.net/weixin_46039239/article/details/115212393====----  
# 解决解决解决解决解决
#tensorflow.python.framework.errors_impl.NotFoundError:  No algorithm worked!
         #[[node sequential/conv2d/Relu (defined at gazebo_classification_test.py:79) ]] [Op:__inference_train_function_909]
# W tensorflow/core/kernels/data/generator_dataset_op.cc:107] Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.[[{{node PyFunc}}]]


# 数据所在文件夹
base_dir = '/home/cwh/デスクトップ/gazebo_dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 训练集
train_cats_dir = os.path.join(train_dir, 'success')
train_dogs_dir = os.path.join(train_dir, 'failure')

# 验证集
validation_cats_dir = os.path.join(validation_dir, 'success')
validation_dogs_dir = os.path.join(validation_dir, 'failure')

#构造卷积神经网络模型
#对于GPU可以输入224*224
#对于CPU输入64*64，速度可以快10倍

#以下第一种方法科学 直观展示summary
model = tf.keras.models.Sequential([
    # 如果训练慢，可以把数据设置的更小一些   Conv2D方形卷积核、行列相同步长
    #           卷积核数目 过滤器的大小 激活函数 当使用该层作为模型第一层时，需要提供input_shape
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
#CNN中的最大池化（MaxPooling2D）                   隐藏层ReLU
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # 为全连接层准备
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    # 二分类sigmoid就够了
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # 文件夹路径
        target_size=(64, 64),  # 指定resize成的大小
        batch_size=20,
        # 如果one-hot就是categorical，二分类用binary就可以
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=10,  # 2000 images = batch_size * steps_per_epoch
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * validation_steps
      verbose=2)


#tf.saved_model.save(model, "/home/cwh/デスクトップ/gazebo_dataset/saved1")
model.save('/home/cwh/デスクトップ/gazebo_dataset/saved1/keras_model_hdf5_version.h5')
