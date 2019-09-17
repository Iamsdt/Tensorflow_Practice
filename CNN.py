import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(256*32*32, input_shape=(100, )))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
