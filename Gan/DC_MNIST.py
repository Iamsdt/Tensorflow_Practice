import tensorflow as tf

print(tf.__version__)

# make database
train, test = tf.keras.datasets.mnist.load_data()
train_image, train_labels = train



