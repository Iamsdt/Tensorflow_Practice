import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

base_url = 'https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip'
zip_file = get_file("Flowers_data.zip", base_url, extract=True, cache_dir='data/')
print(zip_file)

base_path='/tmp/.keras/datasets/flower_data'

train_data = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, validation_split=0.2)
train_dataset = train_data.flow_from_directory(
    base_path+"/train", target_size=(256, 256), class_mode='spare', subset='validation',
)


test_data = ImageDataGenerator(rescale=1./255)
test_dataset = test_data.flow_from_directory(
    base_path+"/valid", target_size=(256, 256), class_mode='spare'
)

base_model = applications.ResNet50(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(102, activation='softmax')
])

model.compile(
    optimizers.Adam(learning_rate=0.003),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(train_dataset, workers=4, epochs=10)

model.evaluate(test_dataset, workers=4)

# fine tune
base_model.trainable = True

for layer in base_model.layers[:200]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(102, activation='softmax')
])

len(model.trainable_variables)

model.compile(
    optimizers.Adam(learning_rate=0.003),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(train_dataset, workers=4, epochs=2)

model.evaluate(test_dataset, workers=4)

