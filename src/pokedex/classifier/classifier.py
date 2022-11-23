# Imports needed
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 28
img_width = 28
batch_size = 2

class_names = ["bulbasaur", "ivysaur", "venusaur"]

# Create a model
model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

# Generates a dataset from image files in a directory.
# Supported image formats: jpeg, png, bmp, gif. Animated gifs are truncated to the first frame.
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=class_names,
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    # class_names=class_names,
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


ds_train = ds_train.map(augment)

# Custom Loops
for epochs in range(10):
    for x, y in ds_train:
        # train here
        pass


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2)

# test_loss, test_acc = model.evaluate(ds_train, verbose=2)
# 
# print('\nTest accuracy:', test_acc)
# 
# 
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# 
# predictions = probability_model.predict(ds_validation)
# 
# print(predictions[0])
