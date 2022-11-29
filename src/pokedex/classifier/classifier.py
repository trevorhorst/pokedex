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

img_height = 180
img_width = 180
batch_size = 5
validation_split = 0.1

# Generates a dataset from image files in a directory.
# Supported image formats: jpeg, png, bmp, gif. Animated gifs are truncated to the first frame.
ds_train = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=class_names,
    # color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    # shuffle=True,
    seed=123,
    validation_split=validation_split,
    subset="training",
)

ds_validation = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    # class_names=class_names,
    # color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    # shuffle=True,
    seed=123,
    validation_split=validation_split,
    subset="validation",
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)

class_names = ds_train.class_names
num_classes = len(class_names)
print(class_names)

'''
Overfitting generally occurs when there are a small number of training examples. 
Data augmentation takes the approach of generating additional training data from
your existing examples by augmenting them using random transformations that yield
believable-looking images. This helps expose the model to more aspects of the 
data and generalize better.
'''
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# # Display some of the augmented images
# plt.figure(figsize=(10, 10))
# for images, _ in ds_train.take(1):
#   for i in range(9):
#     augmented_images = data_augmentation(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()


# Create a model
model = keras.Sequential(
    [
        data_augmentation,
        layers.Input((img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(num_classes),
    ]
)


# def augment(x, y):
#     image = tf.image.random_brightness(x, max_delta=0.05)
#     return image, y
# 
# 
# ds_train = ds_train.map(augment)
# 
# # Custom Loops
# for epochs in range(10):
#     for x, y in ds_train:
#         # train here
#         pass


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

model.summary()

epochs = 15
history = model.fit(
    ds_train, 
    validation_data=ds_validation, 
    epochs=epochs,
    verbose=2)

'''
Visualize the training results
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


img = tf.keras.utils.load_img("testset/bulbasaur_0.png", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])



# 
# image = tf.keras.preprocessing.image.load_img("testset/venusaur_0.png", target_size=(img_height, img_width), color_mode="grayscale")
# input_arr = tf.keras.preprocessing.image.img_to_array(
#     image
#     )
# input_arr = np.array([input_arr])  # Convert single image to a batch.
# predictions = model.predict(input_arr)
# print(predictions[0])
# score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


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
