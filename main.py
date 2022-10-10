import os

import keras
import tensorflow as tf
from keras.layers import Rescaling, Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
import tensorflow_model_optimization as tfmot
from keras.applications import MobileNet

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=5120)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# get the dataset, used dataset is https://www.kaggle.com/datasets/ucffool/dice-d4-d6-d8-d10-d12-d20-images
# where the d4 and d12 classes are not used
dataset_path = "./image_set/dice"

train_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/train", image_size=(227, 227),
                                                            seed=123, batch_size=32)
test_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/valid", image_size=(227, 227),
                                                           seed=123, batch_size=32)

# caches images
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# too big in filesize it doesn't work maybe the optimizer will make it smaller
# define the model, the architecture it is based on is AlexNet
# more info see https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
model = tf.keras.models.Sequential([
    # Rescaling(1. / 255),  # rescale the colour of the image, so it is between 0 and 1
    # added this layer to see if this then takes 480 by 480 images, it does so yay
    # Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(480, 480, 3)),
    # Commented out the batch layers as it otherwise doesn't work
    # BatchNormalization(),

    Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    # BatchNormalization(),

    MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    # BatchNormalization(),

    MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    # BatchNormalization(),

    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    # BatchNormalization(),

    Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    # BatchNormalization(),

    MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# compile and fit
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=30,
    shuffle=True,
    batch_size=32)

model.summary()

# quantise the model
quantize_model = tfmot.quantization.keras.quantize_model
quantized_model = quantize_model(model)
quantized_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])

# train the quantised model
quantized_model.fit(train_dataset, validation_data=test_dataset, shuffle=True, batch_size=32, epochs=5)
quantized_model.evaluate(test_dataset, verbose=2)

# convert to tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_model = converter.convert()

with open('dice_classifier.tflite', 'wb') as f:
    f.write(tf_model)

# convert to C array and add the necessary code for eIQ
os.system("xxd.exe -i dice_classifier.tflite > dice_classifier.h")

with open("dice_classifier.h", 'r') as original:
    data = original.read()

with open("dice_classifier.h", "w") as modified:
    modified.write("""
#ifndef __XCC__
#include <cmsis_compiler.h>
#else
#define __ALIGNED(x) __attribute__((aligned(x)))
#endif
#define MODEL_NAME "dice_classifier"
#define MODEL_INPUT_MEAN 0.0f
#define MODEL_INPUT_STD 255.0f\n
    """ + data)
