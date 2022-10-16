import os

import tflite_interpreter
import tensorflow as tf
from keras.layers import Rescaling, Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
import tensorflow_model_optimization as tfmot

# from keras.applications import MobileNet

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DICE_DATASET = True

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

image_size = (227, 227)
batch_size = 32

if DICE_DATASET:
    # get the dataset, used dataset is https://www.kaggle.com/datasets/ucffool/dice-d4-d6-d8-d10-d12-d20-images
    # where the d4 and d12 classes are not used
    dataset_path = "./image_set/dice"

    train_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/train", image_size=image_size,
                                                                seed=123, batch_size=batch_size)
    test_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/valid", image_size=image_size,
                                                               seed=123, batch_size=batch_size)
else:
    import pathlib

    # download the dataset
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)

    # load the dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",
                                                                seed=123, image_size=image_size, batch_size=batch_size)
    test_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation",
                                                               seed=123, image_size=image_size, batch_size=batch_size)

# caches images
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# too big in filesize it doesn't work maybe the optimizer will make it smaller
# define the model, the architecture it is based on is AlexNet, layers are still based on it but not the parameters
# more info see https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
# the original implementation (of alexnet) is found here:
# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
model = tf.keras.models.Sequential([
    # Rescaling(1. / 255),  # rescale the colour of the image, so it is between 0 and 1
    # Commented out the batch layers as it otherwise doesn't work
    # BatchNormalization(),

    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(227, 227, 3)),
    # BatchNormalization(),

    MaxPool2D(pool_size=(3, 3)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"),
    # BatchNormalization(),

    MaxPool2D(pool_size=(3, 3)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"),
    # BatchNormalization(),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"),
    # BatchNormalization(),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"),
    # BatchNormalization(),

    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# compile and fit
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# checkpoint to save the best model weight from the training
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/checkpoint.ckpt", save_weights_only=True,
#                                                 verbose=1, save_best_only=True)

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    shuffle=True,
    batch_size=batch_size,
    # callbacks=[checkpoint]
)

model.summary()
model.save_weights("./checkpoints/checkpoint_1", save_format="tf")

# quantise the model
quantize_model = tfmot.quantization.keras.quantize_model
quantized_model = quantize_model(model)
quantized_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])

# train the quantised model
quantized_model.fit(train_dataset, validation_data=test_dataset, shuffle=True, batch_size=32, epochs=5)
quantized_model.evaluate(test_dataset, verbose=2)

# def representative_data_gen():
#     for input_value in tf.keras.utils.image_dataset_from_directory(dataset_path + "/train").batch(1).take(100):
#         # Model has only one input so each data point has one element.
#         yield [input_value]


# convert to tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tf_model = converter.convert()

if DICE_DATASET:
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
else:
    with open('flower_classifier.tflite', 'wb') as f:
        f.write(tf_model)

    # convert to C array and add the necessary code for eIQ
    os.system("xxd.exe -i flower_classifier.tflite > flower_classifier.h")

    with open("flower_classifier.h", 'r') as original:
        data = original.read()

    with open("flower_classifier.h", "w") as modified:
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

# predict to see if it works
if (DICE_DATASET):
    import numpy as np

    predict_set = tf.keras.utils.image_dataset_from_directory("./image_set/predict", image_size=image_size)
    print("Normal model")
    predictions = model.predict(predict_set)
    print(predictions)

    print("Quantized model")
    predictions = quantized_model.predict(predict_set)
    print(predictions)

    print("With tflite interpreter")

    tflite_interpreter.predict("dice_classifier")
