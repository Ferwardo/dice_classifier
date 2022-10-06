import os
import tensorflow as tf
import numpy as np

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

# get the dataset
dataset_path = "./image_set/dice"

train_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/train", image_size=(480, 480),
                                                            seed=123, batch_size=32)
test_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/valid", image_size=(480, 480),
                                                           seed=123, batch_size=32)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# define the model, used architecture is AlexNet
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                           input_shape=(227, 227, 3)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    shuffle=True,
    batch_size=32)

model.summary()
