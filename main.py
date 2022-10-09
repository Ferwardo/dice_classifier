import os
import tensorflow as tf
from keras.layers import Rescaling, Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
import tensorflow_model_optimization as tfmot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# caches images
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# define the model, used architecture is AlexNet more info see https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
model = tf.keras.models.Sequential([
    Rescaling(1. / 255),  # rescale the colour of the image so it is between 0 and 1
    Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    BatchNormalization(),

    MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),

    MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),

    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),

    Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),

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
    epochs=10,
    shuffle=True,
    batch_size=32)

model.summary()

# quantise the model
quantize_model = tfmot.quantization.keras.quantize_model
quantized_model = quantize_model(model)
quantized_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

# train the quantised model
quantized_model.fit(train_dataset, validation_split=0.25, batch_size=32, epochs=5)
quantized_model.evaluate(test_dataset, verbose=2)

# convert to tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_model = converter.convert()
with open('dice_classifier.tflite', 'wb') as f:
    f.write(tf_model)
