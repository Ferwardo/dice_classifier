import os

import keras
import tensorflow as tf
from keras.layers import Rescaling, Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
import tensorflow_model_optimization as tfmot
import keras.regularizers as regularizers
import matplotlib.pyplot as plt
from keras.applications import MobileNet

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# setup
DICE_DATASET = True
USE_MOBILENET = False
EPOCHS = 60

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

if USE_MOBILENET:
    image_size = (224, 224)
else:
    image_size = (227, 227)
batch_size = 32


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def save_history(model_log, filepath):
    plt.figure()
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(filepath)
    plt.close()

    plt.figure()
    plt.plot(model_log.history['accuracy'])
    plt.plot(model_log.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig("accuracy_" + filepath)
    plt.close()


# Load in the dataset depending on the set flag
if DICE_DATASET:
    # get the dataset, used dataset is https://www.kaggle.com/datasets/ucffool/dice-d4-d6-d8-d10-d12-d20-images
    # where the d4 and d12 classes are not used
    dataset_path = "./image_set/dice_230"

    train_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/train",
                                                                image_size=image_size,
                                                                seed=123,
                                                                batch_size=batch_size)
    test_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/valid",
                                                               image_size=image_size,
                                                               seed=123,
                                                               batch_size=batch_size)
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

print(train_dataset.class_names)


# normalise the data
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


train_dataset = train_dataset.map(process)
test_dataset = test_dataset.map(process)

# caches images
AUTOTUNE = tf.data.AUTOTUNE

norm_train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
norm_test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

if USE_MOBILENET:
    model = MobileNet((224, 224, 3))
else:
    # Define the model, the architecture this model is based on is AlexNet, layers are still based on it but not the
    # parameters more info see https://dl.acm.org/doi/abs/10.1145/3065386
    # the original implementation (of alexnet) is found here:
    # https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
    model = tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(227, 227, 3),
               kernel_regularizer=regularizers.l2(0.001)),
        # BatchNormalization(),

        MaxPool2D(pool_size=(3, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same",
               kernel_regularizer=regularizers.l2(0.001)),
        # BatchNormalization(),

        MaxPool2D(pool_size=(3, 3)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same",
               kernel_regularizer=regularizers.l2(0.001)),
        # BatchNormalization(),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same",
               kernel_regularizer=regularizers.l2(0.001)),
        # BatchNormalization(),

        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same",
               kernel_regularizer=regularizers.l2(0.001)),
        # BatchNormalization(),

        MaxPool2D(pool_size=(3, 3)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
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

if DICE_DATASET:
    # this is to balance the classes as otherwise the classification isn't good.
    class_weights = {
        0: 2.0,
        1: 1.0,
        2: 2.6,
        3: 2.3,
        4: 1.2
    }

    history = model.fit(
        norm_train_dataset,
        validation_data=norm_test_dataset,
        epochs=EPOCHS,
        shuffle=True,
        batch_size=batch_size,
        class_weight=class_weights,
        # callbacks=[checkpoint]
    )
else:
    history = model.fit(
        norm_train_dataset,
        validation_data=norm_test_dataset,
        epochs=EPOCHS,
        shuffle=True,
        batch_size=batch_size,
        # callbacks=[checkpoint]
    )
model.summary()

save_history(history, 'model_'+str(EPOCHS)+'_epochs.png')

# model.save_weights("./checkpoints/checkpoint_1", save_format="tf")
# model.load_weights("./checkpoints/checkpoint_1")

# quantise the model
quantize_model = tfmot.quantization.keras.quantize_model
quantized_model = quantize_model(model)
# quantized_model = keras.Sequential([
#     Rescaling(1. / 255),
#     quantized_model
# ])

quantized_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])

# train the quantised model
quantized_history = quantized_model.fit(train_dataset, validation_data=test_dataset, shuffle=True, batch_size=32,
                                        epochs=int(EPOCHS / 4))
print(quantized_model.evaluate(test_dataset, verbose=1))
save_history(quantized_history, 'quantized_model_'+str(int(EPOCHS/4))+'_epochs.png')
quantized_model.save("./quantized_model")
model.save("./model")

# def representative_data_gen():
#     for input_value in tf.keras.utils.image_dataset_from_directory(dataset_path + "/train").batch(1).take(100):
#         # Model has only one input so each data point has one element.
#         yield [input_value]

# convert to tflite format
converter = tf.lite.TFLiteConverter.from_saved_model("./quantized_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
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
if DICE_DATASET:
    predict_set = tf.keras.utils.image_dataset_from_directory("./image_set/dice/predict", image_size=image_size,
                                                              seed=123)
    norm_predict_set = predict_set.map(process)

    print("Normal model")
    predictions = model.predict(norm_predict_set, verbose=1)
    labels = load_labels("dice_labels.txt")

    class_names = predict_set.class_names
    for images, image_labels in predict_set.take(1):
        for i in range(len(image_labels)):
            print(class_names[image_labels[i]])

    for j in predictions:
        print(j)
        top_k1 = j.argsort()[-5:][::-1]
        for i in top_k1:
            print('{:08.6f}: {}'.format(float(j[i]), labels[i]))
        print("\n")

    print("Quantized model")
    predictions = quantized_model.predict(norm_predict_set, verbose=1)

    for j in predictions:
        print(j)
        top_k1 = j.argsort()[-5:][::-1]
        for i in top_k1:
            print('{:08.6f}: {}'.format(float(j[i]), labels[i]))
        print("\n")

    import tflite_interpreter

    tflite_interpreter.predict("dice_classifier")
