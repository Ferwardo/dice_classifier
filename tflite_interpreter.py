from PIL import Image
import numpy as np
import tensorflow as tf


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def normalize_image(image):
    array = np.array(image).astype(np.float32) / 255.0
    return Image.fromarray(array.astype('uint8'), 'RGB')


def predict(classifier_name):
    print("With tflite interpreter")

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=classifier_name + ".tflite", )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test with a D6 or rose
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    if classifier_name == "dice_classifier":
        img1 = tf.keras.utils.load_img(
            "./image_set/dice/predict/d6/d6_predict.jpg", target_size=(width, height)
        )
    else:
        img1 = tf.keras.utils.load_img("./image_set/flowers/roses/download.jpg", target_size=(width,height))


    img1 = tf.keras.utils.img_to_array(img1)

    input_data1 = tf.expand_dims(img1, axis=0)/255

    interpreter.set_tensor(input_details[0]['index'], input_data1)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data1 = interpreter.get_tensor(output_details[0]['index'])
    results1 = np.squeeze(output_data1)
    top_k1 = results1.argsort()[-5:][::-1]
    print(results1)
    print(top_k1)

    labels = load_labels("dice_labels.txt")

    print("D6: ")
    for i in top_k1:
        print('{:08.6f}: {}'.format(float(results1[i]), labels[i]))

    # Test with a D8 or daisy
    if classifier_name == "dice_classifier":
        img2 = tf.keras.utils.load_img(
            "./image_set/dice/predict/d8/d8_predict.jpg", target_size=(width, height)
        )
    else:
        img2 = tf.keras.utils.load_img("./image_set/flowers/roses/download.jpg", target_size=(width, height))

    img2 = tf.keras.utils.img_to_array(img2)

    input_data2 = tf.expand_dims(img2, axis=0)/255

    interpreter.set_tensor(input_details[0]['index'], input_data2)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data2 = interpreter.get_tensor(output_details[0]['index'])
    results2 = np.squeeze(output_data2)
    top_k2 = results2.argsort()[-5:][::-1]
    print(results2)
    print(top_k2)

    print("\nD8: ")
    for i in top_k2:
        print('{:08.6f}: {}'.format(float(results2[i]), labels[i]))

    # Test with a D20
    if classifier_name == "dice_classifier":
        img3 = tf.keras.utils.load_img(
            "./image_set/dice/predict/d20/d20_predict.jpg", target_size=(width, height)
        )
    else:
        img3 = tf.keras.utils.load_img("./image_set/flowers/roses/download.jpg", target_size=(width, height))

    img3 = tf.keras.utils.img_to_array(img3)

    input_data3 = tf.expand_dims(img3, axis=0) / 255

    interpreter.set_tensor(input_details[0]['index'], input_data3)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data3 = interpreter.get_tensor(output_details[0]['index'])
    results3 = np.squeeze(output_data3)
    top_k3 = results3.argsort()[-5:][::-1]
    print(results3)
    print(top_k3)

    print("\nD20: ")
    for i in top_k3:
        print('{:08.6f}: {}'.format(float(results3[i]), labels[i]))
