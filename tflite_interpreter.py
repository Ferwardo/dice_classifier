from PIL import Image
import numpy as np
import tensorflow as tf


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def export_preprocessed_image_as_byte_array(image_name, image_dir):
    image = tf.keras.utils.load_img(image_dir + image_name, target_size=(227, 227))
    img1 = tf.keras.utils.img_to_array(image)
    array = tf.expand_dims(img1, axis=0) / 255
    byte_array = tf.io.serialize_tensor(tf.squeeze(array)).numpy()

    file = open("./image_set/"+image_name+".bin", "wb")
    file.write(byte_array)
    file.close()
    return byte_array


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
        img1 = tf.keras.utils.load_img("./image_set/flowers/roses/download.jpg", target_size=(width, height))

    img1 = tf.keras.utils.img_to_array(img1)

    input_data1 = tf.expand_dims(img1, axis=0) / 255

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

    input_data2 = tf.expand_dims(img2, axis=0) / 255

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

    # Test with a second D20
    if classifier_name == "dice_classifier":
        img4 = tf.keras.utils.load_img(
            "./image_set/d20_predict2.jpg", target_size=(width, height)
        )
    else:
        img4 = tf.keras.utils.load_img("./image_set/flowers/roses/download.jpg", target_size=(width, height))

    img4 = tf.keras.utils.img_to_array(img4)

    input_data4 = tf.expand_dims(img4, axis=0) / 255

    interpreter.set_tensor(input_details[0]['index'], input_data4)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data4 = interpreter.get_tensor(output_details[0]['index'])
    results4 = np.squeeze(output_data4)
    top_k4 = results4.argsort()[-5:][::-1]
    print(results4)
    print(top_k4)

    print("\nSecond D20: ")
    for i in top_k4:
        print('{:08.6f}: {}'.format(float(results4[i]), labels[i]))

    # Test with a second D6
    if classifier_name == "dice_classifier":
        img5 = tf.keras.utils.load_img(
            "./image_set/d6_predict2.jpg", target_size=(width, height)
        )
    else:
        img5 = tf.keras.utils.load_img("./image_set/flowers/roses/download.jpg", target_size=(width, height))

    img5 = tf.keras.utils.img_to_array(img5)

    input_data5 = tf.expand_dims(img5, axis=0) / 255

    interpreter.set_tensor(input_details[0]['index'], input_data5)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data5 = interpreter.get_tensor(output_details[0]['index'])
    results5 = np.squeeze(output_data5)
    top_k5 = results5.argsort()[-5:][::-1]
    print(results5)
    print(top_k5)

    print("\nSecond D6: ")
    for i in top_k5:
        print('{:08.6f}: {}'.format(float(results5[i]), labels[i]))
