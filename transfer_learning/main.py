import os
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker.config import QuantizationConfig

# load the data
dataset_path = "../dice_classifier/image_set/dice_230"

train_data = ImageClassifierDataLoader.from_folder(dataset_path + "/train")
predict_data = ImageClassifierDataLoader.from_folder(dataset_path + "/predict")
data = ImageClassifierDataLoader.from_folder(dataset_path + "/valid")
validation_data, test_data = data.split(0.7)

# fine-tune the model on our training data
model_spec = image_classifier.ModelSpec(uri="https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/5")
model_spec.input_image_shape = [230, 230]
model = image_classifier.create(train_data, validation_data=validation_data, model_spec=model_spec)
model.summary()

print("Done training")
loss, accuracy = model.evaluate(test_data)

print("Quantise and convert model")
config = QuantizationConfig(inference_input_type=tf.int8, inference_output_type=tf.int8,
                            supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8], representative_data=test_data)
model.export(export_dir='.', tflite_filename="dice_classifier.tflite", label_filename="dice_labels.txt",
             with_metadata=False, quantization_config=config)

print(model.evaluate_tflite("dice_classifier.tflite", predict_data))

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
