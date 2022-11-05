import PIL
import os
from PIL import Image

# Base path
dir_path = "D:/masterproef_code/dice_classifier/image_set/dice"
new_dir_path = "D:/masterproef_code/dice_classifier_transfer_learning/image_set/dice_128"

# prediction photos
category_path = "/predict"

SIZE = (128,128)

for directory in os.listdir(dir_path + category_path):
    # make the new directories if they don't exist.
    os.makedirs(new_dir_path + category_path + "/" + directory, exist_ok=True)

    # resize all the images and place them into the new directories
    for file in os.listdir(dir_path + category_path + "/" + directory):
        image_path = category_path + "/" + directory + "/" + file
        image = Image.open(dir_path + image_path)
        image = image.resize(SIZE)
        image.save(new_dir_path + image_path)

# training photos
category_path = "/train"

for directory in os.listdir(dir_path + category_path):
    # make the new directories if they don't exist.
    os.makedirs(new_dir_path + category_path + "/" + directory, exist_ok=True)

    # resize all the images and place them into the new directories
    for file in os.listdir(dir_path + category_path + "/" + directory):
        image_path = category_path + "/" + directory + "/" + file
        image = Image.open(dir_path + image_path)
        image = image.resize(SIZE)
        image.save(new_dir_path + image_path)

# validation photos
category_path = "/valid"

for directory in os.listdir(dir_path + category_path):
    # make the new directories if they don't exist.
    os.makedirs(new_dir_path + category_path + "/" + directory, exist_ok=True)

    # resize all the images and place them into the new directories
    for file in os.listdir(dir_path + category_path + "/" + directory):
        image_path = category_path + "/" + directory + "/" + file
        image = Image.open(dir_path + image_path)
        image = image.resize(SIZE)
        image.save(new_dir_path + image_path)
