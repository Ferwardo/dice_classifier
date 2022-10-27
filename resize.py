import PIL
import os
from PIL import Image

# Base path
dir_path = "D:/masterproef_code/dndDice/image_set/dice"
new_dir_path = "D:/masterproef_code/dndDice/image_set/dice_230"

# prediction photos
category_path = "/predict"

for directory in os.listdir(dir_path + category_path):
    # make the new directories if they don't exist.
    os.makedirs(new_dir_path + category_path + "/" + directory, exist_ok=True)

    # resize all the images and place them into the new directories
    for file in os.listdir(dir_path + category_path + "/" + directory):
        image_path = category_path + "/" + directory + "/" + file
        image = Image.open(dir_path + image_path)
        image = image.resize((230, 230))
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
        image = image.resize((230, 230))
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
        image = image.resize((230, 230))
        image.save(new_dir_path + image_path)
