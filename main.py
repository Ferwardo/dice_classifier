import tensorflow as tf

# get the dataset
dataset_path = "./image_set/dice"

train_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/train", image_size=(480, 480))
test_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path + "/valid")

# normalize data
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, 4))
model.add(tf.keras.layers.MaxPool2D(2))

model.add(tf.keras.layers.Conv2D(32, 4))
model.add(tf.keras.layers.MaxPool2D(2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

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
