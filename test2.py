# Import necessary libraries
import os
import numpy as np
from keras.applications import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from openimages import download

# Define classes and their respective amount of images to be downloaded
classes = ['Cat', 'Dog', 'Person']
n_images = 1000  # number of images per class

# Download images for each class using Open Images

download.download_dataset(
    class_labels=classes,
    dest_dir='./dataset',
    limit=n_images
)

# Path to the directory where the images are stored
base_dir = './dataset'

# Define parameters for the loader
batch_size = 20
img_height = 224
img_width = 224

# Load the training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.3)  # set validation split

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

# Load the validation data
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

# Load the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the base model
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = 5)

# Save the model
model.save('model.h5')