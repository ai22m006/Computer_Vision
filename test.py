import fiftyone as fo
import fiftyone.zoo as foz
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Define your classes
classes = ['Cat', 'Dog', 'Person']

# Load Open Images Dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    classes=classes,
    max_samples=5000,  # adjust as needed
    seed=51,
    shuffle=True
)

# Export the data to a directory in a format suitable for Keras ImageDataGenerator
export_dir = '/path/to/export/directory'
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.ImageDirectory,
    overwrite=True  # overwrite existing directory
)

# Define image data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # use 20% of the data for validation
)

# Define training and validation generators
train_generator = datagen.flow_from_directory(
    export_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    export_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load the VGG19 model with imagenet weights
base_model = VGG19(weights='imagenet', include_top=False)

# Add new layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Define the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Save the model
model.save('model.h5')
