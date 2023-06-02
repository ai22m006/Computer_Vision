from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
import fiftyone as fo
import fiftyone.zoo as foz


# Define your classes
classes = ['Cat', 'Dog', 'Person']

print("load")
# Load Open Images Dataset
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    classes=classes,
    label_types=["detections", "classifications"],
    max_samples=10,  # adjust as needed
    dataset_name="cats-dogs-people"
)

print("export")
# Export the data to a directory in a format suitable for Keras ImageDataGenerator
export_dir = '../../../Users/a760660/fiftyone/open-images-v7/train/'
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.ImageDirectory,
    overwrite=True  # overwrite existing directory
)

print("datagen")
# Define image data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # use 20% of the data for validation
)

print("train")
# Define training and validation generators
train_generator = datagen.flow_from_directory(
    export_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

print("val")
validation_generator = datagen.flow_from_directory(
    export_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("vgg19")
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

print("compile")
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
