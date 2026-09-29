import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small


# -----------------------------------------
# Settings
# -----------------------------------------

DATASET_DIR = "dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


# -----------------------------------------
# Load dataset
# -----------------------------------------

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="both",
    seed=42
)

train_ds, validation_ds = dataset


# -----------------------------------------
# Class names
# -----------------------------------------

class_names = train_ds.class_names

print("Classes:")
print(class_names)

print("Number of classes:", len(class_names))


# Save class names

os.makedirs("models", exist_ok=True)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)


# -----------------------------------------
# Improve performance
# -----------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)


# -----------------------------------------
# Data augmentation
# -----------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# -----------------------------------------
# MobileNetV3
# -----------------------------------------

base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers

base_model.trainable = False


# -----------------------------------------
# Build model
# -----------------------------------------

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs,
    outputs
)


# -----------------------------------------
# Compile
# -----------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------------------
# Train
# -----------------------------------------

print("\nTraining started...\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS
)


# -----------------------------------------
# Save model
# -----------------------------------------

model.save(
    "models/fruit_mobilenetv3.keras"
)

print("\nModel saved successfully!")

print(
    "models/fruit_mobilenetv3.keras"
)


# -----------------------------------------
# Accuracy graph
# -----------------------------------------

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "MobileNetV3 Training Accuracy"
)

plt.legend()

plt.show()