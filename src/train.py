import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.waste_classifier import build_transfer_model

IMG_ROWS, IMG_COLS = 150, 150
BATCH_SIZE = 32
N_CLASSES = 2
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 10

DATA_DIR = os.path.join('data', 'o-vs-r-split', 'train')
TEST_DIR = os.path.join('data', 'o-vs-r-split', 'test')

train_datagen = ImageDataGenerator(
    validation_split=VAL_SPLIT,
    rescale=1.0/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    validation_split=VAL_SPLIT,
    rescale=1.0/255.0
)

train_generator = train_datagen.flow_from_directory(
    directory=DATA_DIR,
    seed=SEED,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    target_size=(IMG_ROWS, IMG_COLS),
    subset='training'
)
val_generator = val_datagen.flow_from_directory(
    directory=DATA_DIR,
    seed=SEED,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    target_size=(IMG_ROWS, IMG_COLS),
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    class_mode='binary',
    seed=SEED,
    batch_size=BATCH_SIZE,
    shuffle=False,
    target_size=(IMG_ROWS, IMG_COLS)
)

model = build_transfer_model(input_shape=(IMG_ROWS, IMG_COLS, 3))
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, mode='min', min_delta=0.01),
    ModelCheckpoint('models/waste_classifier_vgg16.keras', monitor='val_loss', save_best_only=True, mode='min')
]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

# Unfreeze some layers and fine-tune
for layer in model.layers[-5:]:
    layer.trainable = True

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-5),
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

# Save training history plot
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(fine_tune_history.history['accuracy'], label='Training Accuracy')
plt.plot(fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.summary()

print(len(train_generator))