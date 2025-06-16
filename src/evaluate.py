import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_ROWS, IMG_COLS = 150, 150
BATCH_SIZE = 32
TEST_DIR = os.path.join('data', 'o-vs-r-split', 'test')

# Load the trained model
model = load_model('waste_classifier_vgg16.keras')

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(IMG_ROWS, IMG_COLS),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get a batch of images and labels from the test generator
images, labels = next(iter(test_generator))
preds = model.predict(images)
pred_labels = (preds > 0.5).astype(int).flatten()

# Plot the first 5 test images with actual and predicted labels
for i in range(min(5, len(images))):
    plt.imshow(images[i])
    plt.title(f"Actual: {int(labels[i])}, Predicted: {pred_labels[i]}")
    plt.axis('off')
    plt.show()

# Get true labels and predictions for the entire test set
test_generator.reset()
preds = model.predict(test_generator)
pred_labels = (preds > 0.5).astype(int).flatten()
true_labels = test_generator.classes

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=test_generator.class_indices.keys()))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))