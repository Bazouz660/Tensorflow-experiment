import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the saved model
model = keras.saving.load_model('emotion_classification_model.keras')

# Print TensorFlow and Keras versions
print("=====================================")
print("Loaded Model and Preparing to Fine-Tune")
print("TensorFlow version: ", tf.__version__)
print("Keras version: ", tf.keras.__version__)
print("=====================================")

# Check if CUDA is available
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# Define constants for fine-tuning
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
EPOCHS = 10  # Additional epochs for fine-tuning
NUM_CLASSES = 6
AUTOTUNE = tf.data.AUTOTUNE
NEW_LEARNING_RATE = 0.0001  # Reduced learning rate for fine-tuning

# Define data directory
data_dir = 'dataset'

# Load dataset using tf.data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Prefetching
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Compile the loaded model with a reduced learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=NEW_LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tune the model)
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# Plot fine-tuning results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Fine-Tuning Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Fine-Tuning Loss')
plt.show()

print("Training complete. Saving the model...")
model.save('emotion_classification_model.keras')
print("Model saved successfully.")
