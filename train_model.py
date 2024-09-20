import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import matplotlib.pyplot as plt

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Print TensorFlow and Keras versions
print("=====================================")
print("Is TensorFlow using mixed precision? ",
      mixed_precision.global_policy().name)
print("TensorFlow version: ", tf.__version__)
print("Keras version: ", tf.keras.__version__)
print("=====================================")

# Check if CUDA is available
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 6  # surprise, sad, neutral, happy, angry, ahegao
AUTOTUNE = tf.data.AUTOTUNE

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

# Data augmentation with RandomWidth and RandomHeight
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal', dtype='float32'),
    layers.RandomRotation(0.2, dtype='float32'),
    layers.RandomZoom(0.2, dtype='float32'),
    layers.RandomWidth(0.2, dtype='float32'),
    layers.RandomHeight(0.2, dtype='float32'),
    layers.RandomContrast(0.2, dtype='float32'),
], name='data_augmentation')

# Prefetching
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Define the model with Resizing layer
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    layers.Resizing(IMG_HEIGHT, IMG_WIDTH),  # Resizing to fixed dimensions
    layers.Rescaling(1./255),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax',
                 dtype='float32')  # Ensure output is float32
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# Plot training results (optional)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Save the model
print("Training complete. Saving the model...")
model.save('emotion_classification_model.keras')
print("Model saved successfully.")

# Attempt to save the model as a h5 file
print("=====================================")
print("Saving model as an HDF5 file")
model.save('model.h5')
print("Model saved successfully as an HDF5 file.")
