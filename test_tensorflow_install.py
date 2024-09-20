import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Print TensorFlow and Keras versions
print("=====================================")
print("Loaded Model and Preparing to Fine-Tune")
print("Keras version: ", tf.keras.__version__)
print("TensorFlow version: ", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices(
    'GPU') else "NOT AVAILABLE")
print("=====================================")
