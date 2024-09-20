import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('emotion_classification_model.keras')

# Check if the model loaded correctly
model.summary()
