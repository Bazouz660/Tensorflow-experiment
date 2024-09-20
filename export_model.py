import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
print("=====================================")
print("Loading Model")
model = load_model('emotion_classification_model.keras')
print("Model Loaded")

print("=====================================")
print("Exporting Model")
# Create the artifact
filepath = "emotion_classification_model"
model.export(filepath)
print("Model Exported")
