import tensorflow as tf
import numpy as np
import cv2

# Load your trained model
model = tf.keras.models.load_model('emotion_classification_model.keras')

# Define your class labels
class_labels = ['surprise', 'sad', 'neutral', 'happy', 'angry', 'ahegao']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Optional: Flip the frame horizontally (mirror image)
    frame = cv2.flip(frame, 1)

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Perform inference
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_labels[predicted_class_index]

    # Display the resulting frame with prediction
    cv2.putText(frame, f'Emotion: {predicted_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
