import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model (Make sure to provide the correct path to the model)
MODEL_PATH = 'best_model_mobnetv3.keras'
model = load_model(MODEL_PATH)

def load_image(image_file):
    """ Load the uploaded image file as an array """
    img = Image.open(image_file)
    return img

def preprocess_image(img, target_size=(50, 50)):
    """ Preprocess the image to fit the MobileNet model input """
    img = img.resize(target_size)  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert to array for processing
    img = img / 255.0  # Scale pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(img_array):
    """ Use the model to predict the class of the image """
    prediction = model.predict(img_array)
    return prediction[0]

def main():
    st.title("Breast Cancer Detection using MobileNet")
    st.write("This tool predicts whether histopathologic scans of lymph node sections contain metastatic tissue.")

    # Upload image
    image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Display the uploaded image
        img = load_image(image_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image and predict
        img_array = preprocess_image(img)
        st.write("Processing the image...")
        prediction = predict_image(img_array)

        # Display the prediction
        st.write("Prediction Probabilities:")
        st.write("Non-cancer Probability: ", prediction[0])
        st.write("Cancer Probability: ", prediction[1])
        if np.argmax(prediction) == 0:
            st.write("Prediction: Non-cancerous")
        else:
            st.write("Prediction: Cancerous")

# Run the app
if __name__ == '__main__':
    main()
