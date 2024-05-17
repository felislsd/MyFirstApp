from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import imghdr  # for file content validation
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO


# Full screen
st.set_page_config(layout="wide")

# Load model
@st.cache_resource
def load_cached_model():
    model_path = '/app/catsvsdogs_modelv3.h5'
    model = load_model(model_path)
    return model

model = load_cached_model()



st.title("Cat vs Dog Detector")
st.write("This is a simple app that distinguishes between cats and dogs in a photo")

subject_img = None

use_local_image = st.checkbox("Use Local Image", value=False) #False - it will use url
cols = st.columns(2)

if use_local_image:
    subject_file = cols[0].file_uploader("Choose Subject Image...", type = ["jpg", "png", "jpeg"], key='subject')
    if subject_file is not None:
        subject_img = Image.open(subject_file)
else:
    subject_url = cols[0].text_input("Enter Subject Image URL", "")
    if subject_url:
        response = requests.get(subject_url)
        if response.status_code == 200:
            subject_img = Image.open(BytesIO(requests.get(subject_url).content))





# Model predict


def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Define prediction function
def predict_image(img, model):
    img = preprocess_image(img)
    prediction = model.predict(img)

    # Return class label
    class_labels = ['Cat', 'Dog', 'Other']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    predicted_probabilities = prediction[0] #all classes
    return predicted_class, predicted_probabilities

# Make prediction

if subject_img is not None:
    cols[0].image(subject_img, caption = 'Subject Image', use_column_width=True)
    open_cv_image = cv2.cvtColor(np.array(subject_img), cv2.COLOR_RGB2BGR)
    predicted_class, predicted_probabilities = predict_image(open_cv_image, model)

    # Display results
    probabilities = {
        "Cat": predicted_probabilities[0],
        "Dog": predicted_probabilities[1],
        "Other": predicted_probabilities[2]
    }



    # Find the class with the highest probability
    max_class = max(probabilities, key=probabilities.get)


    # Set the color of each bar based on whether it corresponds to the max_class
    colors = ["#00FF80" if cls == max_class else "#E0E0E0" for cls in probabilities.keys()]



    # Create a horizontal bar chart using matplotlib
    fig, ax = plt.subplots()


    bars = ax.barh(list(probabilities.keys()), list(probabilities.values()), color=colors)

    # Set the colors for axes and text
    ax.xaxis.label.set_color('#E0E0E0')
    ax.yaxis.label.set_color('#E0E0E0')
    ax.tick_params(axis='x', colors='#E0E0E0')
    ax.tick_params(axis='y', colors='#E0E0E0')

    # Set the background color of the chart
    fig.patch.set_facecolor('#0e1117')  # Dark background color

    # Set the background color of the chart area
    ax.set_facecolor('#0e1117')  # Dark background color

    fig.subplots_adjust(left=0.5)

    # Add labels and title
    ax.set_title('Probabilities').set_color('#E0E0E0')

    # Add probability values on bars
    for bar, prob in zip(bars, probabilities.values()):
        ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, f'{prob:.2f}', va='center', ha='left')

    # Display the chart
    cols[1].pyplot(fig)
