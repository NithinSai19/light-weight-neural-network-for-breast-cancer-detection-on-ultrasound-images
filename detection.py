import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model,load_model
from joblib import load
import cv2
from PIL import Image

st.write('''
# A Light Weight Neural Network for detecting Breast Cancer using classification and feature selection techniqies
''')

# Get the uploaded image file
uploaded_file = st.file_uploader(" Upload a Breast Ultrasound Image Here:", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)    
    image = np.array(image)
    st.write("Uploaded Image Size : ",image.shape)


    model = load_model('best_model_128_128.h5',compile=False)    
    lwnn_model = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)
    with open('Random_forest_model_250.joblib', 'rb') as file:
        rf_model = load(file)

    Img = cv2.resize(image, (128, 128))

    col1, col2 = st.columns(2)
    with col1:
        st.image(Img, caption='Uploaded Image', width=300)

    img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)

    img_feature = lwnn_model.predict(img)
    img_feature = img_feature.reshape(-1, img_feature.shape[3])

    prediction = rf_model.predict(img_feature)
    predicted_mask = prediction.reshape((128,128))

    colored_mask = cv2.merge([np.zeros_like(predicted_mask), np.zeros_like(predicted_mask),255 * np.ones_like(predicted_mask)])
    colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=predicted_mask)

    # Overlay the colored mask onto the original image
    detected_image = cv2.addWeighted(Img, 0.7, colored_mask, 0.3, 0)
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    with col2:
        st.image(detected_image_rgb, caption='Detected Image', width=300)