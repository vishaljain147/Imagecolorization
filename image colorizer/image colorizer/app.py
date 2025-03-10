#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os


def colorizer(img):
    try:
        # Convert image to grayscale, then to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Path to pre-trained model files (Ensure these paths are correct)
        prototxt = r"D:\python\Ai-Black-and-white-image-colorization-with-OpenCV\Ai Black and white image colorization with OpenCV\models\models_colorization_deploy_v2.prototxt"
        model = r"D:\python\Ai-Black-and-white-image-colorization-with-OpenCV\Ai Black and white image colorization with OpenCV\models\colorization_release_v2.caffemodel"
        points = r"D:\python\Ai-Black-and-white-image-colorization-with-OpenCV\Ai Black and white image colorization with OpenCV\models\pts_in_hull.npy"
        
        # Check if model files exist
        if not all(map(os.path.exists, [prototxt, model, points])):
            st.error("Model files are missing or paths are incorrect. Please check the file paths.")
            return None
        
        # Load pre-trained model
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        pts = np.load(points)
        
        # Add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        # Scale the pixel intensities to the range [0, 1], and convert the image from BGR to Lab color space
        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        
        # Resize the image and perform mean centering on the 'L' channel
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        # Predict the 'a' and 'b' channels
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # Resize the 'ab' channels to match the original image dimensions
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
        
        # Combine the 'L' channel with predicted 'ab' channels
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        # Convert the result from Lab to RGB and clip values
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        
        # Convert back to uint8 and return the result
        colorized = (255 * colorized).astype("uint8")
        return colorized
        
    except Exception as e:
        st.error(f"An error occurred during colorization: {e}")
        return None


# Streamlit UI
st.write("""
          # Colorizing Black & White Images
          This app colorizes your black and white images using a deep learning model.
          """)

st.write("Please upload an image file to colorize it.")

file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image.")
else:
    # Load the uploaded image
    image = Image.open(file)
    img = np.array(image)

    # Show original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Original Image")
        st.image(image, use_container_width=True)  # Updated to use_container_width
    
    with col2:
        # Show colorized image
        with st.spinner('Colorizing...'):
            color = colorizer(img)
            if color is not None:
                st.text("Colorized Image")
                st.image(color, use_container_width=True)  # Updated to use_container_width
    
    st.success("Done!")
