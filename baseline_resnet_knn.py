import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
import time
import requests

from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

# Streamlit page configuration
st.set_page_config(page_title="Fashion Recommender (ResNet)", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>👗 Fashion Recommender - ResNet50 + KNN</h1>", unsafe_allow_html=True)


# Load ResNet50 model (cached)
@st.cache_resource(show_spinner=False)
def load_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='max')

model = load_model()

# Load embeddings and filenames (cached)
@st.cache_data(show_spinner=False)
def load_pickle_data():
    features = np.array(pickle.load(open('all_embeddings_resnet_max.pkl', 'rb')))
    names = pickle.load(open('filenames.pkl', 'rb'))  # These should now be URLs
    return features, names

feature_list, filenames = load_pickle_data()

# Save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None


# Feature Extraction (cached)
@st.cache_data(show_spinner=False)
def feature_extraction(uploaded_file, model):
    img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# KNN-based Recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File Upload UI
uploaded_file = st.file_uploader("Upload a clothing image")

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    
    if file_path:
        st.markdown("### Uploaded Image:")
        st.image(uploaded_file, width=200)

        with st.spinner("Extracting features and finding similar items..."):
            start_feat = time.time()
            features = feature_extraction(file_path,model)
            feature_time = time.time() - start_feat

            start_rec = time.time()
            indices = recommend(features, feature_list)
            search_time = time.time() - start_rec

        st.success("Recommendations Ready!")

        st.markdown(f"**Feature Extraction Time:** `{feature_time:.3f} sec`")
        st.markdown(f"**Similarity Search Time:** `{search_time:.3f} sec`")

        # Display top 5 recommendations (excluding the uploaded image itself)
        st.markdown("###Recommended Similar Items:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            url = filenames[indices[0][i + 1]]  # skipping the uploaded image
            col.image(url, width=140, caption=f"Item {i+1}")
    else:
        st.error("Failed to process uploaded image.")
