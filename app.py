import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import cv2
from skimage.feature import hog

# --- Streamlit Frontend UI ---
st.set_page_config(page_title="Teeth Disease Identifier (Offline AI)", page_icon="🦷", layout="centered")

st.title("🦷 AI Teeth Disease Identifier (Advanced Offline)")
st.markdown("Upload a close-up image of teeth, and our local Machine Learning model will analyze its shapes, gaps, and color profile to identify **Tooth Discoloration** or **Hypodontia**, exclusively using computer vision and the provided dataset.")

# Try to load the local ML model
model_path = "local_dental_model.pkl"
try:
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        model_loaded = True
    else:
        model_loaded = False
except Exception as e:
    model_loaded = False

if not model_loaded:
    st.warning("⚠️ Training local AI Model. Please run 'python train_local_model.py' in the terminal to generate the needed ML model first if you haven't.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

img = None
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

# Button to trigger analysis
submit = st.button("Identify Disease", type="primary")

def extract_features(img_array):
    # Convert RGB array to HSV for better color feature extraction (Crucial for "Tooth Discoloration")
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Process Color Histograms
    hist_h = cv2.calcHist([hsv_image], [0], None, [32], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv_image], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_image], [2], None, [32], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    
    # Normalize color features
    if np.sum(color_features) > 0:
        color_features = color_features / np.sum(color_features)
        
    # Process HOG features (Crucial for "Hypodontia"/Shapes/Gaps)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        gray_image, 
        orientations=8, 
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    # Combine shape (HOG) and color (Histogram) features
    combined_features = np.concatenate([color_features, hog_features])
    return combined_features


if submit:
    if not model_loaded:
        st.error("Cannot perform analysis. Please train the local AI model first by running `python train_local_model.py` in your terminal.")
    elif img is not None:
        try:
            with st.spinner("Analyzing image patterns against local dataset computer vision model... Please wait."):
                # Preprocess the picture identically to the advanced training script
                # 1. Convert to RGB (in case of RGBA/Grayscale)
                processed_img = img.convert('RGB')
                # 2. Resize to (128, 128)
                processed_img = processed_img.resize((128, 128))
                # 3. Convert to Array and grab custom structural features
                img_array = np.array(processed_img)
                features = extract_features(img_array)
                
                # Get prediction probabilities to check for undefined/random images
                probabilities = clf.predict_proba([features])[0]
                max_prob = np.max(probabilities)
                
                # If the model is not confident, it's likely not a valid teeth image
                is_undefined = max_prob < 0.65
                
                if not is_undefined:
                    # Otherwise, proceed with the prediction
                    prediction = clf.classes_[np.argmax(probabilities)]
                    
            if is_undefined:
                st.error("don't use undefiend images")
            else:
                st.success("Analysis Complete!")
                st.markdown("### Diagnosis Report")
                
                st.markdown(f"**🔬 Identified Condition:** {prediction.title()}")
                st.markdown(f"*Data pattern matched with locally provided dataset classes utilizing HOG shape mapping & HSV Color Histograms.*")
                
                # Additional context output based on prediction
                if prediction.lower() == "tooth discoloration":
                    st.write("**Detailed Observations:** The computer vision model detected surface hue, shadowing, or pigmentations heavily matching our dataset records for tooth discoloration.")
                    st.write("**General Recommendation:** Recommended to consider professional dental cleaning, bleaching, or consulting a cosmetic dentist regarding surface stains vs internal discolorations.")
                elif prediction.lower() == "hypodontia":
                    st.write("**Detailed Observations:** The structural layout, edges, and visual gaps indicate characteristics typical of hypodontia (missing developmental teeth) as per the modeled dataset.")
                    st.write("**General Recommendation:** A panoramic X-ray and a clinical evaluation by an orthodontist or prosthodontist is strongly recommended.")
                    
                st.write("---")
                st.write("*Disclaimer:* This assessment is made using a locally trained Offline Computer Vision ML model evaluated exclusively on the provided target dataset. This does NOT substitute a real, professional dental diagnosis.")
            
        except Exception as e:
            st.error(f"An error occurred during local analysis. Ensure you retrained your model: {e}")
    else:
        st.warning("Please upload an image first before clicking 'Identify Disease'.")
