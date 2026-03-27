import os
import glob
import numpy as np
from PIL import Image
import joblib
import cv2
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset_dir = r"c:\Users\SRI CHARAN\OneDrive\Desktop\datasets iomp\dental"
classes = ["Tooth Discoloration", "hypodontia"]

# Resize images to speed up training while preserving structural and color features
IMG_SIZE = (128, 128)

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

def load_data():
    X = []
    y = []
    for label in classes:
        class_dir = os.path.join(dataset_dir, label)
        # Recursively find common image formats
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(glob.glob(os.path.join(class_dir, '**', ext), recursive=True))
            
        print(f"Found {len(image_paths)} images for {label}. Extracting advanced structure/color features...")
        count = 0
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                
                # Extract customized features
                features = extract_features(img_array)
                
                X.append(features)
                y.append(label)
                count += 1
            except Exception as e:
                pass
        print(f"Successfully loaded {count} advanced samples for {label}.")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print(f"Starting ADVANCED model building process...")
    print(f"Looking in: {dataset_dir}")
    X, y = load_data()
    print(f"Total structured dataset shape: {X.shape}")
    
    if len(X) == 0:
        print("Error: No images were found or loaded properly!")
        exit(1)
        
    print("Training enhanced Random Forest Classifier on structural and color mappings...")
    # Train robust random forest utilizing deep decision trees instead of basic flattening
    clf = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, n_jobs=-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    
    print("\n--- Evaluating Model Accuracy on Test Set Validation ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    model_path = "local_dental_model.pkl"
    print(f"Saving trained enhanced model to {model_path}...")
    joblib.dump(clf, model_path)
    print("Completed successfully! Your intelligent offline model is ready.")
