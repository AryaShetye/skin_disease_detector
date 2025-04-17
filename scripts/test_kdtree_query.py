import sys
import os

# Add the root directory of your project to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import pickle
from core.feature_extraction.extractor import extract_features  # Now it should work

# Load tree, data, and label map
with open('kdtree.pkl', 'rb') as f:
    tree = pickle.load(f)

features_array = np.load('features.npy')
labels_array = np.load('labels.npy')

# Load the label map
with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# Path to the test image
test_img = cv2.imread(r"C:\Users\aryas\OneDrive\Desktop\skin_disease_detector\assets\test_images\BCC-test.jpg")
feat = extract_features(test_img)

# Debug: Check the shape of the feature vector and the training data
print(f"Feature vector shape (query): {feat.shape}")
print(f"Training data shape: {features_array.shape}")

# Ensure feat is reshaped to match the input format expected by the KDTree query
feat = feat.reshape(1, -1)  # Reshape to a 2D array (1, n_features)
# Trim or adjust feat to match the feature array's dimensionality (521)
if feat.shape[1] > features_array.shape[1]:
    feat = feat[:, :features_array.shape[1]]  # Trim to match 521 dimensions
# Debug: Check reshaped feature vector shape
print(f"Trimmed feature vector shape: {feat.shape}")
# Debug: Check reshaped feature vector shape
print(f"Reshaped feature vector shape: {feat.shape}")

# Query the tree
dist, idx = tree.query(feat, k=5)

print("✅ Closest matches:")
for i in range(len(idx[0])):
    disease_label = labels_array[idx[0][i]]
    disease_name = label_map[disease_label]
    print(f"- {disease_name} (Label ID: {disease_label})")
