import sys
import os
import glob
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KDTree
import pickle

# Make sure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.feature_extraction.extractor import extract_features  # Adjusted for new feature extraction

# Set paths for dataset and save directory
DATASET_PATH = r'C:\Users\aryas\Downloads\IMG_CLASSES'
SAVE_PATH = '.'  # Or 'models/' if you want a separate folder

IMG_SIZE = (128, 128)  # Resize images for uniformity

# 2. Helper Function to Extract Features
# Ensure consistent feature vector size
def extract_features(image):
    # Resize and grayscale
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Color Histogram (RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # LBP Texture Features (Local Binary Pattern)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(),
                                bins=np.arange(0, 10),
                                range=(0, 9))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # Combine both histograms into a single feature vector (make sure this matches the training data shape)
    features = np.hstack([hist, lbp_hist])  # This should give 521 features

    return features



# 3. Load Dataset and Extract Features
def load_and_extract_features():
    features_list = []
    labels_list = []
    label_map = {}
    label_names = sorted(os.listdir(DATASET_PATH))

    for idx, disease_folder in enumerate(label_names):
        full_path = os.path.join(DATASET_PATH, disease_folder)
        if not os.path.isdir(full_path):
            continue

        label_clean = disease_folder.split(' ')[0].strip().lower().replace(' ', '_')
        label_map[idx] = label_clean

        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in glob.glob(os.path.join(full_path, ext)):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    feat = extract_features(img)
                    features_list.append(feat)
                    labels_list.append(idx)
                except Exception as e:
                    print(f"Error with {img_path}: {e}")

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    return features_array, labels_array, label_map


# 4. Build KD-Tree for Feature Matching
def build_kdtree(features_array):
    tree = KDTree(features_array)
    print(f"KD-Tree built with {features_array.shape[0]} feature vectors.")
    return tree


# 5. Main Function for Extracting Features, Building KD-Tree, and Saving Models
def main():
    features_array, labels_array, label_map = load_and_extract_features()

    if len(features_array) > 0:
        # Build KD-Tree
        tree = build_kdtree(features_array)

        # Save KD-Tree and model data
        with open(os.path.join(SAVE_PATH, 'kdtree.pkl'), 'wb') as f:
            pickle.dump(tree, f)

        np.save(os.path.join(SAVE_PATH, 'features.npy'), features_array)
        np.save(os.path.join(SAVE_PATH, 'labels.npy'), labels_array)

        with open(os.path.join(SAVE_PATH, 'label_map.pkl'), 'wb') as f:
            pickle.dump(label_map, f)

        print("📦 KD-Tree, features, and labels saved successfully.")
    else:
        print("⚠️ No features extracted. Please check your image folder paths and formats.")


if __name__ == "__main__":
    main()


# import sys
# import os
# import glob
# import cv2
# import numpy as np
# import pickle
# from sklearn.neighbors import KDTree

# # Make sure custom modules can be imported
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from core.feature_extraction.extractor import extract_features


# DATASET_PATH = r'C:\Users\aryas\OneDrive\Desktop\skin_disease_detector\data\IMG_CLASSES'
# SAVE_PATH = '.'  # or 'models/' if you want a separate folder


# def main():
#     features_list = []
#     labels_list = []
#     label_map = {}
#     label_names = sorted(os.listdir(DATASET_PATH))

#     for idx, disease_folder in enumerate(label_names):
#         full_path = os.path.join(DATASET_PATH, disease_folder)
#         if not os.path.isdir(full_path):
#             continue

#         label_clean = disease_folder.split(' ')[0].strip().lower().replace(' ', '_')
#         label_map[idx] = label_clean

#         for ext in ('*.jpg', '*.jpeg', '*.png'):
#             for img_path in glob.glob(os.path.join(full_path, ext)):
#                 try:
#                     img = cv2.imread(img_path)
#                     if img is None:
#                         continue
#                     feat = extract_features(img)
#                     features_list.append(feat)
#                     labels_list.append(idx)
#                 except Exception as e:
#                     print(f"Error with {img_path}: {e}")

#     features_array = np.array(features_list)
#     labels_array = np.array(labels_list)

#     if len(features_array) > 0:
#         tree = KDTree(features_array)
#         print(f"✅ KD-Tree built with {features_array.shape[0]} feature vectors.")

#         # Save everything
#         with open(os.path.join(SAVE_PATH, 'kdtree.pkl'), 'wb') as f:
#             pickle.dump(tree, f)

#         np.save(os.path.join(SAVE_PATH, 'features.npy'), features_array)
#         np.save(os.path.join(SAVE_PATH, 'labels.npy'), labels_array)

#         with open('kdtree.pkl', 'wb') as f:
#             pickle.dump(tree, f)

#         # # Create label_map dynamically
#         # label_map = {}
#         # for idx, folder_name in enumerate(DATASET_PATH):
#         #     disease_name = folder_name.split(' ')[0].strip()  # Get the disease name from the folder
#         #     label_map[idx] = disease_name

#         with open('label_map.pkl', 'wb') as f:
#             pickle.dump(label_map, f)

#         print("📦 KD-Tree, features, and labels saved successfully.")
#     else:
#         print("⚠️ No features extracted. Please check your image folder paths and formats.")


# if __name__ == "__main__":
#     main()
