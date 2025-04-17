import os
import glob
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KDTree

# Constants
IMG_SIZE = (128, 128)

# ------------------------------
# Feature Extraction Function
# ------------------------------
def extract_features(image):
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Color Histogram (RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # LBP Texture
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 11),
                                 range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    return np.hstack([hist, lbp_hist])


# ------------------------------
# Dataset Loader + KDTree Builder
# ------------------------------
def load_dataset_and_build_kdtree(dataset_path):
    features_list = []
    labels_list = []
    label_map = {}
    label_names = sorted(os.listdir(dataset_path))

    for idx, disease_folder in enumerate(label_names):
        full_path = os.path.join(dataset_path, disease_folder)
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

    if len(features_array) > 0:
        tree = KDTree(features_array)
        print(f"✅ KD-Tree built with {features_array.shape[0]} feature vectors.")
        return tree, features_array, labels_array, label_map
    else:
        print("⚠️ No features extracted. Please check your image folder paths and formats.")
        return None, None, None, None
