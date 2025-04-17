# import numpy as np
# from sklearn.neighbors import KDTree
# import pickle

# def load_features_and_tree(path='features_tree.pkl'):
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     return data['tree'], data['features'], data['labels'], data['label_map']

import pickle
import numpy as np

def load_kdtree():
    with open('kdtree.pkl', 'rb') as f:
        tree = pickle.load(f)
    return tree

def load_features():
    return np.load('features.npy')

def load_labels():
    return np.load('labels.npy')

def load_label_map():
    with open('label_map.pkl', 'rb') as f:
        return pickle.load(f)
