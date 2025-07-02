import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# Define 12 basic colors (RGB)
COLOR_NAMES = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (139, 69, 19),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "cyan": (0, 255, 255),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "beige": (245, 245, 220),
    "indigo": (75, 0, 130),
    "turquoise": (64, 224, 208),
    "coral": (255, 127, 80),
    "salmon": (250, 128, 114),
    "lavender": (230, 230, 250),
    "magenta": (255, 0, 255),
    "orchid": (218, 112, 214),
    "skyblue": (135, 206, 235),
    "chocolate": (210, 105, 30),
    "khaki": (240, 230, 140)
}


# Euclidean distance between two RGB colors
def closest_color(rgb):
    min_dist = float('inf')
    closest = None
    for name, ref_rgb in COLOR_NAMES.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

# Predict dominant color
def get_dominant_color(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))  # Resize to speed up

    # Reshape image to a list of pixels
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Find the largest cluster
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    # Match to nearest known color
    color_name = closest_color(dominant)
    return color_name
