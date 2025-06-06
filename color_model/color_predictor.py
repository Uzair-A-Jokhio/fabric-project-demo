import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from matplotlib import colors as mcolors
from collections import Counter, defaultdict
import json

CSS4_RGB = {
    name: tuple(int(c * 255) for c in mcolors.to_rgb(hex))
    for name, hex in mcolors.CSS4_COLORS.items()
}

def closest_color(requested_color):
    """Find the closest CSS4 color name to an RGB tuple."""
    min_distance = float("inf")
    closest_name = None
    for name, rgb in CSS4_RGB.items():
        distance = sum((component - rc) ** 2 for component, rc in zip(rgb, requested_color))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def predict_combined_colors(image_path, top_n=5):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50))

    pixels = img.reshape((-1, 3))
    total_pixels = len(pixels)

    kmeans = MiniBatchKMeans(n_clusters=top_n, random_state=42)
    kmeans.fit(pixels)

    counts = Counter(kmeans.labels_)
    combined_colors = defaultdict(int)

    for label, count in counts.items():
        rgb = tuple(map(int, kmeans.cluster_centers_[label]))
        color_name = closest_color(rgb)
        combined_colors[color_name] += count

    result = [
        {
            "color": name,
            "percentage": f"{round((count / total_pixels) * 100, 2)}%"
        }
        for name, count in sorted(combined_colors.items(), key=lambda x: x[1], reverse=True)
    ]

    return {"dominant_colors": result}
