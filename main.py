from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import tensorflow as tf
from PIL import Image
import numpy as np
from color_model.color_predictor import predict_combined_colors
from color_model.color_12_predict import get_dominant_color
import os

app = FastAPI()

# Load TensorFlow models
pattern_model = tf.keras.models.load_model("models/pattern_model_3.h5")
type_model = tf.keras.models.load_model("models/fabric_texture_model_3.h5")

def save_temp_image(upload_file: UploadFile) -> str:
    """Save uploaded file temporarily and return path."""
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_path

def preprocess_image(image_path, size=(224, 224)):
    img = Image.open(image_path).convert("RGB").resize(size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict/pattern/")
async def predict_pattern(file: UploadFile = File(...)):
    image_path = save_temp_image(file)
    try:
        img_tensor = preprocess_image(image_path)
        pred = pattern_model.predict(img_tensor)[0]
        label = int(np.argmax(pred))
        confidence = float(np.max(pred))
        LABEL_NAMES = {
            0: "abstract_graphic",
            1: "animals",
            2: "birds",
            3: "checks",
            4: "damasks",
            5: "floral",
            6: "geometric",
            7: "kids",
            8: "leaf_trees",
            9: "plains_textures",
            10: "spots",
            11: "stripes"
        }
        return {
            "fabric_pattern": {
                "label": label,
                "label_name": LABEL_NAMES.get(label, "unknown"),
                "confidence": round(confidence * 100, 2)
            }
        }
    finally:
        os.remove(image_path)

@app.post("/predict/type/")
async def predict_type(file: UploadFile = File(...)):
    image_path = save_temp_image(file)
    try:
        img_tensor = preprocess_image(image_path)
        pred = type_model.predict(img_tensor)[0]
        label = int(np.argmax(pred))
        confidence = float(np.max(pred))
        labels = {
            0: "Acrylic",
            1: "Blended",
            2: "Cotton",
            3: "Denim",
            4: "Fleece",
            5: "Fur",
            6: "Leather",
            7: "Other",
            8: "Polyester",
            9: "Silk",
            10: "Velvet",
            11: "Wool"
        }
        return {
            "fabric_type": {
                "label": label,
                "label_name": labels.get(label, "unknown"),
                "confidence": round(confidence * 100, 2)
            }
        }
    finally:
        os.remove(image_path)

@app.post("/predict/color/")
async def predict_color(file: UploadFile = File(...)):
    image_path = save_temp_image(file)
    try:
        colors = get_dominant_color(image_path)
        return {
            "dominant_colors": colors
        }
    finally:
        os.remove(image_path)
