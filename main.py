import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# Enable CORS to allow requests from your frontend (adjust origins as needed)
origins = [
    "http://localhost",
    "http://localhost:8080",
    "*",  # Caution: This allows all origins. Specify your frontend's origin in production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IMPORTANT: Load your trained model here ---
# Replace 'path/to/your/saved_model.h5' with the actual path to your saved model file.
try:
    model_path = 'models/demo_3_classifier.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define your class labels
CLASS_NAMES = ["abstract_pattern", "animals_pattern", "birds_pattern"]
IMG_HEIGHT = 224
IMG_WIDTH = 224

async def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.get("/")
async def welcome():
     return {
        "message": "Welcome to the Fabric Pattern Classifier API!",
        "how_to_run": "1. Save this code as a Python file (e.g., main.py).",
        "install_dependencies": "2. Install required libraries: pip install fastapi uvicorn tensorflow numpy Pillow python-multipart",
        "run_server": "3. Run the FastAPI server using Uvicorn: uvicorn main:app --reload",
        "predict_endpoint": "4. Send a POST request with an image file to the /predict/ endpoint to classify the fabric pattern.",
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        processed_image = await preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        predicted_probabilities = predictions[0]
        predicted_class_index = np.argmax(predicted_probabilities)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(predicted_probabilities[predicted_class_index])

        class_probabilities = {
            CLASS_NAMES[i]: float(predicted_probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "prediction": predicted_class_name,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
