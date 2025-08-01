import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
from typing import List
from bson import ObjectId
from sklearn.cluster import KMeans
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.getenv("MODEL_PATH", "./models/fashion_model_improved.h5")
print("Model path exists:", os.path.exists(MODEL_PATH))
model = load_model(MODEL_PATH)

# MongoDB Connection

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB", "fashion_ai_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "clothing_items")

client = MongoClient(
    MONGO_URI,
    server_api=ServerApi('1'),
    tls=True,
    tlsAllowInvalidCertificates=False
)

try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
except Exception as e:
    print("❌ MongoDB connection error:", e)

db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Fashion MNIST labels
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    return img_array


def detect_dominant_color(image_bytes, k=3):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((100, 100))
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)

        not_white = np.logical_not(np.all(pixels > [200, 200, 200], axis=1))
        not_black = np.logical_not(np.all(pixels < [50, 50, 50], axis=1))
        foreground_pixels = pixels[np.logical_and(not_white, not_black)]

        if len(foreground_pixels) == 0:
            foreground_pixels = pixels

        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(foreground_pixels)
        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)].astype(int).tolist()

        return dominant_color
    except Exception as e:
        print(f"Color detection error: {e}")
        return [0, 0, 0]


def extract_clothes_outline(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, mask]
    clothes_png = cv2.merge(rgba)
    _, buffer = cv2.imencode(".png", clothes_png)
    return buffer.tobytes()


@app.post("/predict/")
async def predict_clothing(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        dominant_color = detect_dominant_color(contents)
        processed_image_bytes = extract_clothes_outline(contents)
        processed_img = preprocess_image(processed_image_bytes)
        prediction = model.predict(processed_img)
        predicted_label = labels[np.argmax(prediction)]
        image_base64 = base64.b64encode(processed_image_bytes).decode("utf-8")

        item_data = {
            "filename": file.filename,
            "image_data": image_base64,
            "clothing_type": predicted_label,
            "dominant_color": dominant_color,
            "confidence": float(np.max(prediction))
        }

        inserted = collection.insert_one(item_data)
        item_data["_id"] = str(inserted.inserted_id)
        results.append(item_data)

    return JSONResponse(content={"predictions": results})


@app.get("/items/")
async def get_stored_items():
    items = list(collection.find({}))
    for item in items:
        item["_id"] = str(item["_id"])
    return JSONResponse(content={"items": items})


@app.get("/items/grouped")
async def get_items_grouped():
    pipeline = [
        {"$group": {
            "_id": "$clothing_type",
            "items": {"$push": {
                "_id": {"$toString": "$_id"},
                "filename": "$filename",
                "image_data": "$image_data",
                "dominant_color": "$dominant_color"
            }}
        }}
    ]
    grouped_items = list(collection.aggregate(pipeline))
    return JSONResponse(content={"grouped_items": grouped_items})


@app.delete("/items/{item_id}")
async def delete_item(item_id: str):
    if not ObjectId.is_valid(item_id):
        raise HTTPException(status_code=400, detail="Invalid item ID")
    result = collection.delete_one({"_id": ObjectId(item_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return JSONResponse(content={"message": "Item deleted successfully"})


@app.put("/items/{item_id}")
async def update_item(item_id: str, data: dict = Body(...)):
    update_data = {}

    if "clothing_type" in data:
        update_data["clothing_type"] = data["clothing_type"]


    if not update_data:
        return {"message": "No valid fields to update"}

    result = collection.update_one({"_id": ObjectId(item_id)}, {"$set": update_data})

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"message": "Item updated successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("APP_HOST", "0.0.0.0"), port=int(os.getenv("APP_PORT", 8000)))
