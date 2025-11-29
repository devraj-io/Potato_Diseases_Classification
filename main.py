from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.layers.TFSMLayer("../models/1", call_endpoint="serving_default")

CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_img(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_img(await file.read())
    img_batch = np.expand_dims(image, axis=0).astype(np.float32)

    # Get the raw dictionary output from the TFSMLayer
    prediction_dict = MODEL(img_batch)

    # Extract the prediction tensor (usually under 'serving_default' or similar)
    prediction_values = list(prediction_dict.values())[0].numpy()[0]

    predicted_class = CLASS_NAMES[np.argmax(prediction_values)]
    confidence = float(np.max(prediction_values))

    return {
        "class": predicted_class,
        "confidence": confidence
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)
