import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import base64
import re

from main import Net  # reuse your training model

MODEL_PATH = "mnist_cnn.pt"
DEVICE = torch.device("cpu")

# Load model once at startup
model = Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

app = FastAPI(title="MNIST Inference API")


class PredictRequest(BaseModel):
    image_base64: str

def safe_b64decode(data: str) -> bytes:
    # Remove whitespace
    data = data.strip()

    # Remove data:image/...;base64, prefix if present
    data = re.sub(r"^data:image/.+;base64,", "", data)

    # Fix padding
    padding = len(data) % 4
    if padding != 0:
        data += "=" * (4 - padding)

    return base64.b64decode(data, validate=True)


@app.post("/predict")
def predict(req: PredictRequest):
    # Decode base64 image
    try:
        image_bytes = safe_b64decode(req.image_base64)
    except Exception as e:
        return {"error": f"Invalid base64 image: {str(e)}"}
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))

    # Convert to tensor
    image_np = np.asarray(image, dtype=np.float32) / 255.0

    # Invert if background is white
    image_np = 1.0 - image_np

    # MNIST normalization
    image_np = (image_np - 0.1307) / 0.3081

    tensor = torch.from_numpy(image_np)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    tensor = tensor.contiguous().to(DEVICE)


    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()

    return {
        "prediction": pred
    }
