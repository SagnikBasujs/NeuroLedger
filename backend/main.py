from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV file into dataframe
        df = pd.read_csv(file.file)

        # Simulate prediction for 4 classes (e.g., emotions)
        num_classes = 4
        random_probs = np.random.rand(num_classes)
        random_probs /= random_probs.sum()  # normalize to sum = 1
        percentages = (random_probs * 100).round(2).tolist()

        # Map class labels to percentages
        results = {f"Class {i}": percentages[i] for i in range(num_classes)}

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}
