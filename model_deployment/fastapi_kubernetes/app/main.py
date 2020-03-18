from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import lightgbm as lgb

app = FastAPI()
model = lgb.Booster(model_file='/app/model.txt')


class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class PredictResponse(BaseModel):
    score: float


class BatchPredictResponse(BaseModel):
    scores: List[float]


@app.get("/")
def hello_world():
    return {"message": "Hello World"}


@app.post("/predict", response_model=PredictResponse)
def predict(features: Features):
    """Endpoint for getting the score for a single record."""
    row = np.array([getattr(features, name) for name in model.feature_name()])

    # .predict expects a 2d array, hence for single record prediction, we need
    # to reshape it to 2d first
    row = row.reshape(1, -1)
    score = model.predict(row)[0]
    return {"score": score}


@app.post("/batch/predict", response_model=BatchPredictResponse)
def batch_predict(batch_features: List[Features]):
    """Endpoint for getting scores for a batch of record."""
    num_features = len(model.feature_name())
    rows = np.zeros((len(batch_features), num_features))

    for i, features in enumerate(batch_features):
        row = np.array([getattr(features, name) for name in model.feature_name()])
        rows[i] = row

    scores = model.predict(rows).tolist()
    return {"scores": scores}
