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


@app.get("/")
def hello_world():
    return {"message": "Hello World"}

@app.post("/predict", response_model=PredictResponse)
def predict(features: Features):
    row = np.array([getattr(features, name) for name in model.feature_name()])

    # .predict expects a 2d array, hence for single record prediction, we need
    # to reshape it to 2d first
    row = row.reshape(1, -1)
    score = model.predict(row)[0]
    return {"score": score}
