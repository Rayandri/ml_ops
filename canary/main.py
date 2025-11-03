from threading import RLock
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow, numpy as np, random

app = FastAPI()

STARTUP_URI = "models:/iris_logreg/1"

class PredictRequest(BaseModel):
    data: list[list[float]]

class UpdateModelRequest(BaseModel):
    uri: str

class SetPRequest(BaseModel):
    p: float

lock = RLock()
current_model = mlflow.pyfunc.load_model(STARTUP_URI)
next_model = current_model
current_uri = STARTUP_URI
next_uri = STARTUP_URI
p = 0.5

@app.post("/predict")
def predict(body: PredictRequest):
    x = np.array(body.data)
    with lock:
        model = current_model if random.random() < p else next_model
    y = model.predict(x).tolist()
    return {"prediction": y}

@app.post("/update-model")
def update_model(body: UpdateModelRequest):
    global next_model, next_uri
    model = mlflow.pyfunc.load_model(body.uri)
    with lock:
        next_model = model
        next_uri = body.uri
    return {"next_model_updated": next_uri}

@app.post("/accept-next-model")
def accept_next_model():
    global current_model, next_model, current_uri, next_uri
    with lock:
        current_model = next_model
        current_uri = next_uri
    return {"accepted": True, "current_uri": current_uri}

@app.get("/status")
def status():
    with lock:
        return {"p": p, "current_uri": current_uri, "next_uri": next_uri}

@app.post("/set-p")
def set_p(body: SetPRequest):
    global p
    pv = float(body.p)
    if pv < 0.0:
        pv = 0.0
    if pv > 1.0:
        pv = 1.0
    with lock:
        p = pv
    return {"p": p}
