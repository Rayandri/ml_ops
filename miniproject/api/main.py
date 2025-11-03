from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict():
    return {"y_pred": 2}


class HouseFeatures(BaseModel):
    size: float
    bedrooms: int
    garden: bool


@app.post("/predict")
def predict_post(features: HouseFeatures):
    return {"y_pred": 2}


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )



