from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Classifier Microservice")

X, y = make_classification(
    n_samples=2000, n_features=8, n_informative=6,
    n_redundant=2, n_clusters_per_class=1, n_classes=2, random_state=123
)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=150, random_state=42)
model.fit(X_scaled, y)


class Features(BaseModel):
    param_1: float
    param_2: float
    param_3: float
    param_4: float
    param_5: float
    param_6: float
    param_7: float
    param_8: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: Features):
    x = np.array([[
        features.param_1, features.param_2, features.param_3, features.param_4,
        features.param_5, features.param_6, features.param_7, features.param_8
    ]])
    x_scaled = scaler.transform(x)
    prediction = int(model.predict(x_scaled)[0])
    probability = float(model.predict_proba(x_scaled)[0][prediction])
    return {"prediction": prediction, "probability": round(probability, 4)}
