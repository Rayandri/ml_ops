#!/usr/bin/env bash
set -euo pipefail

cd /home/rayan/epita/majeur/mlops
if [ ! -d .venv ]; then python3 -m venv .venv; fi
./.venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
./.venv/bin/pip install -r requirements.txt >/dev/null 2>&1

mkdir -p mlruns
./.venv/bin/mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////home/rayan/epita/majeur/mlops/mlflow.db \
  --default-artifact-root file:///home/rayan/epita/majeur/mlops/mlruns \
  --workers 1 --gunicorn-opts "--log-level warning" &
MLFLOW_PID=$!

for i in {1..60}; do code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/ || true); [ "$code" != "000" ] && break; sleep 0.5; done

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 ./.venv/bin/python - <<'PY'
import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X, y = datasets.load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42, test_size=0.2)
with mlflow.start_run(run_name="iris_v1"):
    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, ytr)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="iris_logreg")
PY

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 ./.venv/bin/uvicorn canary.main:app --host 127.0.0.1 --port 8000 --workers 1 --log-level warning &
API_PID=$!

for i in {1..60}; do code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/status || true); [ "$code" = "200" ] && break; sleep 0.5; done

curl -s http://127.0.0.1:8000/status || true
curl -s -X POST http://127.0.0.1:8000/set-p -H "Content-Type: application/json" -d '{"p":0.8}' || true
curl -s http://127.0.0.1:8000/status || true
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"data":[[5.1,3.5,1.4,0.2],[6.2,3.4,5.4,2.3]]}' || true

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 ./.venv/bin/python - <<'PY'
import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
X, y = datasets.load_iris(return_X_y=True)
with mlflow.start_run(run_name="iris_v2"):
    model = LogisticRegression(max_iter=300, C=0.5)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="iris_logreg")
PY

curl -s -X POST http://127.0.0.1:8000/update-model -H "Content-Type: application/json" -d '{"uri":"models:/iris_logreg/2"}' || true
curl -s http://127.0.0.1:8000/status || true
curl -s -X POST http://127.0.0.1:8000/accept-next-model || true
curl -s http://127.0.0.1:8000/status || true

kill $API_PID >/dev/null 2>&1 || true
kill $MLFLOW_PID >/dev/null 2>&1 || true

