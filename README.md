## Mini README

### Objectif
Service FastAPI de prédiction avec déploiement canary. Les modèles sont versionnés dans MLflow. À l’amorçage, `current` et `next` pointent sur `models:/iris_logreg/1`.

### Arborescence
- **canary/**: API FastAPI (`main.py`) avec endpoints `/predict`, `/update-model`, `/accept-next-model`, `/status`, `/set-p`.
- **scripts/**: utilitaires, dont `e2e_canary.sh` pour un test bout‑à‑bout local.
- **ml_transpiler_folder/**: exemples d’entraînement et utilitaires liés au transpiler.
- **requirements.txt**: dépendances Python.
- **.venv/**, **mlruns/**, **mlflow.db**: générés localement lors des essais.

### Démarrage rapide (local, venv isolé)
```bash
cd /home/rayan/epita/majeur/mlops
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Lancer MLflow localement
```bash
./.venv/bin/mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////home/rayan/epita/majeur/mlops/mlflow.db \
  --default-artifact-root file:///home/rayan/epita/majeur/mlops/mlruns
```

### Créer un modèle d’exemple (iris v1)
```python
import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X,y = datasets.load_iris(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y,random_state=42,test_size=0.2)
with mlflow.start_run(run_name="iris_v1"):
    model = LogisticRegression(max_iter=200)
    model.fit(Xtr,ytr)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="iris_logreg")
```

### Démarrer l’API
```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  ./
.venv/bin/uvicorn canary.main:app --host 127.0.0.1 --port 8000
```

### Appels utiles
```bash
curl -s http://127.0.0.1:8000/status
curl -s -X POST http://127.0.0.1:8000/set-p -H 'Content-Type: application/json' -d '{"p":0.8}'
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"data":[[5.1,3.5,1.4,0.2],[6.2,3.4,5.4,2.3]]}'
curl -s -X POST http://127.0.0.1:8000/update-model -H 'Content-Type: application/json' -d '{"uri":"models:/iris_logreg/2"}'
curl -s -X POST http://127.0.0.1:8000/accept-next-model
```

### Notes
- Le routage canary utilise une probabilité `p` pour `current` et `1-p` pour `next`.
- `MLFLOW_TRACKING_URI` doit pointer vers le serveur MLflow utilisé par `models:/...`.

