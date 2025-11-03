from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


def main() -> None:
    data = load_breast_cancer()
    model = LogisticRegression(max_iter=5000)
    model.fit(data.data, data.target)
    output = Path(__file__).resolve().parent / "logistic_regression.joblib"
    joblib.dump(model, output)
    print(output)


if __name__ == "__main__":
    main()

