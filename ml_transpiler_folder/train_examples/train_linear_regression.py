from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


def main() -> None:
    data = load_diabetes()
    model = LinearRegression()
    model.fit(data.data, data.target)
    output = Path(__file__).resolve().parent / "linear_regression.joblib"
    joblib.dump(model, output)
    print(output)


if __name__ == "__main__":
    main()

