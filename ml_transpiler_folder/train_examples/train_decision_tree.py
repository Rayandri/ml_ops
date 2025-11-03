from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    data = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(data.data, data.target)
    output = Path(__file__).resolve().parent / "decision_tree.joblib"
    joblib.dump(model, output)
    print(output)


if __name__ == "__main__":
    main()

