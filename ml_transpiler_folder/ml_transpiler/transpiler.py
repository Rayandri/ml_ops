import argparse
import subprocess
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def format_float(value: float) -> str:
    return f"{value:.8f}f" if abs(value) >= 1e-9 else "0.0f"


def generate_linear_code(model: LinearRegression) -> str:
    weights = np.asarray(model.coef_).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    n_features = model.n_features_in_

    lines = [
        "#include <stdio.h>\n\n",
        "float predict(const float *features, int n_features) {\n",
        "    (void)n_features;\n",
        f"    float result = {format_float(intercept)};\n",
    ]
    for i, w in enumerate(weights):
        lines.append(f"    result += {format_float(w)} * features[{i}];\n")
    sample_vals = ", ".join(["0.0f"] * n_features)
    lines += [
        "    return result;\n}\n\n",
        "int main(void) {\n",
        f"    float sample[{n_features}] = {{{sample_vals}}};\n",
        f"    float prediction = predict(sample, {n_features});\n",
        '    printf("Prediction: %f\\n", prediction);\n',
        "    return 0;\n}\n",
    ]
    return "".join(lines)


def generate_logistic_code(model: LogisticRegression) -> str:
    if model.classes_.shape[0] != 2:
        raise ValueError("Only binary logistic regression supported")

    weights = np.asarray(model.coef_).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    n_features = model.n_features_in_

    lines = [
        "#include <math.h>\n#include <stdio.h>\n\n",
        "float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }\n\n",
        "float predict_proba(const float *features, int n_features) {\n",
        "    (void)n_features;\n",
        f"    float score = {format_float(intercept)};\n",
    ]
    for i, w in enumerate(weights):
        lines.append(f"    score += {format_float(w)} * features[{i}];\n")
    sample_vals = ", ".join(["0.0f"] * n_features)
    lines += [
        "    return sigmoid(score);\n}\n\n",
        "int predict(const float *features, int n_features) {\n",
        "    return predict_proba(features, n_features) >= 0.5f;\n}\n\n",
        "int main(void) {\n",
        f"    float sample[{n_features}] = {{{sample_vals}}};\n",
        f"    float p = predict_proba(sample, {n_features});\n",
        f"    int y = predict(sample, {n_features});\n",
        '    printf("Probability: %f\\nPrediction: %d\\n", p, y);\n',
        "    return 0;\n}\n",
    ]
    return "".join(lines)


def build_tree_body(tree, node, depth):
    indent = "    " * depth
    if tree.children_left[node] == -1:
        cls = int(np.argmax(tree.value[node]))
        return f"{indent}return {cls};\n"
    feat = tree.feature[node]
    thr = format_float(tree.threshold[node])
    left = build_tree_body(tree, tree.children_left[node], depth + 1)
    right = build_tree_body(tree, tree.children_right[node], depth + 1)
    return (
        f"{indent}if (features[{feat}] <= {thr}) {{\n{left}{indent}}} else {{\n{right}{indent}}}\n"
    )


def generate_tree_code(model: DecisionTreeClassifier) -> str:
    tree = model.tree_
    n_features = model.n_features_in_
    body = build_tree_body(tree, 0, 1)
    sample_vals = ", ".join(["0.0f"] * n_features)
    return (
        "#include <stdio.h>\n\n"
        "int predict_tree(const float *features, int n_features) {\n"
        "    (void)n_features;\n"
        f"{body}"
        "}\n\n"
        "int main(void) {\n"
        f"    float sample[{n_features}] = {{{sample_vals}}};\n"
        f"    int pred = predict_tree(sample, {n_features});\n"
        '    printf("Prediction: %d\\n", pred);\n'
        "    return 0;\n}\n"
    )


def detect_model(model):
    if isinstance(model, LinearRegression):
        return generate_linear_code(model)
    if isinstance(model, LogisticRegression):
        return generate_logistic_code(model)
    if isinstance(model, DecisionTreeClassifier):
        return generate_tree_code(model)
    raise TypeError("Unsupported model type")


def run(model_path, output="model_generated.c", compile_binary=False, binary="model"):
    model = joblib.load(model_path)
    code = detect_model(model)
    Path(output).write_text(code)
    print(f"Generated C code at {output}")
    if compile_binary:
        subprocess.run(["gcc", output, "-lm", "-o", binary], check=True)
        print(f"Compiled binary: {binary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="model_generated.c")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--binary", default="model")
    args = parser.parse_args()
    run(args.model, args.output, args.compile, args.binary)
    pass
