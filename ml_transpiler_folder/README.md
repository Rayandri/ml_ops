# ml_transpiler

## Installation

```bash
git clone git@github.com:Rayandri/ml_ops.git
cd ml_ops/ml_transpiler_folder
```

## Génération du code C

```bash
python -m ml_transpiler.transpiler --model path/to/model.joblib --output model_generated.c
```

## Compilation

```bash
gcc model_generated.c -lm -o model
```

## Exemple de sortie

```
Generated C code at model_generated.c
Compile with: gcc model_generated.c -lm -o model
```
# ml_ops
