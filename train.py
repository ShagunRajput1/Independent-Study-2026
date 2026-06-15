# train.py
import numpy as np
import csv
import os
from features import mutation_to_vector
from neural_net import NeuralNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_dataset(filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, "variant_summary.txt")
    X, y = [], []
    seen = set()

    print(" Building dataset from ClinVar...")
    with open(filepath, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sig = row.get("ClinicalSignificance", "")
            mutation = row.get("Name", "")

            if "Pathogenic" in sig and "Likely" not in sig:
                label = 1.0
            elif "Benign" in sig and "Likely" not in sig:
                label = 0.0
            else:
                continue

            if mutation in seen:
                continue
            seen.add(mutation)

            vec = mutation_to_vector(mutation)
            if vec is not None:
                X.append(vec)
                y.append(label)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    print(f" Dataset: {len(X)} samples | "
          f"{int(y.sum())} pathogenic | {int((1-y).sum())} benign")
    return X, y

def normalize(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

def train():
    X, y = build_dataset()
    X, mean, std = normalize(X)
    np.savez(os.path.join(BASE_DIR, "norm_params.npz"), mean=mean, std=std)

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test,  y_test  = X[idx[split:]], y[idx[split:]]

    model = NeuralNet(input_size=X.shape[1])

    print("\n  Training...\n")
    for epoch in range(200):
        mask = (np.random.rand(*X_train.shape) > 0.2).astype(float)
        X_dropped = X_train * mask

        model.forward(X_dropped)
        model.backward(X_dropped, y_train, learning_rate=0.005)

        if epoch % 20 == 0:
            loss = model.binary_cross_entropy(model.output, y_train)
            preds = (model.forward(X_test) >= 0.5).astype(int)
            acc = (preds == y_test.astype(int)).mean() * 100
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Test Accuracy: {acc:.1f}%")

    model.save(os.path.join(BASE_DIR, "model_weights.npz"))
    print("\n Training complete!")

if __name__ == "__main__":
    train()