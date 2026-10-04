import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "processed_train.csv"

MODEL_DIR = BASE_DIR / "backend" / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


class CodeBugModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):
        return self.network(x)


def convert_labels(labels):

    unique_labels = sorted(
        list(
            set(
                str(x)
                for x in labels
            )
        )
    )

    label_to_id = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    encoded = np.array([
        label_to_id[str(label)]
        for label in labels
    ])

    return encoded, label_to_id


def main():

    if not DATA_PATH.exists():

        print(
            "processed_train.csv not found."
        )

        print(
            "First run: python training/preprocess.py"
        )

        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    codes = df["code"].astype(str).values
    labels = df["label"].values

    print(
        "Total training samples:",
        len(codes)
    )

    # Convert labels
    y, label_mapping = convert_labels(labels)

    print(
        "Label mapping:",
        label_mapping
    )

    if len(label_mapping) < 2:

        raise ValueError(
            "Dataset must contain at least 2 classes."
        )

    # TF-IDF feature extraction
    vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=(2, 5),

        max_features=3000,

        lowercase=False,

        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        codes
    ).toarray().astype(
        np.float32
    )

    print(
        "Feature shape:",
        X.shape
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # Create neural network
    model = CodeBugModel(

        input_size=X.shape[1],

        num_classes=len(label_mapping)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=0.001
    )

    epochs = 15

    print("\nStarting Deep Learning training...\n")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        outputs = model(
            X_train
        )

        loss = criterion(
            outputs,
            y_train
        )

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 1 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {loss.item():.4f}"
            )

    # Evaluation
    model.eval()

    with torch.no_grad():

        predictions = model(
            X_test_tensor
        )

        predicted_classes = torch.argmax(
            predictions,
            dim=1
        ).numpy()

    accuracy = accuracy_score(
        y_test,
        predicted_classes
    )

    print("\n============================")
    print("MODEL ACCURACY")
    print("============================")
    print(
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predicted_classes,
            zero_division=0
        )
    )

    # Save vectorizer
    vectorizer_path = (
        MODEL_DIR /
        "vectorizer.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    # Save labels
    labels_path = (
        MODEL_DIR /
        "labels.json"
    )

    id_to_label = {
        str(value): key
        for key, value in label_mapping.items()
    }

    with open(
        labels_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            id_to_label,
            file,
            indent=4
        )

    # Export ONNX
    model.eval()

    dummy_input = torch.randn(
        1,
        X.shape[1],
        dtype=torch.float32
    )

    onnx_path = (
        MODEL_DIR /
        "model.onnx"
    )

    torch.onnx.export(

        model,

        dummy_input,

        onnx_path,

        input_names=[
            "input"
        ],

        output_names=[
            "output"
        ],

        dynamic_axes={
            "input": {
                0: "batch"
            },

            "output": {
                0: "batch"
            }
        },

        opset_version=17
    )

    print("\n============================")
    print("TRAINING COMPLETE")
    print("============================")

    print(
        "ONNX model:",
        onnx_path
    )

    print(
        "Vectorizer:",
        vectorizer_path
    )

    print(
        "Labels:",
        labels_path
    )


if __name__ == "__main__":
    main()