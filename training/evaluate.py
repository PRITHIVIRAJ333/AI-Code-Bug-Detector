import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR /
    "dataset" /
    "processed_train.csv"
)

MODEL_DIR = (
    BASE_DIR /
    "backend" /
    "model"
)


def main():

    model_path = (
        MODEL_DIR /
        "model.onnx"
    )

    vectorizer_path = (
        MODEL_DIR /
        "vectorizer.joblib"
    )

    labels_path = (
        MODEL_DIR /
        "labels.json"
    )

    if not model_path.exists():

        print(
            "Model not found."
        )

        print(
            "Run train_model.py first."
        )

        return

    df = pd.read_csv(
        DATA_PATH
    )

    vectorizer = joblib.load(
        vectorizer_path
    )

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(file)

    codes = df["code"].astype(str).values

    y = np.array([
        int(x)
        for x in df["label"].values
    ])

    X = vectorizer.transform(
        codes
    ).toarray().astype(
        np.float32
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {
            input_name: X
        }
    )[0]

    predictions = np.argmax(
        output,
        axis=1
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        "Classes:",
        labels
    )


if __name__ == "__main__":
    main()