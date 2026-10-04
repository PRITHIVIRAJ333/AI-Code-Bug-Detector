import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR /
    "backend" /
    "model"
)


MODEL_PATH = (
    MODEL_DIR /
    "model.onnx"
)

VECTORIZER_PATH = (
    MODEL_DIR /
    "vectorizer.joblib"
)

LABELS_PATH = (
    MODEL_DIR /
    "labels.json"
)


_session = None
_vectorizer = None
_labels = None


def load_model():

    global _session
    global _vectorizer
    global _labels

    if _session is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "AI model not found. "
            "Run training first."
        )

    _session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    _vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        _labels = json.load(file)


def predict_code(code):

    load_model()

    features = _vectorizer.transform(
        [code]
    ).toarray().astype(
        np.float32
    )

    input_name = (
        _session
        .get_inputs()[0]
        .name
    )

    output = _session.run(
        None,
        {
            input_name: features
        }
    )[0][0]

    # Softmax
    exp_values = np.exp(
        output - np.max(output)
    )

    probabilities = (
        exp_values /
        np.sum(exp_values)
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = _labels.get(
        str(predicted_index),
        str(predicted_index)
    )

    confidence = float(
        probabilities[predicted_index]
        * 100
    )

    return {
        "label": predicted_label,
        "confidence": round(
            confidence,
            2
        )
    }