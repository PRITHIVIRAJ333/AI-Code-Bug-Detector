import os
import json
import numpy as np
import joblib
import onnxruntime as ort


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.onnx"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "vectorizer.joblib"
)

LABELS_PATH = os.path.join(
    MODEL_DIR,
    "labels.json"
)


session = None
vectorizer = None
labels = {}


def load_model():

    global session
    global vectorizer
    global labels

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.onnx not found."
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "vectorizer.joblib not found."
        )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    if os.path.exists(LABELS_PATH):

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            labels = json.load(file)

    else:

        labels = {
            "0": "No Bug",
            "1": "Bug Detected"
        }


def predict_single(code):

    if session is None:
        load_model()

    features = vectorizer.transform(
        [code]
    )

    input_data = features.toarray().astype(
        np.float32
    )

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {
            input_name: input_data
        }
    )

    probabilities = outputs[0][0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class] * 100
    )

    label = labels.get(
        str(predicted_class),
        "Bug Detected"
        if predicted_class == 1
        else "No Bug"
    )

    return {
        "label": label,
        "confidence": round(
            confidence,
            2
        )
    }


def split_code(code, max_chars=4000):

    lines = code.splitlines()

    chunks = []

    current = []
    current_size = 0

    for line in lines:

        line_size = len(line) + 1

        if (
            current
            and current_size + line_size > max_chars
        ):

            chunks.append(
                "\n".join(current)
            )

            current = []
            current_size = 0

        current.append(line)
        current_size += line_size

    if current:

        chunks.append(
            "\n".join(current)
        )

    return chunks


def predict_large_code(code):

    chunks = split_code(code)

    if not chunks:

        return {
            "label": "No Bug",
            "confidence": 0
        }

    predictions = []

    for chunk in chunks:

        try:

            result = predict_single(
                chunk
            )

            predictions.append(result)

        except Exception:
            continue

    if not predictions:

        return {
            "label": "No Bug",
            "confidence": 0
        }

    bug_results = [
        x for x in predictions
        if x["label"].lower()
        in [
            "bug detected",
            "bug",
            "1"
        ]
    ]

    if bug_results:

        confidence = max(
            x["confidence"]
            for x in bug_results
        )

        return {
            "label": "Bug Detected",
            "confidence": round(
                confidence,
                2
            )
        }

    confidence = max(
        x["confidence"]
        for x in predictions
    )

    return {
        "label": "No Bug",
        "confidence": round(
            confidence,
            2
        )
    }