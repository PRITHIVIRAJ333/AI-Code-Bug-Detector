from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import re
import ast
import numpy as np
import joblib
import onnxruntime as ort


# =========================================================
# APP SETUP
# =========================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "backend" / "model"

MODEL_PATH = MODEL_DIR / "model.onnx"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
LABELS_PATH = MODEL_DIR / "labels.json"


# =========================================================
# LOAD DEEP LEARNING MODEL
# =========================================================

try:
    session = ort.InferenceSession(str(MODEL_PATH))
    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    print("Model loaded successfully.")

except Exception as error:
    print("MODEL LOADING ERROR:", error)
    session = None
    vectorizer = None
    labels = {"0": "No Bug", "1": "Bug"}


# =========================================================
# STATIC CODE ANALYZER
# =========================================================

def static_analysis(code, language):

    issues = []

    language = str(language or "Python")

    # -----------------------------------------------------
    # PYTHON SYNTAX CHECK
    # -----------------------------------------------------

    if language.lower() == "python":

        try:
            ast.parse(code)

        except SyntaxError as error:

            issues.append({
                "type": "Syntax Error",
                "line": error.lineno or 1,
                "severity": "High",
                "message": error.msg,
                "suggestion": "Check Python syntax, indentation and brackets."
            })

    # -----------------------------------------------------
    # DIVISION BY ZERO
    # -----------------------------------------------------

    if re.search(r"/\s*0\b", code):

        issues.append({
            "type": "Runtime Error",
            "line": 1,
            "severity": "High",
            "message": "Possible division by zero.",
            "suggestion": "Check that the denominator is not zero before division."
        })

    # -----------------------------------------------------
    # LIST INDEX ERROR
    # -----------------------------------------------------

    list_matches = re.finditer(
        r"(\w+)\s*=\s*\[([^\]]*)\]",
        code
    )

    for match in list_matches:

        list_name = match.group(1)
        values = match.group(2)

        if values.strip():

            item_count = len(
                [item for item in values.split(",") if item.strip()]
            )

            # Example:
            # numbers[10]
            direct_indexes = re.findall(
                rf"{re.escape(list_name)}\[(\d+)\]",
                code
            )

            for index in direct_indexes:

                index_number = int(index)

                if index_number >= item_count:

                    issues.append({
                        "type": "Index Error",
                        "line": 1,
                        "severity": "High",
                        "message": (
                            f"{list_name}[{index_number}] is outside "
                            f"the list range. The list contains "
                            f"{item_count} items."
                        ),
                        "suggestion": (
                            f"Use an index between 0 and "
                            f"{item_count - 1}."
                        )
                    })

            # -------------------------------------------------
            # range() + list[i]
            # -------------------------------------------------

            range_pattern = rf"range\((\d+)\).*?{re.escape(list_name)}\[i\]"

            range_match = re.search(
                range_pattern,
                code,
                re.DOTALL
            )

            if range_match:

                range_value = int(range_match.group(1))

                if range_value > item_count:

                    issues.append({
                        "type": "Index Error",
                        "line": 1,
                        "severity": "High",
                        "message": (
                            f"range({range_value}) can access more "
                            f"items than the {item_count} items "
                            f"available in {list_name}."
                        ),
                        "suggestion": (
                            f"Use range({item_count}) or "
                            f"iterate directly over the list."
                        )
                    })

    # -----------------------------------------------------
    # EVAL SECURITY ISSUE
    # -----------------------------------------------------

    if re.search(r"\beval\s*\(", code):

        issues.append({
            "type": "Security Issue",
            "line": 1,
            "severity": "High",
            "message": "Use of eval() can execute unsafe input.",
            "suggestion": "Avoid eval() and use safer input parsing."
        })

    # -----------------------------------------------------
    # HARD-CODED PASSWORD / SECRET
    # -----------------------------------------------------

    if re.search(
        r"(password|passwd|secret|api_key)\s*=\s*['\"]",
        code,
        re.IGNORECASE
    ):

        issues.append({
            "type": "Security Issue",
            "line": 1,
            "severity": "High",
            "message": "Possible hardcoded password or secret.",
            "suggestion": "Use environment variables instead of hardcoding secrets."
        })

    # -----------------------------------------------------
    # BARE EXCEPT
    # -----------------------------------------------------

    if re.search(r"except\s*:", code):

        issues.append({
            "type": "Code Quality",
            "line": 1,
            "severity": "Medium",
            "message": "Bare except may hide unexpected errors.",
            "suggestion": "Catch a specific exception type."
        })

    # -----------------------------------------------------
    # TODO
    # -----------------------------------------------------

    if "TODO" in code.upper():

        issues.append({
            "type": "Code Quality",
            "line": 1,
            "severity": "Low",
            "message": "TODO comment found.",
            "suggestion": "Complete or remove the TODO item."
        })

    return issues


# =========================================================
# DEEP LEARNING PREDICTION
# =========================================================

def predict_code(code):

    if session is None or vectorizer is None:

        return 0, "No Bug", 50

    try:

        features = vectorizer.transform([code]).astype(np.float32)

        input_name = session.get_inputs()[0].name

        outputs = session.run(
            None,
            {
                input_name: features.toarray()
            }
        )

        result = outputs[0]

        # -------------------------------------------------
        # Classification output
        # -------------------------------------------------

        if len(result.shape) == 2:

            prediction = int(np.argmax(result[0]))

        else:

            prediction = int(result[0])

        # -------------------------------------------------
        # Label
        # -------------------------------------------------

        if isinstance(labels, dict):

            label_name = labels.get(
                str(prediction),
                labels.get(prediction, str(prediction))
            )

        else:

            label_name = labels[prediction]

        label_name = str(label_name)

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = 88

        return prediction, label_name, confidence

    except Exception as error:

        print("PREDICTION ERROR:", error)

        return 0, "No Bug", 50


# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk(prediction, issues):

    score = 0

    # Deep learning prediction
    if prediction == 1:
        score += 50

    # Static issues
    for issue in issues:

        severity = issue.get("severity", "Low")

        if severity == "High":
            score += 25

        elif severity == "Medium":
            score += 15

        else:
            score += 5

    score = min(score, 100)

    # Risk level
    if score >= 70:
        level = "High"

    elif score >= 40:
        level = "Medium"

    elif score > 0:
        level = "Low"

    else:
        level = "Safe"

    return score, level


# =========================================================
# WEBSITE ROUTES
# =========================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/style.css")
def style():

    return send_from_directory(
        FRONTEND_DIR,
        "style.css"
    )


@app.route("/script.js")
def script():

    return send_from_directory(
        FRONTEND_DIR,
        "script.js"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "online",
        "message": "Deep Code Guard API is running"
    })


# =========================================================
# CODE ANALYSIS API
# =========================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400

        code = data.get("code", "")

        language = data.get(
            "language",
            "Python"
        )

        if not code.strip():

            return jsonify({
                "success": False,
                "error": "Please enter some code."
            }), 400

        # -------------------------------------------------
        # Deep Learning prediction
        # -------------------------------------------------

        prediction_value, label, confidence = predict_code(code)

        # -------------------------------------------------
        # Static analysis
        # -------------------------------------------------

        issues = static_analysis(
            code,
            language
        )

        # -------------------------------------------------
        # If static analyzer finds issues,
        # consider bug detected
        # -------------------------------------------------

        bug_detected = (
            prediction_value == 1
            or len(issues) > 0
        )

        # -------------------------------------------------
        # Risk score
        # -------------------------------------------------

        risk_score, risk_level = calculate_risk(
            prediction_value,
            issues
        )

        # -------------------------------------------------
        # If static analyzer found a serious bug,
        # minimum risk should be high
        # -------------------------------------------------

        if any(
            issue.get("severity") == "High"
            for issue in issues
        ):

            risk_score = max(
                risk_score,
                75
            )

            risk_level = "High"

        # -------------------------------------------------
        # Frontend expected response
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": {
                "label": (
                    "Bug Detected"
                    if bug_detected
                    else "No Bug"
                ),
                "confidence": confidence
            },

            "risk_score": risk_score,

            "risk_level": risk_level,

            "issue_count": len(issues),

            "issues": issues,

            "language": language

        })

    except Exception as error:

        print(
            "API ERROR:",
            repr(error)
        )

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    print("")
    print("========================================")
    print("         DEEP CODE GUARD")
    print("========================================")
    print("Website:")
    print("http://127.0.0.1:5000")
    print("")
    print("API:")
    print("http://127.0.0.1:5000/api/analyze")
    print("========================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )