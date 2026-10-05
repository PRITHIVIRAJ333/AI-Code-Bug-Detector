
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

from flask_cors import CORS

import os

from code_analyzer import (
    analyze_python_code
)

from bug_detector import (
    predict_large_code
)


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


# ==========================================
# FLASK APP
# ==========================================

app = Flask(
    __name__
)

CORS(app)


# ==========================================
# FRONTEND
# ==========================================

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


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "success": True,

        "message":
        "AI Code Bug Detector is running.",

        "language":
        "Python"

    })


# ==========================================
# ANALYZE PYTHON CODE
# ==========================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                "No data received."

            }), 400


        code = data.get(
            "code",
            ""
        )


        if not isinstance(
            code,
            str
        ):

            return jsonify({

                "success": False,

                "error":
                "Code must be text."

            }), 400


        if not code.strip():

            return jsonify({

                "success": False,

                "error":
                "Please enter Python code."

            }), 400


        # ==================================
        # STATIC ANALYSIS
        # ==================================

        all_issues = analyze_python_code(
            code
        )


        # ==================================
        # ONLY REAL PROGRAMMING ERRORS
        # ==================================

        real_error_types = {

            "SyntaxError",
            "IndentationError",
            "ZeroDivisionError",
            "IndexError",
            "KeyError",
            "TypeError",
            "AttributeError",
            "NameError",
            "ValueError",
            "ImportError"

        }


        actual_errors = []

        warnings = []


        for issue in all_issues:

            error_type = issue.get(
                "type",
                ""
            )


            if error_type in real_error_types:

                actual_errors.append(
                    issue
                )

            else:

                warnings.append(
                    issue
                )


        # ==================================
        # REMOVE DUPLICATES
        # ==================================

        unique_errors = []

        seen = set()


        for issue in actual_errors:

            key = (
                issue["line"],
                issue["type"]
            )

            if key not in seen:

                seen.add(key)

                unique_errors.append(
                    issue
                )


        actual_errors = unique_errors


        # ==================================
        # FINAL PREDICTION
        # ==================================

        if len(actual_errors) > 0:

            label = "Bug Detected"

            issue_count = len(
                actual_errors
            )


            # Risk calculation

            risk_score = min(
                100,
                issue_count * 20
            )


            if risk_score >= 70:

                risk_level = "High"

            elif risk_score >= 40:

                risk_level = "Medium"

            else:

                risk_level = "Low"


            confidence = min(
                99,
                90 + issue_count
            )


        else:

            label = "No Bug"

            issue_count = 0

            risk_score = 0

            risk_level = "Safe"

            confidence = 96


            # IMPORTANT
            # No actual error means
            # issues must be empty in
            # final response.

            actual_errors = []


        # ==================================
        # AI MODEL
        # ==================================
        #
        # Model prediction is only
        # supporting information.
        #
        # It DOES NOT decide the final
        # Bug / No Bug result.
        #

        try:

            model_prediction = (
                predict_large_code(code)
            )

        except Exception:

            model_prediction = {

                "label":
                "Unavailable",

                "confidence":
                0

            }


        # ==================================
        # FINAL RESPONSE
        # ==================================

        response = {

            "success": True,

            "prediction": {

                "label":
                label,

                "confidence":
                confidence

            },

            "risk_score":
            risk_score,

            "risk_level":
            risk_level,

            "issue_count":
            issue_count,

            "issues":
            actual_errors,

            "warnings":
            warnings,

            "language":
            "Python",

            "model_prediction":
            model_prediction

        }


        return jsonify(
            response
        )


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
            str(e)

        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )
    print(
        "       AI CODE BUG DETECTOR"
    )
    print(
        "=========================================="
    )
    print(
        "Language : Python"
    )
    print(
        "Server   : http://127.0.0.1:5000"
    )
    print(
        "=========================================="
    )
    print()


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )

