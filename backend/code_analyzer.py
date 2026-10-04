import ast
import re


def analyze_python(code):

    issues = []

    lines = code.splitlines()

    # Syntax checking
    try:

        tree = ast.parse(code)

    except SyntaxError as error:

        line_number = (
            error.lineno
            if error.lineno
            else 1
        )

        issues.append({
            "type": "Syntax Error",
            "severity": "Critical",
            "line": line_number,
            "message": str(error.msg),
            "suggestion":
                "Check brackets, indentation, "
                "quotes and Python syntax."
        })

        return issues

    # Division by zero pattern
    for index, line in enumerate(
        lines,
        start=1
    ):

        if re.search(
            r"/\s*0\b",
            line
        ):

            issues.append({
                "type": "Division by Zero",
                "severity": "High",
                "line": index,
                "message":
                    "The code directly divides by zero.",
                "suggestion":
                    "Check the denominator before division."
            })

    # Dangerous eval
    for index, line in enumerate(
        lines,
        start=1
    ):

        if re.search(
            r"\beval\s*\(",
            line
        ):

            issues.append({
                "type": "Unsafe eval",
                "severity": "High",
                "line": index,
                "message":
                    "eval() can execute dynamically supplied code.",
                "suggestion":
                    "Avoid eval() and use safer parsing methods."
            })

    # Hardcoded password
    for index, line in enumerate(
        lines,
        start=1
    ):

        if re.search(
            r"(password|passwd|secret)\s*=",
            line,
            re.IGNORECASE
        ):

            issues.append({
                "type": "Hardcoded Secret",
                "severity": "High",
                "line": index,
                "message":
                    "A possible secret is hardcoded.",
                "suggestion":
                    "Use environment variables instead."
            })

    # Bare except
    for index, line in enumerate(
        lines,
        start=1
    ):

        if re.match(
            r"\s*except\s*:",
            line
        ):

            issues.append({
                "type": "Bare Exception",
                "severity": "Medium",
                "line": index,
                "message":
                    "Bare except catches every exception.",
                "suggestion":
                    "Catch the specific exception type."
            })

    return issues


def analyze_general_code(code):

    issues = []

    lines = code.splitlines()

    for index, line in enumerate(
        lines,
        start=1
    ):

        if "TODO" in line.upper():

            issues.append({
                "type": "Incomplete Code",
                "severity": "Low",
                "line": index,
                "message":
                    "TODO comment found.",
                "suggestion":
                    "Complete the pending implementation."
            })

    return issues


def analyze_code(code, language="python"):

    if language.lower() in [
        "python",
        "py"
    ]:

        return analyze_python(code)

    return analyze_general_code(code)