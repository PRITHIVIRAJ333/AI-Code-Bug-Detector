
import ast
import builtins


class PythonBugAnalyzer(ast.NodeVisitor):

    def __init__(self, code):
        self.code = code
        self.issues = []

        self.variables = {}
        self.functions = {}
        self.lists = {}
        self.dictionaries = {}
        self.strings = {}
        self.numbers = {}

    # =========================================
    # ADD ISSUE
    # =========================================

    def add_issue(
        self,
        line,
        error_type,
        message,
        severity="High"
    ):

        issue = {
            "line": int(line) if line else 1,
            "type": error_type,
            "severity": severity,
            "message": message
        }

        for existing in self.issues:

            if (
                existing["line"] == issue["line"]
                and existing["type"] == issue["type"]
            ):
                return

        self.issues.append(issue)

    # =========================================
    # SYNTAX CHECK
    # =========================================

    def check_syntax(self):

        try:

            return ast.parse(
                self.code
            )

        except SyntaxError as e:

            self.add_issue(
                e.lineno or 1,
                "SyntaxError",
                e.msg or "Invalid Python syntax.",
                "High"
            )

            return None

    # =========================================
    # COLLECT SIMPLE VARIABLES
    # =========================================

    def collect_definitions(self, tree):

        for node in ast.walk(tree):

            # Functions
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):

                self.functions[
                    node.name
                ] = len(
                    node.args.args
                )

            # Simple assignments
            if isinstance(
                node,
                ast.Assign
            ):

                for target in node.targets:

                    if not isinstance(
                        target,
                        ast.Name
                    ):
                        continue

                    name = target.id

                    self.variables[
                        name
                    ] = node.lineno

                    value = node.value

                    # List / Tuple
                    if isinstance(
                        value,
                        (ast.List, ast.Tuple)
                    ):

                        self.lists[
                            name
                        ] = len(
                            value.elts
                        )

                    # Dictionary
                    elif isinstance(
                        value,
                        ast.Dict
                    ):

                        keys = set()

                        for key in value.keys:

                            if isinstance(
                                key,
                                ast.Constant
                            ):

                                keys.add(
                                    key.value
                                )

                        self.dictionaries[
                            name
                        ] = keys

                    # String
                    elif isinstance(
                        value,
                        ast.Constant
                    ):

                        if isinstance(
                            value.value,
                            str
                        ):

                            self.strings[
                                name
                            ] = value.value

                        elif isinstance(
                            value.value,
                            (int, float)
                        ):

                            self.numbers[
                                name
                            ] = value.value

    # =========================================
    # ZERO DIVISION
    # =========================================

    def visit_BinOp(self, node):

        if isinstance(
            node.op,
            (
                ast.Div,
                ast.FloorDiv,
                ast.Mod
            )
        ):

            # Example:
            # 10 / 0

            if isinstance(
                node.right,
                ast.Constant
            ):

                if node.right.value == 0:

                    self.add_issue(
                        node.lineno,
                        "ZeroDivisionError",
                        "Division or modulo by zero.",
                        "High"
                    )

            # Example:
            # divisor = 0
            # 10 / divisor

            elif isinstance(
                node.right,
                ast.Name
            ):

                name = node.right.id

                if name in self.numbers:

                    if self.numbers[name] == 0:

                        self.add_issue(
                            node.lineno,
                            "ZeroDivisionError",
                            (
                                f"Variable '{name}' "
                                "contains zero."
                            ),
                            "High"
                        )

        self.generic_visit(node)

    # =========================================
    # DEFINITE LIST INDEX ERROR
    # =========================================

    def check_list_indexes(self, tree):

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Subscript
            ):
                continue

            if not isinstance(
                node.value,
                ast.Name
            ):
                continue

            list_name = node.value.id

            if list_name not in self.lists:
                continue

            length = self.lists[
                list_name
            ]

            index = node.slice

            # Python constant index
            if isinstance(
                index,
                ast.Constant
            ):

                if isinstance(
                    index.value,
                    int
                ):

                    value = index.value

                    if (
                        value >= length
                        or value < -length
                    ):

                        self.add_issue(
                            node.lineno,
                            "IndexError",
                            (
                                f"Index {value} is "
                                f"outside the valid range "
                                f"of '{list_name}'."
                            ),
                            "High"
                        )

    # =========================================
    # DEFINITE DICTIONARY KEY ERROR
    # =========================================

    def check_dictionary_keys(self, tree):

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Subscript
            ):
                continue

            if not isinstance(
                node.value,
                ast.Name
            ):
                continue

            dictionary_name = node.value.id

            if dictionary_name not in self.dictionaries:
                continue

            keys = self.dictionaries[
                dictionary_name
            ]

            key_node = node.slice

            if isinstance(
                key_node,
                ast.Constant
            ):

                key = key_node.value

                if key not in keys:

                    self.add_issue(
                        node.lineno,
                        "KeyError",
                        (
                            f"Key '{key}' does not "
                            f"exist in dictionary "
                            f"'{dictionary_name}'."
                        ),
                        "High"
                    )

    # =========================================
    # DEFINITE INTEGER CONVERSION ERROR
    # =========================================

    def check_integer_conversion(self, tree):

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call
            ):
                continue

            if not isinstance(
                node.func,
                ast.Name
            ):
                continue

            if node.func.id != "int":
                continue

            if len(node.args) != 1:
                continue

            argument = node.args[0]

            if isinstance(
                argument,
                ast.Constant
            ):

                value = argument.value

                if isinstance(
                    value,
                    str
                ):

                    cleaned = value.strip()

                    if not cleaned.lstrip(
                        "+-"
                    ).isdigit():

                        self.add_issue(
                            node.lineno,
                            "ValueError",
                            (
                                f"Cannot convert "
                                f"'{value}' to integer."
                            ),
                            "High"
                        )

    # =========================================
    # DEFINITE STRING ATTRIBUTE ERROR
    # =========================================

    def check_string_attributes(self, tree):

        valid_methods = {
            "upper",
            "lower",
            "strip",
            "split",
            "replace",
            "startswith",
            "endswith",
            "capitalize",
            "title",
            "isdigit",
            "isalpha",
            "isalnum",
            "find",
            "count"
        }

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call
            ):
                continue

            if not isinstance(
                node.func,
                ast.Attribute
            ):
                continue

            object_node = node.func.value

            if not isinstance(
                object_node,
                ast.Name
            ):
                continue

            variable_name = object_node.id

            if variable_name not in self.strings:
                continue

            method_name = node.func.attr

            if method_name not in valid_methods:

                self.add_issue(
                    node.lineno,
                    "AttributeError",
                    (
                        f"String object does not "
                        f"have attribute "
                        f"'{method_name}'."
                    ),
                    "High"
                )

    # =========================================
    # FUNCTION ARGUMENT CHECK
    # =========================================

    def check_function_calls(self, tree):

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call
            ):
                continue

            if not isinstance(
                node.func,
                ast.Name
            ):
                continue

            function_name = node.func.id

            if function_name not in self.functions:
                continue

            expected = self.functions[
                function_name
            ]

            given = len(
                node.args
            )

            if expected != given:

                self.add_issue(
                    node.lineno,
                    "TypeError",
                    (
                        f"Function '{function_name}' "
                        f"expects {expected} argument(s), "
                        f"but {given} given."
                    ),
                    "High"
                )

    # =========================================
    # RANGE + LIST INDEX
    # =========================================

    def check_range_indexes(self, tree):

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.For
            ):
                continue

            if not isinstance(
                node.target,
                ast.Name
            ):
                continue

            if not isinstance(
                node.iter,
                ast.Call
            ):
                continue

            if not isinstance(
                node.iter.func,
                ast.Name
            ):
                continue

            if node.iter.func.id != "range":
                continue

            if len(
                node.iter.args
            ) != 1:
                continue

            range_arg = node.iter.args[0]

            if not isinstance(
                range_arg,
                ast.Constant
            ):
                continue

            if not isinstance(
                range_arg.value,
                int
            ):
                continue

            maximum = range_arg.value

            loop_variable = (
                node.target.id
            )

            for child in ast.walk(node):

                if not isinstance(
                    child,
                    ast.Subscript
                ):
                    continue

                if not isinstance(
                    child.slice,
                    ast.Name
                ):
                    continue

                if (
                    child.slice.id
                    != loop_variable
                ):
                    continue

                if not isinstance(
                    child.value,
                    ast.Name
                ):
                    continue

                list_name = (
                    child.value.id
                )

                if list_name not in self.lists:
                    continue

                length = self.lists[
                    list_name
                ]

                if maximum > length:

                    self.add_issue(
                        child.lineno,
                        "IndexError",
                        (
                            f"Loop can access "
                            f"'{list_name}' outside "
                            "its valid range."
                        ),
                        "High"
                    )

    # =========================================
    # ANALYZE
    # =========================================

    def analyze(self):

        tree = self.check_syntax()

        if tree is None:
            return self.issues

        self.collect_definitions(
            tree
        )

        self.visit(
            tree
        )

        self.check_list_indexes(
            tree
        )

        self.check_dictionary_keys(
            tree
        )

        self.check_integer_conversion(
            tree
        )

        self.check_string_attributes(
            tree
        )

        self.check_function_calls(
            tree
        )

        self.check_range_indexes(
            tree
        )

        self.issues.sort(
            key=lambda item:
            item["line"]
        )

        return self.issues


def analyze_python_code(code):

    if not code or not code.strip():
        return []

    analyzer = PythonBugAnalyzer(
        code
    )

    return analyzer.analyze()

