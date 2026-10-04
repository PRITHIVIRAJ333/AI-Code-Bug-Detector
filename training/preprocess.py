import pandas as pd
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train.csv"


def clean_code(code):
    if pd.isna(code):
        return ""

    code = str(code)

    # Remove excessive whitespace
    code = re.sub(r"\s+", " ", code)

    # Keep useful programming symbols
    code = code.strip()

    return code


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Columns found:", list(df.columns))

    # Detect code column
    possible_code_columns = [
        "code",
        "source",
        "source_code",
        "snippet",
        "text"
    ]

    code_column = None

    for column in possible_code_columns:
        if column in df.columns:
            code_column = column
            break

    if code_column is None:
        raise ValueError(
            "Code column not found. Expected one of: "
            + ", ".join(possible_code_columns)
        )

    # Detect label column
    possible_label_columns = [
        "label",
        "bug",
        "target",
        "class"
    ]

    label_column = None

    for column in possible_label_columns:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError(
            "Label column not found. Expected one of: "
            + ", ".join(possible_label_columns)
        )

    result = pd.DataFrame()

    result["code"] = df[code_column].apply(clean_code)
    result["label"] = df[label_column]

    result = result.dropna()
    result = result[result["code"].str.len() > 5]

    return result


if __name__ == "__main__":
    data = load_dataset()

    print("\nDataset loaded successfully!")
    print("Total samples:", len(data))
    print("\nLabel distribution:")
    print(data["label"].value_counts())

    output_path = BASE_DIR / "dataset" / "processed_train.csv"

    data.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)