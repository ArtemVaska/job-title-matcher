import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

CLASSIFIER_PATH = Path(__file__).resolve().parent / "data" / "classifier.csv"

CLASSIFIER_COLUMNS = {
    "Код": "code",
    "Наименование должности по классификатору": "position",
}

INPUT_COLUMNS = {
    "id": "id",
    "Исходное наименование должности": "raw_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match job titles with classifier positions."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results.csv",
        help="Path to the output CSV file. Default: results.csv.",
    )

    return parser.parse_args()


def load_csv(
    path: str | Path,
    columns: dict[str, str],
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    missing_columns = set(columns) - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df.rename(columns=columns)


def normalize_position(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = value.replace("ё", "е")

    # Qualification grades and categories do not affect matching.
    value = re.sub(
        r"\b\d+\s+(?:разряд\w*|категори\w*)\b",
        " ",
        value,
    )
    value = re.sub(
        r"\b(?:разряд\w*|категори\w*)\s+\d+\b",
        " ",
        value,
    )

    # Normalize punctuation.
    value = re.sub(r"[-–—/]", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def main() -> None:
    args = parse_args()

    raw_positions = load_csv(
        args.input,
        INPUT_COLUMNS,
    )

    classifier = load_csv(
        CLASSIFIER_PATH,
        CLASSIFIER_COLUMNS,
    )

    raw_positions["normalized_name"] = raw_positions["raw_name"].apply(
        normalize_position
    )

    classifier["normalized_position"] = classifier["position"].apply(normalize_position)

    print(f"Loaded {len(raw_positions)} raw positions.")
    print(f"Loaded {len(classifier)} classifier positions.")

    print(
        raw_positions[["raw_name", "normalized_name"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
