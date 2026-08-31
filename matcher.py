import argparse
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

    print(f"Loaded {len(raw_positions)} raw positions.")
    print(f"Loaded {len(classifier)} classifier positions.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
