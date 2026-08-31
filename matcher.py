import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSIFIER_PATH = DATA_DIR / "classifier.csv"
LABELED_SAMPLE_PATH = DATA_DIR / "labeled_sample.csv"

CLASSIFIER_COLUMNS = {
    "Код": "code",
    "Наименование должности по классификатору": "position",
}

INPUT_COLUMNS = {
    "id": "id",
    "Исходное наименование должности": "raw_name",
}

LABELED_COLUMNS = {
    "id": "id",
    "Исходное наименование должности": "raw_name",
    "Правильный код": "expected_code",
    "Наименование по классификатору": "expected_position",
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


def calculate_matches(
    raw_positions: pd.DataFrame,
    classifier: pd.DataFrame,
) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
    )

    classifier_matrix = vectorizer.fit_transform(classifier["normalized_position"])

    raw_matrix = vectorizer.transform(raw_positions["normalized_name"])

    similarities = cosine_similarity(
        raw_matrix,
        classifier_matrix,
    )

    matches = []

    for row_index, scores in enumerate(similarities):
        top_indices = scores.argsort()[-2:][::-1]

        best_index = top_indices[0]
        second_index = top_indices[1]

        best_score = scores[best_index]
        second_score = scores[second_index]

        matches.append(
            {
                "id": raw_positions.iloc[row_index]["id"],
                "raw_name": raw_positions.iloc[row_index]["raw_name"],
                "normalized_name": raw_positions.iloc[row_index]["normalized_name"],
                "best_code": classifier.iloc[best_index]["code"],
                "best_position": classifier.iloc[best_index]["position"],
                "best_score": best_score,
                "second_position": classifier.iloc[second_index]["position"],
                "second_score": second_score,
                "margin": best_score - second_score,
            }
        )

    return pd.DataFrame(matches)


def evaluate(
    classifier: pd.DataFrame,
) -> None:
    labeled_sample = load_csv(
        LABELED_SAMPLE_PATH,
        LABELED_COLUMNS,
    )

    labeled_sample["normalized_name"] = labeled_sample["raw_name"].apply(
        normalize_position
    )

    matches = calculate_matches(
        labeled_sample,
        classifier,
    )

    evaluation = labeled_sample[
        [
            "id",
            "expected_code",
            "expected_position",
        ]
    ].merge(
        matches,
        on="id",
        how="inner",
    )

    evaluation["is_correct"] = evaluation["best_code"].astype(str) == evaluation[
        "expected_code"
    ].astype(str)

    accuracy = evaluation["is_correct"].mean()

    correct = evaluation[evaluation["is_correct"]]
    errors = evaluation[~evaluation["is_correct"]]

    print()
    print("Evaluation:")
    print(f"Labeled sample size: {len(evaluation)}")
    print(f"Correct matches: {len(correct)}")
    print(f"Errors: {len(errors)}")
    print(f"Top-1 accuracy: {accuracy:.2%}")

    print()
    print("Correct match score statistics:")
    print(
        correct[
            [
                "best_score",
                "margin",
            ]
        ]
        .describe()
        .to_string()
    )

    print()
    print("Lowest-confidence correct matches:")
    print(
        correct[
            [
                "raw_name",
                "best_position",
                "best_score",
                "second_position",
                "second_score",
                "margin",
            ]
        ]
        .sort_values(
            [
                "best_score",
                "margin",
            ]
        )
        .head(10)
        .to_string(index=False)
    )

    if not errors.empty:
        print()
        print("Errors:")
        print(
            errors[
                [
                    "raw_name",
                    "expected_position",
                    "best_position",
                    "best_score",
                    "second_position",
                    "second_score",
                    "margin",
                ]
            ].to_string(index=False)
        )


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

    matches = calculate_matches(
        raw_positions,
        classifier,
    )

    print(f"Loaded {len(raw_positions)} raw positions.")
    print(f"Loaded {len(classifier)} classifier positions.")

    print(
        matches[
            [
                "raw_name",
                "best_position",
                "best_score",
                "second_position",
                "second_score",
                "margin",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    evaluate(classifier)


if __name__ == "__main__":
    main()
