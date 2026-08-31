import argparse


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


def main() -> None:
    args = parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
