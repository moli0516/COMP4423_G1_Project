"""Training entry point placeholder."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="DeskCheck training skeleton")
    parser.add_argument("--config", required=True)
    parser.parse_args()
    parser.exit(2, "Not implemented: validate/export training data and implement ModelTrainer first.\n")


if __name__ == "__main__":
    main()
