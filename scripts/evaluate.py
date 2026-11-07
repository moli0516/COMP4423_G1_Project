"""Full-split evaluation entry point placeholder."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="DeskCheck evaluation skeleton")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.parse_args()
    parser.exit(2, "Not implemented: shared inference and evaluators are required.\n")


if __name__ == "__main__":
    main()
