"""
Converts csv to line-oriented json
"""
import argparse
import logging
import sys
import pandas as pd

# log to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("tojson")

def main():
    """
    Converts csv to line-oriented json
    Defaults input to stdin and output to stdout
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=argparse.FileType("r"), default=sys.stdin, help="Path to the CSV file")
    parser.add_argument("--output", type=argparse.FileType("w", encoding="utf-8"), default=sys.stdout, help="Path to the output file")
    args = parser.parse_args()
    logger.info(f"Converting csv to line-oriented json from {args.input} to {args.output}")

    df = pd.read_csv(args.input)
    df.to_json(args.output, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()