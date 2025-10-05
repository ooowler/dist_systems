import argparse
from typing import List

from pyspark.sql import SparkSession


def compute_fibonacci(n: int) -> int:
    sequence: List[int] = [0, 1]
    if n < 2:
        return sequence[n]
    for _ in range(2, n + 1):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute N-th Fibonacci.")
    parser.add_argument("--n", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("Fibonacci").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"👉🏻 F({args.n}) = {compute_fibonacci(args.n)}")
    spark.stop()


if __name__ == "__main__":
    main()

