from pathlib import Path
import time

from loguru import logger
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


DATA_DIR = "data"
PLAYERS_FILE = "players.csv"
SALARIES_FILE = "salaries_1985to2018.csv"
STATS_FILE = "Seasons_Stats.csv"
PARQUET_DIR = "artifacts/parquet_out"
TOP_N = 5
SHOW_COUNT = 1_000


def get_spark():
    logger.info("Starting Spark session")
    return (
        SparkSession.builder
        .appName("nba-efficiency")
        .getOrCreate()
    )


def load_data(spark, base_dir):
    logger.info(f"Loading data from {base_dir}")
    base = Path(base_dir)

    players = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(base / PLAYERS_FILE))
    )

    salaries = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(base / SALARIES_FILE))
    )

    stats = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(base / STATS_FILE))
    )

    logger.info(
        f"Loaded players={players.count()}, "
        f"salaries={salaries.count()}, stats={stats.count()}"
    )
    return players, salaries, stats


def prepare_dataframe(players, salaries, stats):
    logger.info("Preparing dataframe")

    stats = (
        stats
        .withColumn("PTS", F.coalesce(F.col("PTS"), F.lit(0.0)))
        .withColumn("TRB", F.coalesce(F.col("TRB"), F.lit(0.0)))
        .withColumn("AST", F.coalesce(F.col("AST"), F.lit(0.0)))
        .withColumn("Year", F.col("Year").cast("int"))
    )

    salaries = salaries.withColumn(
        "season_end", F.col("season_end").cast("int")
    )

    stats = stats.withColumn(
        "efficiency",
        F.col("PTS") + F.col("TRB") + F.col("AST"),
    )

    players_small = players.select("_id", "name")

    stats_with_id = (
        stats.join(players_small, stats.Player == players_small.name, "left")
        .withColumnRenamed("_id", "player_id")
    )

    salaries_clean = salaries.select(
        F.col("player_id").alias("salary_player_id"),
        "season_end",
        "salary",
    )

    joined = stats_with_id.join(
        salaries_clean,
        (stats_with_id.player_id == F.col("salary_player_id"))
        & (stats_with_id.Year == F.col("season_end")),
        "inner",
    )

    result = joined.withColumn(
        "cost_per_eff",
        F.when(F.col("efficiency") != 0,
               F.col("salary") / F.col("efficiency"))
    ).select(
        F.col("player_id"),
        F.col("Player"),
        F.col("Year"),
        F.col("salary"),
        F.col("efficiency"),
        F.col("cost_per_eff"),
    )

    return result


def write_parquet(df, out_dir):
    logger.info(f"Writing parquet to {out_dir} partitioned by Year")
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    (
        df.write
        .mode("overwrite")
        .partitionBy("Year")
        .parquet(str(path))
    )




def top_players(df, top_n):
    logger.info(f"Selecting top {top_n} players per Year by cost_per_eff")
    window = Window.partitionBy("Year").orderBy("cost_per_eff")
    ranked = df.withColumn("rank", F.row_number().over(window))
    return ranked.where(F.col("rank") <= top_n)


def main():
    start = time.perf_counter()
    logger.info("Pipeline started")

    spark = get_spark()

    players, salaries, stats = load_data(spark, DATA_DIR)
    df = prepare_dataframe(players, salaries, stats)
    write_parquet(df, PARQUET_DIR)

    df_parquet = spark.read.parquet(str(Path(PARQUET_DIR)))
    top_df = top_players(df_parquet, TOP_N)

    (
        top_df.select("Year", "player_id", "Player", "cost_per_eff")
        .orderBy("Year", "cost_per_eff")
        .show(SHOW_COUNT, truncate=False)
    )

    spark.stop()
    elapsed = time.perf_counter() - start
    logger.info(f"Pipeline finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

