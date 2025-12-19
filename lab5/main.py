from pathlib import Path
import time

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructType, StructField
import pandas as pd


EXCEL_FILE = "Player - Salaries per Year (1990 - 2017).xlsx"
CATALOG_NAME = "local"
DATABASE_NAME = "nba"
TABLE_NAME = "player_salaries"
FULL_TABLE_NAME = f"{CATALOG_NAME}.{DATABASE_NAME}.{TABLE_NAME}"
WAREHOUSE_PATH = "/home/iceberg/warehouse"
TARGET_YEAR = 2016
TOP_N = 10


def get_spark():
    spark = (
        SparkSession.builder
        .appName("nba-salaries-iceberg")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(f"spark.sql.catalog.{CATALOG_NAME}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG_NAME}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG_NAME}.warehouse", WAREHOUSE_PATH)
        .config("spark.sql.defaultCatalog", CATALOG_NAME)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_prepare_data(spark):
    logger.info(f"Loading {EXCEL_FILE}")
    
    pdf = pd.read_excel(EXCEL_FILE)
    
    
    df = spark.createDataFrame(pdf, schema=None)
    
    df = (df
        .withColumnRenamed("Player Name", "player_name")
        .withColumnRenamed("Salary in $", "salary_raw")
        .withColumnRenamed("Season Start", "season_start")
        .withColumnRenamed("Season End", "season_end")
        .withColumnRenamed("Team", "team")
        .withColumnRenamed("Full Team Name", "full_team_name")
    )
    
    df = df.withColumn(
        "salary",
        F.regexp_replace(F.col("salary_raw"), "[$,\\s]", "").cast(DoubleType())
    )
    
    df = df.select(
        "player_name",
        "salary",
        F.col("season_start").cast(IntegerType()),
        F.col("season_end").cast(IntegerType()),
        "team",
        "full_team_name"
    )
    
    logger.info(f"Loaded {df.count()} records")
    return df


def create_table(spark):
    logger.info(f"Creating {FULL_TABLE_NAME}")
    
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG_NAME}.{DATABASE_NAME}")
    spark.sql(f"DROP TABLE IF EXISTS {FULL_TABLE_NAME}")
    
    spark.sql(f"""
        CREATE TABLE {FULL_TABLE_NAME} (
            player_name STRING,
            salary DOUBLE,
            season_start INT,
            season_end INT,
            team STRING,
            full_team_name STRING
        ) USING iceberg
        PARTITIONED BY (season_end)
    """)


def merge_year(spark, df, year):
    logger.info(f"Merging year {year}")
    
    df_year = df.filter(F.col("season_end") == year)
    df_year.createOrReplaceTempView("temp_year")
    
    spark.sql(f"""
        MERGE INTO {FULL_TABLE_NAME} t
        USING temp_year s
        ON t.player_name = s.player_name AND t.season_end = s.season_end
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def read_snapshot(spark, snapshot_id):
    return (spark.read
        .option("snapshot-id", snapshot_id)
        .format("iceberg")
        .load(FULL_TABLE_NAME))


def show_time_travel(spark, year):
    logger.info(f"\nTime Travel: viewing data as of year {year}")
    
    snapshots = spark.sql(f"""
        SELECT snapshot_id, committed_at 
        FROM {FULL_TABLE_NAME}.snapshots
        ORDER BY committed_at
    """).collect()
    
    if not snapshots:
        return
    
    target_idx = min(year - 1990, len(snapshots) - 1)
    snapshot_id = snapshots[target_idx].snapshot_id
    
    logger.info(f"Using snapshot {snapshot_id}")
    
    df = read_snapshot(spark, snapshot_id)
    df.filter(F.col("season_end") == year).orderBy(F.col("salary").desc()).show(10, False)


def show_top_earners(spark, year, top_n):
    logger.info(f"\nTop {top_n} earners in {year}")
    
    (spark.table(FULL_TABLE_NAME)
        .filter(F.col("season_end") == year)
        .orderBy(F.col("salary").desc())
        .limit(top_n)
        .select("player_name", "salary", "team", "full_team_name")
        .show(top_n, False))


def show_snapshots(spark):
    logger.info("\nSnapshots history")
    
    (spark.sql(f"""
        SELECT snapshot_id, committed_at, operation
        FROM {FULL_TABLE_NAME}.snapshots
        ORDER BY committed_at
    """).show(100, False))


def demonstrate_growth(spark):
    logger.info("\nTable growth over time")
    
    snapshots = spark.sql(f"""
        SELECT snapshot_id 
        FROM {FULL_TABLE_NAME}.snapshots
        ORDER BY committed_at
    """).collect()
    
    if len(snapshots) < 2:
        return
    
    first = read_snapshot(spark, snapshots[0].snapshot_id).count()
    last = read_snapshot(spark, snapshots[-1].snapshot_id).count()
    
    logger.info(f"First snapshot: {first} records")
    logger.info(f"Last snapshot:  {last} records")
    logger.info(f"Growth: +{last - first} records")


def main():
    start = time.perf_counter()
    logger.info("Pipeline started")
    
    spark = get_spark()
    
    df = load_and_prepare_data(spark)
    
    create_table(spark)
    
    years = [row.season_end for row in df.select("season_end").distinct().orderBy("season_end").collect()]
    logger.info(f"Processing years: {years[0]}-{years[-1]}")
    
    for year in years:
        merge_year(spark, df, year)
    
    show_snapshots(spark)
    show_time_travel(spark, TARGET_YEAR)
    show_top_earners(spark, TARGET_YEAR, TOP_N)
    demonstrate_growth(spark)
    
    spark.stop()
    
    logger.info(f"\nCompleted in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()

