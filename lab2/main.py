
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

DEFAULT_CSV_PATH = "file:///app/lab2_data.csv"
OUT_DIR = "/app/out"

SCHEMA = T.StructType([
    T.StructField("InvoiceNo",   T.StringType(),  True),
    T.StructField("StockCode",   T.StringType(),  True),
    T.StructField("Description", T.StringType(),  True),
    T.StructField("Quantity",    T.IntegerType(), True),
    T.StructField("InvoiceDate", T.StringType(),  True),
    T.StructField("UnitPrice",   T.DoubleType(),  True),
    T.StructField("CustomerID",  T.IntegerType(), True),
    T.StructField("Country",     T.StringType(),  True),
])


def print_cluster_info(spark: SparkSession) -> None:
    sc = spark.sparkContext
    print("=== Spark environment ===")
    print("Master            :", sc.master)
    print("AppName           :", sc.appName)
    print("DefaultParallelism:", sc.defaultParallelism)
    conf = dict(sc.getConf().getAll())
    print("Driver memory     :", conf.get("spark.driver.memory", "n/a"))
    print("Executor memory   :", conf.get("spark.executor.memory", "n/a"))
    try:
        infos = sc._jsc.sc().statusTracker().getExecutorInfos()
        print("Executors count  :", infos.size())
        for i in range(infos.size()):
            info = infos[i]
            print(f"  - executorId={info.executorId()}, host={info.host()}, cores={info.totalCores()}")
    except Exception:
        pass
    print()


def resolve_input_path() -> str:
    env_path = os.getenv("INPUT_PATH")
    if env_path:
        if "://" in env_path: 
            return env_path
        if env_path.startswith("/"):
            return f"file://{env_path}"
        return f"file://{os.path.abspath(env_path)}"
    return DEFAULT_CSV_PATH


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    df_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .schema(SCHEMA)
        .csv(path)
    )
    df = df_raw.filter((F.col("Quantity") > 0) & (F.col("UnitPrice") > 0))
    return df


def compute_top_products(df: DataFrame, top_n: int = 5) -> DataFrame:
    return (
        df.groupBy("Description")
          .agg(F.sum("Quantity").alias("units_sold"))
          .orderBy(F.desc("units_sold"), F.asc("Description"))
          .limit(top_n)
    )


def compute_customer_metrics(df: DataFrame) -> DataFrame:
    df_lines = df.withColumn("LineTotal", F.col("Quantity") * F.col("UnitPrice"))

    orders_per_customer = (
        df_lines.groupBy("CustomerID")
        .agg(F.countDistinct("InvoiceNo").alias("orders_count"))
    )

    spent_per_customer = (
        df_lines.groupBy("CustomerID")
        .agg(F.round(F.sum("LineTotal"), 2).alias("total_spent"))
    )

    invoice_totals = (
        df_lines.groupBy("CustomerID", "InvoiceNo")
        .agg(F.sum("LineTotal").alias("invoice_total"))
    )

    avg_check_per_customer = (
        invoice_totals.groupBy("CustomerID")
        .agg(F.round(F.avg("invoice_total"), 2).alias("avg_check"))
    )

    return (
        orders_per_customer
        .join(spent_per_customer, on="CustomerID", how="inner")
        .join(avg_check_per_customer, on="CustomerID", how="inner")
        .orderBy("CustomerID")
    )


def save_csv(df: DataFrame, path: str) -> None:
    (
        df.write
          .mode("overwrite")
          .option("header", True)
          .csv(path)
    )


def main() -> None:
    os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Duser.name=spark -Duser.home=/tmp")
    os.environ.setdefault("HADOOP_USER_NAME", "spark")

    spark = (
        SparkSession.builder
        .appName("Lab2-Sales-Analytics")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print_cluster_info(spark)

    input_path = resolve_input_path()
    print(f"📥 Reading: {input_path}")

    df = read_csv(spark, input_path)

    top5 = compute_top_products(df, top_n=5)
    print("🏆 Top-5 товаров по количеству проданных единиц:")
    for row in top5.collect():
        print(f"- {row['Description']}: {row['units_sold']}")

    customers = compute_customer_metrics(df)
    print("\n👤 Метрики по клиентам (первые 10 строк):")
    for row in customers.limit(10).collect():
        print(
            f"CustomerID={row['CustomerID']}, "
            f"orders_count={row['orders_count']}, "
            f"total_spent={row['total_spent']}, "
            f"avg_check={row['avg_check']}"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    save_csv(top5, os.path.join(OUT_DIR, "top_products.csv"))
    save_csv(customers, os.path.join(OUT_DIR, "customers.csv"))

    print(f"\n💾 Saved to: {OUT_DIR}/top_products.csv and {OUT_DIR}/customers.csv")
    print("    (папки с part-*.csv файлами)")

    spark.stop()


if __name__ == "__main__":
    main()

