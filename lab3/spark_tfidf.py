from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, regexp_replace, col, desc, length, log, lit
import argparse


def calculate_tfidf_top(spark, input_file, min_length, top_n):
    text = spark.read.text(input_file)
    
    words = text.select(
        explode(split(regexp_replace(lower(col("value")), r"[^\w\s]", ""), r"\s+")).alias("word")
    ).filter(length("word") >= min_length)
    
    tf = words.groupBy("word").count().withColumnRenamed("count", "tf")
    
    total_words = words.count()
    
    idf = tf.withColumn("idf", log(lit(total_words) / col("tf")))
    
    tfidf = idf.withColumn("tfidf", col("tf") * col("idf"))
    
    result = tfidf.orderBy(desc("tfidf")).limit(top_n)
    
    result.show(top_n, truncate=False)
    result.coalesce(1).write.mode("overwrite").csv("/app/output/tfidf_results", header=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    
    spark = SparkSession.builder.appName("TF-IDF").master("local[*]").getOrCreate()
    
    calculate_tfidf_top(spark, args.input, args.min_length, args.top_n)
    
    spark.stop()


if __name__ == "__main__":
    main()

