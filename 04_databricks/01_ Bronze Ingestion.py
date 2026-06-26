# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Ingestion
# MAGIC
# MAGIC Purpose:
# MAGIC - Ingest raw support ticket dataset
# MAGIC - Preserve source data without transformations
# MAGIC - Create Bronze table for downstream processing
# MAGIC
# MAGIC Source:
# MAGIC synthetic_it_support_tickets.csv

# COMMAND ----------

bronze_df = (
    spark.read
         .format("csv")
         .option("header", "true")
         .option("inferSchema", "true")
         .load("/Volumes/workspace/default/my_volume/synthetic_it_support_tickets.csv")
)

display(bronze_df)

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

print(f"Total Records: {bronze_df.count()}")

# COMMAND ----------

from pyspark.sql.functions import col, count, when

null_counts = bronze_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in bronze_df.columns
])

display(null_counts)

# COMMAND ----------

bronze_df.write \
    .mode("overwrite") \
    .saveAsTable("bronze_it_support_tickets")

# COMMAND ----------

display(
    spark.sql("""
    SELECT *
    FROM bronze_it_support_tickets
    LIMIT 10
    """)
)

# COMMAND ----------

display(spark.sql("SHOW TABLES"))