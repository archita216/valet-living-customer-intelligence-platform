# Databricks notebook source
# MAGIC %md
# MAGIC Property Health Dashboard

# COMMAND ----------

from pyspark.sql.functions import *

silver_df = spark.table("silver_valet_living_service_data")

gold_property_health = (
    silver_df
    .groupBy(
        "property_id",
        "property_name",
        "property_type",
        "service_category"
    )
    .agg(
        count("*").alias("total_complaints"),
        round(avg("csat_score"), 2).alias("avg_csat"),
        max("missed_pickups_30d").alias("total_missed_pickups"),
        round(avg("contract_value"), 2).alias("contract_value")
    )
)

display(gold_property_health)

gold_property_health.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_property_health")

# COMMAND ----------

# MAGIC %md
# MAGIC Churn Risk Dashboard

# COMMAND ----------

from pyspark.sql.functions import *

gold_churn_risk = (
    silver_df
    .groupBy(
        "property_id",
        "property_name"
    )
    .agg(
        count("*").alias("complaint_count"),
        round(avg("csat_score"), 2).alias("avg_csat"),
        max("missed_pickups_30d").alias("missed_pickups_30d"),
        round(avg("contract_value"), 2).alias("contract_value"),
        max("renewal_date").alias("renewal_date")
    )
)

# COMMAND ----------

from pyspark.sql.functions import col, when

gold_churn_risk = gold_churn_risk.withColumn(
    "churn_risk_label",
    when(
        (col("avg_csat") <= 2.5) |
        (col("missed_pickups_30d") >= 8),
        "high"
    )
    .when(
        (col("avg_csat") <= 3.5) |
        (col("missed_pickups_30d") >= 4),
        "medium"
    )
    .otherwise("low")
)

# COMMAND ----------

gold_churn_risk.printSchema()

# COMMAND ----------

gold_churn_risk.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_churn_risk")

# COMMAND ----------

display(
    spark.sql("""
    DESCRIBE gold_churn_risk
    """)
)

# COMMAND ----------

gold_churn_risk.count()

# COMMAND ----------

display(
    spark.sql("""
    SELECT property_id,
           COUNT(*) as records
    FROM gold_churn_risk
    GROUP BY property_id
    HAVING COUNT(*) > 1
    """)
)

# COMMAND ----------

display(
    spark.sql("""
    SELECT *
    FROM gold_churn_risk
    LIMIT 5
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Service Performance Dashboard

# COMMAND ----------

from pyspark.sql.functions import *

gold_service_performance = (
    silver_df
    .groupBy(
        "service_category",
        "property_id",
        "property_name",
        "property_type"
    )
    .agg(
        count("*").alias("total_complaints"),
        round(avg("csat_score"), 2).alias("avg_csat"),
        round(avg("resolution_time_hours"), 2).alias("avg_resolution_time"),
        max("missed_pickups_30d").alias("total_missed_pickups"),
        round(avg("contract_value"), 2).alias("contract_value")
    )
)

display(gold_service_performance)

gold_service_performance.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("gold_service_performance")