# Databricks notebook source
    df = spark.table("bronze_it_support_tickets")

display(df.limit(5))

# COMMAND ----------

df = (
    df.withColumnRenamed("customer_id", "property_id")
      .withColumnRenamed("customer_segment", "property_type")
      .withColumnRenamed("sla_plan", "service_tier")
      .withColumnRenamed("initial_message", "complaint_description")
      .withColumnRenamed("agent_first_reply", "service_response")
      .withColumnRenamed("resolution_summary", "resolution_notes")
      .withColumnRenamed("customer_sentiment", "complaint_sentiment")
)

# COMMAND ----------

# Added Property Names

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

property_names = [
    "Sunset Ridge Apartments", "Oak Grove Communities",
    "Maple Valley Residences", "Riverside Court",
    "Blue Harbor Flats", "Pine Creek Complex",
    "Lakewood Terrace", "Elmwood Park Apartments",
    "Cedar Springs Housing", "Horizon View Communities",
    "Greenfield Manor", "Willowbrook Estates",
    "Stonegate Residences", "Maplewood Commons",
    "Harbor Point Apartments", "The Grandview", "Regal Heights", "MetroLoft Residences", "Silk Road Residences", "Emerald Heights", "Meridian Heights", "The Belvedere", "Tiffany Towers", "The Kensington"
]

# Same property_id always gets same name (consistent across rows)
def assign_property_name(property_id):
    idx = int(property_id.replace("CUST_", "")) % len(property_names)
    return property_names[idx]

name_udf = udf(assign_property_name, StringType())
df = df.withColumn("property_name", name_udf(df.property_id))

# COMMAND ----------

# Added Contract Value

from pyspark.sql.functions import abs, hash

df = df.withColumn(
    "contract_value",
    (abs(hash(df.property_id) % 40000) + 10000).cast("int")
)

# COMMAND ----------

# Added Missed Pickups

from pyspark.sql.functions import floor, abs, hash

# hash(property_id) gives consistent number per property_id
df = df.withColumn(
    "missed_pickups_30d",
    abs(hash(df.property_id) % 11).cast("int")  # 0 to 10
)

# COMMAND ----------

# Added Renewal Dates

from pyspark.sql.functions import abs, hash, expr, date_add, current_date

df = df.withColumn(
    "renewal_date",
    date_add(current_date(), (abs(hash(df.property_id) % 365)).cast("int"))
)

# COMMAND ----------

# Added Service Category

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def assign_service_category(issue):

    mapping = {
        "billing_problem": "Resident Amenities",
        "account_access": "Resident Amenities",
        "performance": "Doorstep Collection",
        "bug": "Maintenance Support",
        "security_concern": "Maintenance Support",
        "feature_request": "Resident Amenities",
        "how_to": "Pet Services",
        "other": "Turn Services"
    }

    return mapping.get(issue, "Doorstep Collection")

service_udf = udf(assign_service_category, StringType())

df = df.withColumn(
    "service_category",
    service_udf(df.issue_type)
)

# COMMAND ----------

# Added Churn Risk Logic

from pyspark.sql.functions import when

df = df.withColumn(
    "churn_risk_label",
    when(
        (df.csat_score <= 2) | (df.missed_pickups_30d >= 8), "high"
    )
    .when(
        (df.csat_score == 3) | (df.missed_pickups_30d >= 4), "medium"
    )
    .otherwise("low")
)

# COMMAND ----------

# DBTITLE 1,Cell 9
df.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("silver_valet_living_service_data")

# COMMAND ----------

display(
    spark.sql("""
    SELECT *
    FROM silver_valet_living_service_data
    LIMIT 10
    """)
)

# COMMAND ----------

# Export the 6 RAG columns as CSV
silver_df = spark.table("silver_valet_living_service_data")

silver_rag = silver_df.select(
    "property_id",
    "complaint_description",
    "service_response",
    "resolution_notes",
    "service_category",
    "churn_risk_label"
)

# Save to Unity Catalog Volume (serverless-compatible)
silver_rag.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("/Volumes/workspace/default/my_volume/silver_rag_source")

print("Exported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary of Transformations from `bronze_it_support_tickets` to `silver_valet_living_service_data`
# MAGIC
# MAGIC - **Column Renaming:** Several columns were renamed for domain alignment with Valet Living's business context  (e.g., `customer_id` → `property_id`, `customer_segment` → `property_type`, etc.).
# MAGIC - **New Columns Added:**
# MAGIC   - `property_name`: Mapped from property_id using a deterministic UDF — each unique property ID consistently maps to a real residential property name.
# MAGIC   - `contract_value`: Derived using hash(property_id) to ensure a consistent contract value per property, ranging from $10,000 to $50,000 — same property always has the same contract value.
# MAGIC   - `missed_pickups_30d`: Derived using hash(property_id) to assign a consistent missed pickup count (0–10) per property — eliminates row-level randomness so property-level aggregations are meaningful.
# MAGIC   - `renewal_date`: Computed as a consistent date per property using hash(property_id), falling within the next 365 days from today.
# MAGIC   - `service_category`: Mapped from issue_type using a custom UDF — categorises each ticket into Valet Living service lines: Doorstep Collection, Maintenance Support, Resident Amenities, Pet Services, and Turn Services.
# MAGIC   - `churn_risk_label`: Derived from csat_score and missed_pickups_30d using business rules — labelled as high, medium, or low (lowercase) to align with DAX measure filters in Power BI.
# MAGIC - **No columns were dropped.**
# MAGIC - **All original columns from the bronze table are retained (possibly renamed), with several new features engineered for analytics.**
# MAGIC
# MAGIC - **Key design principle:** Property-level attributes (property_name, contract_value, missed_pickups_30d, renewal_date, churn_risk_label) use deterministic hashing on property_id rather than row-level randomness — ensuring consistent, aggregatable values across the 100,000 row dataset.
# MAGIC

# COMMAND ----------

silver_df = spark.table("silver_valet_living_service_data")
print(silver_df.count())

# COMMAND ----------

