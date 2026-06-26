# Databricks notebook source
# MAGIC %md
# MAGIC Objective: Move business-ready Gold layer tables from Databricks into Azure SQL Database
# MAGIC
# MAGIC These tables will serve as the structured data source for SQL Analytics, BI Dashboards and the SQL Agent too. 

# COMMAND ----------

# MAGIC %md
# MAGIC Tables being published are:
# MAGIC
# MAGIC     1. gold_churn_risk
# MAGIC     2. gold_property_health
# MAGIC     3. gold_service_performance

# COMMAND ----------

# Read the gold tables created in Databricks

gold_churn_risk = spark.table("gold_churn_risk")
gold_property_health = spark.table("gold_property_health")
gold_service_performance = spark.table("gold_service_performance")

# COMMAND ----------

#Credentials for Configuration

SQL_HOST     = "valet-sql-server.database.windows.net"
SQL_PORT     = "1433"
SQL_DB       = "valetlivingdb"
SQL_USER     = "CloudSAb803d905"
SQL_PASSWORD = "Archita06/21"

# COMMAND ----------

# Read updated gold tables from Databricks

gold_churn_risk         = spark.table("gold_churn_risk")
gold_service_performance = spark.table("gold_service_performance")
gold_property_health    = spark.table("gold_property_health")

print("Gold tables loaded from Databricks:")
print("  gold_churn_risk:          ", gold_churn_risk.count(), "rows")
print("  gold_service_performance: ", gold_service_performance.count(), "rows")
print("  gold_property_health:     ", gold_property_health.count(), "rows")

# COMMAND ----------

jdbcUrl = (
    "jdbc:sqlserver://valet-sql-server.database.windows.net:1433;"
    "database=valetlivingdb;"
    "encrypt=true;"
    "trustServerCertificate=false;"
    "hostNameInCertificate=*.database.windows.net;"
    "loginTimeout=30;"
)

try:
    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbcUrl)
        .option("dbtable", "(SELECT 1 AS test) t")
        .option("user", SQL_USER)
        .option("password", SQL_PASSWORD)
        .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")
        .load()
    )

    display(df)

except Exception as e:
    print(str(e))

# COMMAND ----------


# Helper function to write to Azure SQL

def write_to_sql(df, table_name):
    df.write \
        .format("sqlserver") \
        .mode("overwrite") \
        .option("host", SQL_HOST) \
        .option("port", SQL_PORT) \
        .option("database", SQL_DB) \
        .option("dbtable", table_name) \
        .option("user", SQL_USER) \
        .option("password", SQL_PASSWORD) \
        .option("encrypt", "true") \
        .save()
    print(f"✓ {table_name} written to Azure SQL successfully")

# COMMAND ----------

# Writing all 3 tables to Azure SQL

for df, table_name in [
    (gold_churn_risk, "gold_churn_risk"),
    (gold_service_performance, "gold_service_performance"),
    (gold_property_health, "gold_property_health")
]:
    write_to_sql(df, table_name)

# COMMAND ----------

# Verify the tables were created or not

df = (
    spark.read
    .format("sqlserver")
    .option("host", SQL_HOST)
    .option("port", "1433")
    .option("database", SQL_DB)
    .option("dbtable", "INFORMATION_SCHEMA.TABLES")
    .option("user", SQL_USER)
    .option("password", SQL_PASSWORD)
    .option("encrypt", "true")
    .load()
)

display(df)