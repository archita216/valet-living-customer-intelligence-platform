import os
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root and agents folder to path
PROJECT_ROOT = Path(__file__).resolve().parent
AGENTS_PATH = PROJECT_ROOT / "03_agents"
if str(AGENTS_PATH) not in sys.path:
    sys.path.append(str(AGENTS_PATH))

try:
    from sql_agent import run_query
    HAS_SQL = True
except ImportError:
    HAS_SQL = False

def check_db_connection():
    """Verify if database connection is available and responsive."""
    if not HAS_SQL:
        return False, "SQL agent modules not found in path."
    
    try:
        # Run a simple query to verify connection
        run_query("SELECT 1 AS test")
        return True, "Connected successfully to Azure SQL Database."
    except Exception as e:
        return False, f"Database offline: {e}"

def get_churn_risk_data():
    """
    Load data for churn risk.
    Tries Azure SQL first, then falls back to CSV aggregation + simulation.
    """
    db_connected, _ = check_db_connection()
    
    if db_connected:
        try:
            query = """
                SELECT 
                    property_name,
                    churn_risk_label,
                    complaint_count,
                    avg_csat,
                    missed_pickups_30d,
                    contract_value,
                    renewal_date
                FROM gold_churn_risk
            """
            rows = run_query(query)
            if rows:
                df = pd.DataFrame(rows)
                # Ensure date format
                if "renewal_date" in df.columns:
                    df["renewal_date"] = pd.to_datetime(df["renewal_date"]).dt.strftime("%Y-%m-%d")
                return df, False
        except Exception:
            pass # Fallback to sandbox

    # Fallback / Sandbox mode: Load from local CSV and simulate metrics
    csv_path = PROJECT_ROOT / "01_data" / "silver_rag_source.csv"
    if not csv_path.exists():
        # Create a tiny mock dataframe if CSV is missing
        return pd.DataFrame({
            "property_name": ["Silk Road Residences", "Cedar Springs Housing", "Pine Creek Complex"],
            "churn_risk_label": ["high", "medium", "low"],
            "complaint_count": [23, 12, 3],
            "avg_csat": [1.74, 3.10, 4.50],
            "missed_pickups_30d": [10, 4, 1],
            "contract_value": [49985, 38200, 47055],
            "renewal_date": ["2026-07-15", "2026-10-12", "2027-02-28"]
        }), True

    # Aggregate complaints per property from silver CSV
    df_silver = pd.read_csv(csv_path, usecols=["property_id", "service_category", "churn_risk_label"])
    
    # Clean churn_risk_label (map NaN to 'medium')
    df_silver["churn_risk_label"] = df_silver["churn_risk_label"].fillna("medium").str.lower()
    
    # Group by property_id
    df_prop = df_silver.groupby("property_id").agg(
        complaint_count=("property_id", "count"),
        churn_risk_label=("churn_risk_label", "first")
    ).reset_index()

    # Define property name mappings
    prop_names = {
        "CUST_00861": "Silk Road Residences",
        "CUST_00770": "Cedar Springs Housing",
        "CUST_02559": "Pine Creek Complex",
        "CUST_03557": "Riverside Court",
        "CUST_09556": "MetroLoft Residences",
        "CUST_03844": "Green Valley Apartments",
        "CUST_01017": "Oakridge Estates",
        "CUST_00065": "Summit Ridge Condos",
        "CUST_05893": "Bella Vista Luxury",
    }

    def get_name(pid):
        if pid in prop_names:
            return prop_names[pid]
        prefixes = ["Willow", "Oak", "Maple", "Cedar", "Pine", "Sunset", "Highland", "River", "Summit", "Crestview"]
        suffixes = ["Residences", "Apartments", "Suites", "Estates", "Manor", "Heights", "Commons", "Court", "Crossing"]
        try:
            digits = int(''.join(filter(str.isdigit, pid)))
        except ValueError:
            digits = sum(ord(c) for c in pid)
        pref = prefixes[digits % len(prefixes)]
        suff = suffixes[(digits // 10) % len(suffixes)]
        return f"{pref} {suff}"

    df_prop["property_name"] = df_prop["property_id"].apply(get_name)

    # Generate deterministic metrics based on property ID
    def generate_metrics(row):
        pid = row["property_id"]
        risk = row["churn_risk_label"]
        seed = sum(ord(c) for c in pid)
        rng = np.random.default_rng(seed)

        if risk == "high":
            csat = round(rng.uniform(1.2, 2.5), 2)
            pickups = int(rng.integers(8, 15))
            val = int(rng.uniform(40000, 120000) // 500 * 500)
            days_to_renewal = int(rng.integers(10, 90))
        elif risk == "medium":
            csat = round(rng.uniform(2.6, 3.5), 2)
            pickups = int(rng.integers(4, 8))
            val = int(rng.uniform(30000, 90000) // 500 * 500)
            days_to_renewal = int(rng.integers(91, 240))
        else:
            csat = round(rng.uniform(3.6, 4.9), 2)
            pickups = int(rng.integers(0, 4))
            val = int(rng.uniform(25000, 80000) // 500 * 500)
            days_to_renewal = int(rng.integers(241, 365))

        renewal = (datetime.date.today() + datetime.timedelta(days=days_to_renewal)).strftime("%Y-%m-%d")
        return pd.Series([csat, pickups, val, renewal])

    df_prop[["avg_csat", "missed_pickups_30d", "contract_value", "renewal_date"]] = df_prop.apply(generate_metrics, axis=1)
    
    # Drop property_id and return
    return df_prop.drop(columns=["property_id"]), True

def get_service_performance_data():
    """
    Load service performance.
    Tries Azure SQL first, then falls back to CSV aggregation + simulation.
    """
    db_connected, _ = check_db_connection()
    
    if db_connected:
        try:
            query = """
                SELECT 
                    service_category,
                    SUM(total_complaints) AS total_complaints,
                    AVG(avg_csat) AS avg_csat,
                    AVG(avg_resolution_time) AS avg_resolution_time,
                    SUM(total_missed_pickups) AS total_missed_pickups
                FROM gold_service_performance
                GROUP BY service_category
            """
            rows = run_query(query)
            if rows:
                return pd.DataFrame(rows), False
        except Exception:
            pass

    # Fallback / Sandbox mode
    df_churn, _ = get_churn_risk_data()
    csv_path = PROJECT_ROOT / "01_data" / "silver_rag_source.csv"
    if not csv_path.exists():
        return pd.DataFrame({
            "service_category": ["Doorstep Collection", "Resident Amenities", "Maintenance Support"],
            "total_complaints": [124, 45, 12],
            "avg_csat": [3.40, 2.80, 3.10],
            "avg_resolution_time": [8.5, 24.2, 48.0],
            "total_missed_pickups": [120, 45, 10]
        }), True

    df_silver = pd.read_csv(csv_path, usecols=["service_category"])
    df_serv = df_silver.groupby("service_category").size().reset_index(name="total_complaints")
    
    # Clean and rename categories if empty
    df_serv["service_category"] = df_serv["service_category"].fillna("General support")
    df_serv = df_serv.groupby("service_category").sum().reset_index()

    # Generate deterministic metrics based on service category name
    def gen_service_metrics(row):
        cat = row["service_category"]
        seed = sum(ord(c) for c in cat)
        rng = np.random.default_rng(seed)

        if "doorstep" in cat.lower() or "collection" in cat.lower() or "pickups" in cat.lower() or "valet" in cat.lower():
            csat = round(rng.uniform(3.2, 4.2), 2)
            res_time = round(rng.uniform(4.0, 12.0), 1)
            pickups = int(df_churn["missed_pickups_30d"].sum() * 0.7)
        elif "amenities" in cat.lower() or "resident" in cat.lower():
            csat = round(rng.uniform(2.5, 3.5), 2)
            res_time = round(rng.uniform(12.0, 36.0), 1)
            pickups = int(df_churn["missed_pickups_30d"].sum() * 0.2)
        else:
            csat = round(rng.uniform(2.8, 3.8), 2)
            res_time = round(rng.uniform(24.0, 72.0), 1)
            pickups = int(df_churn["missed_pickups_30d"].sum() * 0.1)

        return pd.Series([csat, res_time, pickups])

    df_serv[["avg_csat", "avg_resolution_time", "total_missed_pickups"]] = df_serv.apply(gen_service_metrics, axis=1)
    return df_serv, True

def get_property_health_data():
    """
    Load property health.
    Tries Azure SQL first, then falls back to CSV aggregation.
    """
    db_connected, _ = check_db_connection()
    
    if db_connected:
        try:
            query = """
                SELECT 
                    property_name,
                    property_type,
                    SUM(total_complaints) AS total_complaints,
                    AVG(avg_csat) AS avg_csat,
                    SUM(total_missed_pickups) AS total_missed_pickups
                FROM gold_property_health
                GROUP BY property_name, property_type
            """
            rows = run_query(query)
            if rows:
                return pd.DataFrame(rows), False
        except Exception:
            pass

    # Fallback / Sandbox mode
    df_churn, _ = get_churn_risk_data()
    
    # Add deterministic property type
    def get_type(name):
        seed = sum(ord(c) for c in name)
        types = ["Garden Style", "Mid-Rise", "High-Rise", "Townhome Community"]
        return types[seed % len(types)]

    df_churn["property_type"] = df_churn["property_name"].apply(get_type)
    
    df_health = df_churn.rename(columns={
        "complaint_count": "total_complaints",
        "missed_pickups_30d": "total_missed_pickups"
    })[["property_name", "property_type", "total_complaints", "avg_csat", "total_missed_pickups"]]
    
    return df_health, True
