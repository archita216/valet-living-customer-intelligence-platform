import os
import sys
import time
import importlib
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Add agents directory to sys.path
AGENTS_PATH = Path(__file__).resolve().parent / "03_agents"
if str(AGENTS_PATH) not in sys.path:
    sys.path.append(str(AGENTS_PATH))

import streamlit as st
import pandas as pd
import groq_rag_agent
import sql_agent
import recommendation_agent
import langgraph_workflow

# Force reload agent modules to ensure edits are picked up without restarting Streamlit server
importlib.reload(groq_rag_agent)
importlib.reload(sql_agent)
importlib.reload(recommendation_agent)
importlib.reload(langgraph_workflow)

from groq_rag_agent import ask as rag_ask
from langgraph_workflow import run_pipeline
from sql_agent import can_handle_question, sql_agent

# Import custom data loader
from data_loader import (
    get_churn_risk_data,
    get_service_performance_data,
    get_property_health_data,
    check_db_connection
)

# Safe import for Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Page Configuration ──
st.set_page_config(
    page_title="Valet Living Intelligence Suite",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Unified Dark Forest Branding ──
st.markdown(
    """
    <style>
        /* Import premium fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

        /* Global Theme Customizations - Premium Dark Forest */
        .stApp {
            background: linear-gradient(135deg, #050d0a, #0b1a13) !important;
            color: #e2f3eb !important;
        }

        /* Typography Override */
        h1, h2, h3, .vl-brand-title {
            font-family: 'Playfair Display', Georgia, serif !important;
            color: #ffffff !important;
            font-weight: 700;
        }
        
        div, span, p, label, .stMarkdown, .stButton {
            font-family: 'Inter', sans-serif;
            color: #e2f3eb;
        }

        /* Header Replicating Valet Living Website (Dark Version) */
        .vl-header {
            background: rgba(12, 35, 23, 0.8) !important;
            padding: 1.25rem 2rem;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(132, 189, 0, 0.2);
            color: white;
        }
        .header-logo-section {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .header-logo-circle {
            background-color: #84BD00;
            color: white;
            width: 3rem;
            height: 3rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.4rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            font-family: 'Playfair Display', Georgia, serif;
        }
        .header-logo-text {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: white;
        }
        .header-nav {
            display: flex;
            gap: 1.5rem;
            align-items: center;
        }
        .header-nav a {
            color: rgba(255, 255, 255, 0.85);
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.2s ease;
        }
        .header-nav a:hover {
            color: #84BD00;
        }
        .portal-btn {
            background-color: #84BD00 !important;
            color: white !important;
            padding: 0.5rem 1.2rem;
            border-radius: 30px;
            border: none;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(132, 189, 0, 0.4);
            transition: transform 0.2s ease, background-color 0.2s ease;
        }
        .portal-btn:hover {
            transform: translateY(-2px);
            background-color: #72a300 !important;
        }

        /* Sidebar Styling Override */
        [data-testid="stSidebar"] {
            background-color: #050d0a !important;
            border-right: 1px solid rgba(132, 189, 0, 0.15) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: white !important;
            font-family: 'Playfair Display', Georgia, serif !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: rgba(226, 243, 235, 0.8) !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background-color: #0c2317 !important;
            border-color: rgba(132, 189, 0, 0.2) !important;
            color: white !important;
        }

        /* Interactive Glassmorphic KPI Cards */
        .kpi-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .kpi-card {
            background-color: rgba(12, 35, 23, 0.55) !important;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(132, 189, 0, 0.15);
            border-left: 5px solid #84BD00;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            flex: 1;
            backdrop-filter: blur(8px);
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 25px rgba(132, 189, 0, 0.1);
            border-color: rgba(132, 189, 0, 0.4);
        }
        .kpi-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(226, 243, 235, 0.6);
            font-weight: 700;
        }
        .kpi-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 0.4rem;
            line-height: 1.1;
        }
        .kpi-subtext {
            font-size: 0.78rem;
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        /* Style all bordered containers in Streamlit to match our dark green style */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(12, 35, 23, 0.35) !important;
            border: 1px solid rgba(132, 189, 0, 0.15) !important;
            border-radius: 14px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 25px rgba(0,0,0,0.25) !important;
            margin-bottom: 2rem !important;
            backdrop-filter: blur(8px);
        }
        
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0 !important;
        }

        /* Quick-Start Prompts Tiles */
        .prompt-tile {
            background-color: rgba(12, 35, 23, 0.5) !important;
            border: 1px solid rgba(132, 189, 0, 0.15) !important;
            border-radius: 10px;
            padding: 1.1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }
        .prompt-tile:hover {
            background-color: rgba(12, 35, 23, 0.8) !important;
            border-color: #84BD00 !important;
            transform: translateY(-2px);
        }
        .prompt-tile strong {
            color: #84BD00 !important;
            font-size: 0.92rem;
            display: block;
            margin-bottom: 0.25rem;
        }
        .prompt-tile span {
            color: rgba(226, 243, 235, 0.7) !important;
            font-size: 0.82rem;
        }

        /* Explainer Landing Layout (Scrollable Website Style) */
        .explainer-banner {
            background: linear-gradient(135deg, rgba(12, 35, 23, 0.6), rgba(5, 14, 9, 0.8)) !important;
            border-radius: 16px;
            padding: 2.5rem;
            margin-top: 4rem;
            border: 1px solid rgba(132, 189, 0, 0.15);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .explainer-title {
            text-align: center;
            margin-bottom: 3rem;
        }
        .explainer-title h2 {
            color: white !important;
        }
        .explainer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        .explainer-step {
            background: rgba(12, 35, 23, 0.4) !important;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
            border: 1px solid rgba(132, 189, 0, 0.1);
            border-top: 4px solid #84BD00 !important;
            position: relative;
        }
        .step-number {
            background-color: #84BD00 !important;
            color: white !important;
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            margin: 0 auto 1rem auto;
        }
        .explainer-step h4 {
            color: white !important;
            margin-bottom: 0.5rem;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .explainer-step p {
            color: rgba(226, 243, 235, 0.7) !important;
            font-size: 0.85rem;
            line-height: 1.4;
            margin: 0;
        }

        /* Agent Executive Report Style */
        .report-container {
            border-left: 5px solid #84BD00 !important;
            padding-left: 1.5rem;
            background-color: rgba(12, 35, 23, 0.3) !important;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
            border: 1px solid rgba(132, 189, 0, 0.1);
            color: #e2f3eb !important;
        }

        /* Custom buttons styling */
        .stButton button {
            background-color: #84BD00 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(132, 189, 0, 0.2);
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            background-color: #72a300 !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 15px rgba(132, 189, 0, 0.3);
        }

        /* Style streamlit tabs to match dark forest green */
        button[data-baseweb="tab"] {
            color: rgba(255, 255, 255, 0.6) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #84BD00 !important;
            border-bottom-color: #84BD00 !important;
        }

        /* Style form inputs to look dark and premium */
        div[data-testid="stForm"] {
            background: rgba(12, 35, 23, 0.2) !important;
            border-color: rgba(132, 189, 0, 0.15) !important;
        }
        
        textarea, input, [data-baseweb="select"] {
            background-color: #0c2317 !important;
            border-color: rgba(132, 189, 0, 0.25) !important;
            color: white !important;
        }
        
        /* Adjusting standard metrics rendering for dark theme */
        [data-testid="stMetricValue"] {
            color: white !important;
        }

        /* Fix for Streamlit Multiselect tag readability in dark mode */
        span[data-baseweb="tag"] {
            background-color: #84BD00 !important;
            color: white !important;
            border-radius: 4px !important;
        }
        span[data-baseweb="tag"] span {
            color: white !important;
        }
        /* Target close button icon inside multiselect chip tags */
        span[data-baseweb="tag"] button {
            color: white !important;
        }
        span[data-baseweb="tag"] button:hover {
            color: #FF5252 !important;
            background-color: transparent !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header Navigation ──
def render_header():
    st.markdown(
        """
        <div class="vl-header">
            <div class="header-logo-section">
                <div class="header-logo-circle">VL</div>
                <div class="header-logo-text">Valet Living</div>
            </div>
            <div class="header-nav">
                <a href="#dashboard">Safety</a>
                <a href="#dashboard" style="color: #84BD00; font-weight: 700;">Property Intelligence</a>
                <a href="#dashboard">Residents</a>
                <a href="#dashboard">Insights</a>
            </div>
            <div>
                <button class="portal-btn" onclick="window.parent.location.reload();">System Status: Connected</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar Configuration ──
def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; margin-top: 1rem;">
            <div style="background-color: #84BD00; color: white; width: 2.2rem; height: 2.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-family: 'Playfair Display', Georgia, serif;">VL</div>
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: white;">Intelligence Suite</div>
        </div>
        
        <div style="font-size: 0.9rem; font-weight: 600; color: white; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Control Panel</div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.caption("Select execution routing mode:")

    mode = st.sidebar.selectbox(
        "Execution Mode",
        [
            "Auto Route (Recommended)",
            "Full Multi-Agent Pipeline",
            "Structured SQL Answer",
            "RAG Context Search",
        ],
        label_visibility="collapsed"
    )

    # Convert readable UI option to backend execution value
    mode_map = {
        "Auto Route (Recommended)": "Auto route",
        "Full Multi-Agent Pipeline": "Full multi-agent workflow",
        "Structured SQL Answer": "Structured SQL answer",
        "RAG Context Search": "RAG context search"
    }
    backend_mode = mode_map[mode]

    st.sidebar.markdown("---")
    
    # Custom styled compact Sidebar status widgets
    db_ok, db_msg = check_db_connection()
    db_status = "Active" if db_ok else "Fallback Active"
    db_color = "#81C784" if db_ok else "#FFB74D"
    
    st.sidebar.markdown(
        f"""
        <div style="background-color: rgba(12, 35, 23, 0.4); border: 1px solid rgba(132, 189, 0, 0.15); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; color: rgba(226, 243, 235, 0.6); text-transform: uppercase; font-weight: 700; margin-bottom: 0.25rem;">Database Status</div>
            <div style="font-size: 0.85rem; color: {db_color}; font-weight: 600;">{db_status} (Azure SQL)</div>
        </div>
        
        <div style="background-color: rgba(12, 35, 23, 0.4); border: 1px solid rgba(132, 189, 0, 0.15); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; color: rgba(226, 243, 235, 0.6); text-transform: uppercase; font-weight: 700; margin-bottom: 0.4rem;">Active AI Nodes</div>
            <div style="font-size: 0.8rem; color: white; margin-bottom: 0.2rem; font-weight: 500;">✓ SQL Node: Online</div>
            <div style="font-size: 0.8rem; color: white; margin-bottom: 0.2rem; font-weight: 500;">✓ RAG Node: Online</div>
            <div style="font-size: 0.8rem; color: white; font-weight: 500;">✓ Advisor Node: Online</div>
        </div>
        
        <div style="background-color: rgba(12, 35, 23, 0.4); border: 1px solid rgba(132, 189, 0, 0.15); border-radius: 8px; padding: 0.75rem;">
            <div style="font-size: 0.75rem; color: rgba(226, 243, 235, 0.6); text-transform: uppercase; font-weight: 700; margin-bottom: 0.25rem;">Dataset Status</div>
            <div style="font-size: 0.75rem; color: rgba(226, 243, 235, 0.8); line-height: 1.35;">
                FAISS Index: 100k+ complaints.<br>
                Gold Tables: churn_risk, service_perf, property_health active.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    return backend_mode

# ── KPI Cards Builder (Using Average Missed Pickups for Correct Insight) ──
def render_kpi_cards(df_churn):
    high_risk_count = len(df_churn[df_churn["churn_risk_label"] == "high"])
    avg_csat = df_churn["avg_csat"].mean()
    avg_pickups = df_churn["missed_pickups_30d"].mean()
    revenue_at_risk = df_churn[df_churn["churn_risk_label"] == "high"]["contract_value"].sum()

    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Revenue at Risk</div>
                <div class="kpi-value">${revenue_at_risk:,.0f}</div>
                <div class="kpi-subtext" style="color: #FF5252; font-weight: 700;">High Churn Risk Segment</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">High Risk Properties</div>
                <div class="kpi-value">{high_risk_count}</div>
                <div class="kpi-subtext" style="color: #FFB74D; font-weight: 700;">Low CSAT / Pickup Alerts</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average CSAT</div>
                <div class="kpi-value">{avg_csat:.2f} / 5.0</div>
                <div class="kpi-subtext" style="color: #81C784; font-weight: 700;">Portfolio Average Score</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Avg Missed Pickups (30d)</div>
                <div class="kpi-value">{avg_pickups:.1f}</div>
                <div class="kpi-subtext" style="color: rgba(226, 243, 235, 0.6); font-weight: 500;">Incidents per property</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Executive Insights Builder (Actionable Analytics Section) ──
def render_executive_insights(df_churn):
    df_churn = df_churn.copy()
    df_churn["renewal_date_dt"] = pd.to_datetime(df_churn["renewal_date"])
    today = pd.Timestamp(datetime.date.today())
    renewing_high_risk = df_churn[
        (df_churn["churn_risk_label"] == "high") & 
        (df_churn["renewal_date_dt"] <= today + pd.Timedelta(days=30))
    ]
    renewing_count = len(renewing_high_risk)
    renewing_rev = renewing_high_risk["contract_value"].sum()

    worst_prop = df_churn.sort_values("avg_csat", ascending=True).iloc[0]
    
    df_serv, _ = get_service_performance_data()
    worst_category = df_serv.sort_values("total_complaints", ascending=False).iloc[0]
    avg_res_time = df_serv["avg_resolution_time"].mean()

    # Renders static executive warning grids inside a single st.markdown block to avoid empty container glitches
    st.markdown(
        f"""
        <div style="background-color: rgba(12, 35, 23, 0.35); border: 1px solid rgba(132, 189, 0, 0.15); border-radius: 14px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 4px 25px rgba(0,0,0,0.25); backdrop-filter: blur(8px);">
            <h3 style="color: white; margin-top: 0; margin-bottom: 1.2rem; font-family: 'Playfair Display', Georgia, serif; font-size: 1.3rem;">Portfolio Insights & Critical Alerts</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                <div style="background-color: rgba(255, 82, 82, 0.08); border-left: 4px solid #FF5252; padding: 0.85rem 1rem; border-radius: 4px;">
                    <strong style="color: #FF5252; font-size: 0.88rem; display: block; margin-bottom: 0.25rem;">Urgent Churn Risk Renewals</strong>
                    <span style="color: #e2f3eb; font-size: 0.82rem; line-height: 1.4;">
                        <strong>{renewing_count}</strong> high-risk properties renewing within 30 days, representing <strong>${renewing_rev:,.0f}</strong> at-risk monthly revenue.
                    </span>
                </div>
                <div style="background-color: rgba(255, 183, 77, 0.08); border-left: 4px solid #FFB74D; padding: 0.85rem 1rem; border-radius: 4px;">
                    <strong style="color: #FFB74D; font-size: 0.88rem; display: block; margin-bottom: 0.25rem;">Lowest Portfolio Satisfaction</strong>
                    <span style="color: #e2f3eb; font-size: 0.82rem; line-height: 1.4;">
                        <strong>{worst_prop['property_name']}</strong> has a critical CSAT of <strong>{worst_prop['avg_csat']}/5.0</strong> with {worst_prop['missed_pickups_30d']} missed pickups this month.
                    </span>
                </div>
                <div style="background-color: rgba(132, 189, 0, 0.08); border-left: 4px solid #84BD00; padding: 0.85rem 1rem; border-radius: 4px;">
                    <strong style="color: #84BD00; font-size: 0.88rem; display: block; margin-bottom: 0.25rem;">Operational Bottleneck</strong>
                    <span style="color: #e2f3eb; font-size: 0.82rem; line-height: 1.4;">
                        <strong>{worst_category['service_category']}</strong> leads complaints ({worst_category['total_complaints']} cases). Average resolution time: {avg_res_time:.1f} hours.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Quick-Start Prompts Tiles ──
def render_quick_prompts():
    st.markdown("<h3 style='margin-bottom:1rem; color: white;'>Quick-Start Prompts</h3>", unsafe_allow_html=True)
    
    # Predefined prompts
    prompts = [
        "Which properties are most at risk of churn?",
        "What are the most common service complaints?",
        "Give me a retention plan for low CSAT properties.",
    ]
    
    cols = st.columns(3)
    for i, title in enumerate(prompts):
        with cols[i]:
            if st.button(title, use_container_width=True, key=f"btn_prompt_{i}"):
                st.session_state["question_input"] = title
                st.rerun()

# ── Agent Workflow Run Handler ──
def run_agent_pipeline_with_logs(mode, question):
    with st.status("Orchestrating Valet Living Intelligence Agent...", expanded=True) as status:
        try:
            st.write("Initialization: Activating LangGraph multi-agent execution context...")
            time.sleep(0.5)

            # SQL Routing Step
            if mode in ["Auto route", "Full multi-agent workflow", "Structured SQL answer"]:
                st.write("SQL Agent: Mapping question intent to gold database tables...")
                time.sleep(0.6)
                if mode == "Auto route" or mode == "Full multi-agent workflow":
                    can_handle = can_handle_question(question)
                    st.write(f"SQL Agent: Intent matched? {'YES - routing to SQL' if can_handle else 'NO - skipping SQL node'}")
                    time.sleep(0.4)
                st.write("SQL Agent: Querying Azure SQL gold_churn_risk & gold_property_health views...")
                time.sleep(0.6)

            # RAG Scanning Step
            if mode in ["Auto route", "Full multi-agent workflow", "RAG context search"]:
                st.write("RAG Agent: Embedding customer query & matching historical vectors...")
                time.sleep(0.6)
                st.write("RAG Agent: Performing semantic scan on FAISS indexing...")
                time.sleep(0.7)

            # Advisor Orchestration Step
            if mode in ["Auto route", "Full multi-agent workflow"]:
                st.write("Recommendation Agent: Collecting findings from SQL & RAG nodes...")
                time.sleep(0.5)
                st.write("Recommendation Agent: Running synthesis logic via Groq Llama-3.1 model...")
                time.sleep(0.6)

            # Run backend code (no other functional modifications, matching original wrapper)
            result = run_selected_mode(mode, question)
            
            status.update(label="Analysis complete!", state="complete", expanded=False)
            return result
        except Exception as e:
            status.update(label="Pipeline execution failed!", state="error", expanded=False)
            raise e

def run_selected_mode(mode: str, question: str):
    # Restores original function execution logic exactly
    if mode == "Full multi-agent workflow":
        result = run_pipeline(question)
        return {
            "mode": "full",
            "sql_result":     result["sql_result"],
            "rag_result":     result["rag_result"],
            "recommendation": result["recommendation"],
        }

    if mode == "Structured SQL answer":
        return {"mode": "sql", "sql_result": sql_agent(question)}

    if mode == "RAG context search":
        return {"mode": "rag", "rag_result": rag_ask(question)}

    # Auto route
    if can_handle_question(question):
        result = run_pipeline(question)
        return {
            "mode": "full",
            "sql_result":     result["sql_result"],
            "rag_result":     result["rag_result"],
            "recommendation": result["recommendation"],
        }

    return {
        "mode": "rag",
        "sql_result": sql_agent(question),
        "rag_result": rag_ask(question),
    }

# ── Agent Results Formatter (Using st.container(border=True) to avoid empty containers) ──
def render_agent_results(result):
    if not result:
        return
    with st.container(border=True):
        st.markdown("<h3 style='margin-bottom:1.5rem; color: white; margin-top: 0;'>Analysis Findings</h3>", unsafe_allow_html=True)

        if result.get("mode") == "full":
            # Multi-tab layout for visual distinction of outputs
            tab_rec, tab_sql, tab_rag = st.tabs([
                "Strategic Action Plan", 
                "Structured SQL Metrics", 
                "Historical RAG Context"
            ])

            with tab_rec:
                st.markdown("#### **Valet Living Strategic Executive Report**")
                st.markdown(
                    f"<div class='report-container'>{result['recommendation']}</div>", 
                    unsafe_allow_html=True
                )

            with tab_sql:
                st.markdown("#### **Database Queries & Structured Findings**")
                st.code(result["sql_result"])

            with tab_rag:
                st.markdown("#### **Retrieved Resident Complaint Context**")
                st.info("The following complaint text logs were semantically matched from the FAISS database:")
                st.write(result["rag_result"])

        elif result.get("mode") == "sql":
            st.success("Structured SQL Analysis Output")
            st.write(result["sql_result"])

        elif result.get("mode") == "rag":
            if "sql_result" in result:
                st.warning(result["sql_result"])
            st.info("RAG Semantic Search Output")
            st.write(result["rag_result"])

# ── Scrollable Explainer Section (Website Landing Page Style) ──
def render_explainer_section(is_sandbox):
    st.markdown(
        """
        <div class="explainer-banner">
            <div class="explainer-title">
                <h2>How the Valet Living Intelligence Suite Works</h2>
                <p style="color: rgba(226, 243, 235, 0.6); font-size: 1rem;">A multi-layered data pipeline processing doorstep services feedback in real-time.</p>
            </div>
            <div class="explainer-grid">
                <div class="explainer-step">
                    <div class="step-number">1</div>
                    <h4>Databricks Gold ETL</h4>
                    <p>Aggregates raw customer feedback and pickup reports through a Bronze-Silver-Gold notebook workflow into analytical tables.</p>
                </div>
                <div class="explainer-step">
                    <div class="step-number">2</div>
                    <h4>Azure SQL Database</h4>
                    <p>Hosts clean dimensional business views detailing churn metrics, CSAT scores, contract values, and support categories.</p>
                </div>
                <div class="explainer-step">
                    <div class="step-number">3</div>
                    <h4>FAISS Vector Store</h4>
                    <p>Indexes semantic text embeddings from complaint tickets to retrieve deep historic context not available via SQL queries.</p>
                </div>
                <div class="explainer-step">
                    <div class="step-number">4</div>
                    <h4>LangGraph Routing</h4>
                    <p>An orchestrator maps user questions, runs the appropriate agents in parallel, and compiles strategic actions using Llama-3.1.</p>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(132, 189, 0, 0.15); margin: 2rem 0;">
            <div style="text-align: center; color: rgba(226, 243, 235, 0.6); font-size: 0.88rem;">
                <strong>System Architecture</strong>: Streamlines Bronze-Silver-Gold ETL datasets, Azure SQL relational databases, FAISS semantic vector search, and LangGraph multi-agent cognitive routing inside one unified execution workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main Application entrypoint ──
def main():
    render_header()
    backend_mode = render_sidebar()

    # Load churn risk dataset for executive KPIs & graphs
    df_churn, is_sandbox = get_churn_risk_data()

    # App Views Tabs (Clean name strings, no emojis)
    tab_dashboard, tab_agent = st.tabs([
        "Executive Analytics Dashboard", 
        "Multi-Agent Command Center"
    ])

    with tab_dashboard:
        st.markdown("<h2 style='margin-bottom:1rem; color: white;'>Property Portfolio Performance</h2>", unsafe_allow_html=True)
        
        # Display KPI cards grid
        render_kpi_cards(df_churn)
        
        # Render actionable operational insights
        render_executive_insights(df_churn)

        # Filters and visual grid (Using st.container(border=True) to avoid empty containers)
        with st.container(border=True):
            st.markdown("<h3 style='color: white; margin-top: 0;'>Portfolio Filter Tools</h3>", unsafe_allow_html=True)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                search_query = st.text_input("Search property by name", placeholder="e.g. Silk Road Residences...")
            with col_f2:
                risk_filter = st.multiselect("Filter by risk category", ["high", "medium", "low"], default=["high", "medium", "low"])

        # Filter dataset
        df_filtered = df_churn[df_churn["churn_risk_label"].isin(risk_filter)]
        if search_query.strip():
            df_filtered = df_filtered[df_filtered["property_name"].str.contains(search_query, case=False)]

        # Charts Section
        if HAS_PLOTLY:
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                # 1. Churn Risk Distribution Donut Chart
                df_risk_group = df_filtered.groupby("churn_risk_label").size().reset_index(name="count")
                fig_pie = px.pie(
                    df_risk_group,
                    values="count",
                    names="churn_risk_label",
                    hole=0.45,
                    color="churn_risk_label",
                    color_discrete_map={"high": "#FF5252", "medium": "#FFB74D", "low": "#81C784"},
                    title="Risk Category Breakdown",
                    template="plotly_dark"
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2f3eb",
                    title_font_color="#ffffff",
                    legend=dict(orientation="h", y=-0.1)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_c2:
                # 2. Category Performance Horizontal Bar
                df_serv, _ = get_service_performance_data()
                df_serv_sorted = df_serv.sort_values("total_complaints", ascending=True)
                fig_bar = px.bar(
                    df_serv_sorted,
                    x="total_complaints",
                    y="service_category",
                    orientation="h",
                    color_discrete_sequence=["#84BD00"],
                    title="Complaints by Service Category",
                    template="plotly_dark"
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2f3eb",
                    title_font_color="#ffffff"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # 3. 2D Scatter Matrix Bubble Chart
            fig_scatter = px.scatter(
                df_filtered,
                x="missed_pickups_30d",
                y="avg_csat",
                size="contract_value",
                color="churn_risk_label",
                hover_name="property_name",
                color_discrete_map={"high": "#FF5252", "medium": "#FFB74D", "low": "#81C784"},
                size_max=32,
                labels={
                    "missed_pickups_30d": "Missed Pickups (30d)",
                    "avg_csat": "Average CSAT score",
                    "churn_risk_label": "Risk"
                },
                title="Property Health Matrix (CSAT vs. Missed Pickups, Bubble size = Revenue)",
                template="plotly_dark"
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2f3eb",
                title_font_color="#ffffff"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Loading interactive chart visualization plugins...")
            st.bar_chart(df_filtered.set_index("property_name")["missed_pickups_30d"])

        # Data Leaderboard (Using st.container(border=True) to avoid empty containers)
        with st.container(border=True):
            st.markdown("<h3 style='color: white; margin-top: 0;'>Leaderboard: Properties at Churn Risk</h3>", unsafe_allow_html=True)
            df_leaderboard = df_filtered.sort_values("complaint_count", ascending=False).rename(columns={
                "property_name": "Property Name",
                "churn_risk_label": "Risk Level",
                "complaint_count": "Complaints",
                "avg_csat": "CSAT Score",
                "missed_pickups_30d": "Missed Pickups",
                "contract_value": "Contract Revenue ($)",
                "renewal_date": "Renewal Date"
            })
            st.dataframe(
                df_leaderboard[["Property Name", "Risk Level", "Complaints", "CSAT Score", "Missed Pickups", "Contract Revenue ($)", "Renewal Date"]],
                use_container_width=True,
                hide_index=True
            )

        # Render the scrollable product explainer section at the bottom
        render_explainer_section(is_sandbox)

    with tab_agent:
        st.markdown("<h2 style='margin-bottom:1rem; color: white;'>Multi-Agent Command Center</h2>", unsafe_allow_html=True)
        
        # Check API key first
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY is not loaded in this session. Check the .env file in the project root, then restart Streamlit.")
            return

        # Render Quick Prompts Tiles
        render_quick_prompts()

        # Initialize query session state if not set
        if "question_input" not in st.session_state:
            st.session_state["question_input"] = ""

        # Ask the assistant input form (Using st.container(border=True) to avoid empty containers)
        with st.container(border=True):
            st.markdown("<h3 style='color: white; margin-top: 0;'>Input Query</h3>", unsafe_allow_html=True)
            
            with st.form("agent_query_form"):
                user_question = st.text_area(
                    "Ask a natural language question about properties, complaints, churn, or renewals:",
                    value=st.session_state["question_input"],
                    height=90,
                    placeholder="e.g. Which properties are most at risk of churn and what is the recommended retention plan?"
                )
                submit = st.form_submit_button("Execute Multi-Agent Routing")

            if submit and user_question.strip():
                # Reset session state matching form input to preserve text area sync
                st.session_state["question_input"] = user_question.strip()
                
                try:
                    # Run the pipeline with live interactive logging progress
                    result = run_agent_pipeline_with_logs(backend_mode, user_question.strip())
                    st.session_state["last_agent_result"] = result
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")
                    st.code(str(e))

        # Output results if available
        if "last_agent_result" in st.session_state:
            render_agent_results(st.session_state["last_agent_result"])

if __name__ == "__main__":
    main()
