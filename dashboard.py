import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from streamlit_mermaid import st_mermaid
import os
import subprocess
from datetime import datetime
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FraudGuard MLOps Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        border: 1px solid #ff4b4b;
        background-color: #0e1117;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def get_db_connection():
    return sqlite3.connect("mlflow.db")

def run_pipeline():
    try:
        process = subprocess.Popen(["python", "run_pipeline.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process
    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")
        return None

# --- SIDEBAR ---
st.sidebar.title("🛡️ FraudGuard MLOps")
st.sidebar.markdown("---")
st.sidebar.info("This dashboard acts as a native Windows orchestrator for your Fraud Detection pipeline.")

if st.sidebar.button("🚀 TRIGGER TRAINING PIPELINE"):
    with st.spinner("Initializing Pipeline..."):
        proc = run_pipeline()
        if proc:
            st.sidebar.success("Pipeline Started in Background!")
            st.sidebar.warning("Note: Refresh the page in 1-2 minutes to see new results.")

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.write("✅ **MLflow Backend:** SQLite")
st.sidebar.write("✅ **Tracking Folder:** `./mlruns`")
st.sidebar.write("✅ **Model Engine:** XGBoost / RF / LogReg")

# --- MAIN CONTENT ---
st.title("Fraud Detection Pipeline Orchestrator")
st.markdown("### Visualizing Experiment Lifecycle & Model Registry")

tabs = st.tabs(["📊 Performance Overview", "📐 Pipeline Architecture (DAG)", "📈 Model Comparisons"])

with tabs[0]:
    st.header("Latest Experiment Results")
    
    try:
        conn = get_db_connection()
        # Fetching PR-AUC and Recall for recently completed runs
        query = """
        SELECT 
            r.run_uuid, 
            r.start_time,
            (SELECT value FROM metrics WHERE run_uuid = r.run_uuid AND key = 'pr_auc' ORDER BY timestamp DESC LIMIT 1) as pr_auc,
            (SELECT value FROM metrics WHERE run_uuid = r.run_uuid AND key = 'recall' ORDER BY timestamp DESC LIMIT 1) as recall,
            (SELECT value FROM params WHERE run_uuid = r.run_uuid AND key = 'best_model_name' LIMIT 1) as model_name
        FROM runs r
        WHERE r.status = 'FINISHED'
        ORDER BY r.start_time DESC
        LIMIT 6
        """
        df_results = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df_results.empty:
            # Clean up times
            df_results['start_time'] = pd.to_datetime(df_results['start_time'], unit='ms')
            
            col1, col2, col3 = st.columns(3)
            
            # Metrics for the very last run
            latest = df_results.iloc[0]
            col1.metric("Latest PR-AUC", f"{latest['pr_auc']:.4f}", delta="Champion")
            col2.metric("Latest Recall", f"{latest['recall']:.4f}")
            col3.metric("Last Run Time", latest['start_time'].strftime('%H:%M:%S'))
            
            st.markdown("---")
            st.subheader("Recent Runs History")
            st.dataframe(df_results[['start_time', 'pr_auc', 'recall']].style.format({"pr_auc": "{:.4f}", "recall": "{:.4f}"}), use_container_width=True)
        else:
            st.warning("No finished runs found in mlflow.db. Trigger a pipeline to see data!")
            
    except Exception as e:
        st.error(f"Error connecting to MLflow Database: {e}")
        st.info("Ensure you have run the training pipeline at least once to create 'mlflow.db'.")

with tabs[1]:
    st.header("Pipeline DAG (Directed Acyclic Graph)")
    st.markdown("The orchestration flow defined in `pipelines/training_pipeline.py` and simulated by `run_pipeline.py`.")
    
    mermaid_code = """
    graph TD
        DATA[creditcard.csv] --> LOAD(Data Ingestion)
        LOAD --> SPLIT(Stratified Split)
        SPLIT --> PREP(ColumnTransformer / RobustScaler)
        
        subgraph Multi_Model_Training
            PREP --> LR[Logistic Regression]
            PREP --> RF[Random Forest]
            PREP --> XGB[XGBoost]
        end
        
        LR --> EVAL(Evaluate Metrics)
        RF --> EVAL
        XGB --> EVAL
        
        EVAL --> LOG(Log to MLflow Registry)
        LOG --> SELECT{Best Model Selector}
        
        SELECT -->|Highest PR-AUC + Recall Constraints| CHAMP[Register Champion Model]
    """
    st_mermaid(mermaid_code, height="500px")
    
    st.markdown("---")
    st.subheader("Model Artifacts")
    st.code("mlruns/ # Raw artifacts folder\nmlflow.db # Metadata and Registry (SQLite)")

with tabs[2]:
    st.header("Model Performance Comparison")
    
    if not df_results.empty:
        # Fetching all metrics for the latest 3 model types
        # Note: In a real dashboard, we'd join on more specific run data
        fig = px.bar(df_results, x='start_time', y=['pr_auc', 'recall'], 
                     title="PR-AUC vs Recall Trends",
                     barmode='group',
                     color_discrete_sequence=['#ff4b4b', '#00d4ff'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Confusion Matrix Gallery")
        
        # Displaying the confusion matrices saved in logs/
        log_dir = Path("logs")
        if log_dir.exists():
            cm_files = list(log_dir.glob("*_cm_standalone.png"))
            if cm_files:
                cols = st.columns(len(cm_files))
                for idx, cm_file in enumerate(cm_files):
                    cols[idx].image(str(cm_file), caption=cm_file.name.replace("_cm_standalone.png", ""))
            else:
                st.info("Run the pipeline to generate confusion matrix plots in 'logs/'.")
        else:
            st.info("'logs/' folder not found.")
    else:
        st.info("Run the training pipeline to populate comparison data.")

st.markdown("---")
st.caption("Credit Card Fraud Detection MLOps | Built for Evaluation & Portfolio")
