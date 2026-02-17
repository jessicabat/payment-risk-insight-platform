"""
Phase 4: Analyst Experience & Decision Support Interface
--------------------------------------------------------
An interactive Streamlit dashboard for reviewing high-risk transactions,
analyzing SHAP drivers, and generating LLM-powered audit narratives.
"""

import streamlit as st
import json
import pandas as pd
import altair as alt
import sys
from pathlib import Path

# Ensure Python can find our custom modules
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.genai.narrative_generator import generate_insight
from src.audit.logger import log_generation

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Payment Risk Insight Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

POLICY_FILE = Path("artifacts/policies/decision_policy_v1.json")
SHAP_FILE = Path("artifacts/explainability/xgb_v1_sample_explanations.json")

# --- DATA LOADING (Cached for performance) ---
@st.cache_data
def load_data():
    with open(POLICY_FILE) as f:
        policy = json.load(f)
    with open(SHAP_FILE) as f:
        transactions = json.load(f)
    return policy, transactions

policy, transactions = load_data()
threshold = policy["threshold"]

# --- SIDEBAR: Transaction Selector ---
st.sidebar.title("🛡️ Risk Operations")

# NEW: Queue stats to make it feel like a live ops tool
st.sidebar.metric("High-Risk Queue Size", len(transactions))
avg_risk = sum(t['predicted_risk_score'] for t in transactions) / len(transactions) if transactions else 0
st.sidebar.metric("Avg Risk Score", f"{avg_risk:.2f}")

st.sidebar.divider()
st.sidebar.markdown("### Queue: High-Risk Alerts")

# Create a clean list of transaction IDs for the dropdown
txn_options = {f"TXN-{t['transaction_index']} (Score: {t['predicted_risk_score']:.2f})": t for t in transactions}
selected_txn_label = st.sidebar.selectbox("Select Transaction to Review:", list(txn_options.keys()))
selected_txn = txn_options[selected_txn_label]

st.sidebar.divider()
st.sidebar.markdown("**System Telemetry**")
st.sidebar.caption(f"Active Policy: `{policy['policy_name']}`")
st.sidebar.caption(f"Model: `{policy['model_artifact'].split('/')[-1]}`")
st.sidebar.caption(f"Economic Threshold: `{threshold}`")


# --- MAIN CONTENT AREA ---
score = selected_txn["predicted_risk_score"]
decision = policy["decision_labels"]["block"] if score >= threshold else policy["decision_labels"]["allow"]

st.title(f"Transaction Review: {selected_txn['transaction_index']}")

# NEW: Context banner for instant narrative clarity
st.caption(
    "This dashboard simulates a fraud analyst reviewing high-risk transactions "
    "flagged by an economically optimized risk policy."
)

# Top Row: Key Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="AI Risk Score", 
        value=f"{score:.3f}", 
        delta=f"{(score - threshold):.3f} over threshold" if score >= threshold else f"{(threshold - score):.3f} under threshold",
        delta_color="inverse"
    )
with col2:
    # NEW: Decision badge with color semantics for a governed feel
    decision_color = "red" if decision == "DECLINE" else "green"
    st.markdown(f"### Policy Decision: :{decision_color}[{decision}]")
with col3:
    st.metric(label="Actual Fraud Label (Ground Truth)", value="Fraudulent" if selected_txn["actual_is_fraud_label"] == 1 else "Legitimate")

st.divider()

# Middle Row: Explainability Chart & Details
# NEW: Renamed for a more analyst-native tone
st.markdown("### Primary Behavioral Risk Drivers")

col_chart, col_details = st.columns([2, 1])

with col_chart:
    # Prepare data for Altair chart
    df_shap = pd.DataFrame(selected_txn["top_risk_drivers"])
    df_shap = df_shap.sort_values(by="shap_impact", ascending=True) # Sort for horizontal bar chart
    
    # Create a clean bar chart
    chart = alt.Chart(df_shap).mark_bar(color='#E44D26').encode(
        x=alt.X('shap_impact:Q', title='Impact on Risk Score (SHAP Value)'),
        y=alt.Y('feature:N', sort='-x', title=''),
        tooltip=['feature', 'actual_value', 'shap_impact']
    ).properties(height=300)
    
    # NEW: Add a zero reference line for easier visual parsing
    zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='gray').encode(x='x:Q')
    
    st.altair_chart(chart + zero_line, use_container_width=True)

with col_details:
    st.markdown("#### Raw Feature Values")
    st.dataframe(
        df_shap[['feature', 'actual_value']].set_index('feature'),
        use_container_width=True
    )

st.divider()

# Bottom Row: GenAI Integration
st.markdown("### AI Analyst Summary")

# NEW: Governance disclaimer to calm regulatory/managerial nerves
st.caption(
    "GenAI is used strictly for explanation. "
    "All decisions are made upstream by a governed risk policy."
)

if st.button("Generate Narrative via GenAI", type="primary"):
    with st.spinner("Initializing local LLM and generating narrative... (This may take 10-45 seconds)"):
        # Call our generator
        result = generate_insight(selected_txn, decision)
        
        # Log the metrics
        audit_record = {
            "transaction_index": selected_txn["transaction_index"],
            "risk_score": score,
            "decision": decision,
            "llm_latency_ms": result["latency_ms"],
            "guardrail_passed": result["guardrail_passed"],
            "error": result["error"],
            "model_version": policy["model_artifact"].split("/")[-1],
            "policy_version": policy["policy_name"],
            "prompt_version": "v1.0"
        }
        log_generation(audit_record)
        
        # Display the UI result
        if result["guardrail_passed"]:
            st.success("Guardrails Passed: Explanation complies with data governance policies.")
            st.info(f"**Insight:** {result['narrative']}")
        else:
            st.error("Guardrail Failure: Unverified assumptions detected.")
            st.warning(result['narrative'])
            
        # NEW: Show product metrics inline (massive product signal)
        st.markdown("#### Generation Telemetry")
        st.code(
            f"Latency: {result['latency_ms']} ms\n"
            f"Guardrails Passed: {result['guardrail_passed']}\n"
            f"Prompt Version: v1.0",
            language="text"
        )
        
        # NEW: Provide a workflow action (shows deep operational awareness)
        st.markdown("#### Analyst Action")
        action = st.selectbox(
            "Next Step:",
            ["Auto-Decline", "Escalate to Manual Review", "Mark as False Positive"]
        )