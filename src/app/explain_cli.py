"""
Phase 3: CLI Interface for GenAI Explanations
---------------------------------------------
Simulates an analyst dashboard requesting an explanation for a high-risk transaction.
"""

import json
import sys
from pathlib import Path
from src.audit.logger import log_generation

# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.genai.narrative_generator import generate_insight

SHAP_FILE = Path("artifacts/explainability/xgb_v1_sample_explanations.json")
POLICY_FILE = Path("artifacts/policies/decision_policy_v1.json")

def main():
    print("Loading Policy and Transaction Data...")
    
    with open(POLICY_FILE) as f:
        policy = json.load(f)
        
    with open(SHAP_FILE) as f:
        samples = json.load(f)
        
    # Grab the riskiest transaction (Index 0 from our Phase 2 output)
    target_txn = samples[0]
    score = target_txn["predicted_risk_score"]
    
    # Apply the Policy
    if score >= policy["threshold"]:
        decision = policy["decision_labels"]["block"]
    else:
        decision = policy["decision_labels"]["allow"]

    print("\n" + "="*50)
    print(f" TRANSACTION: {target_txn['transaction_index']}")
    print(f" SCORE:       {score:.4f} (Threshold: {policy['threshold']})")
    print(f" POLICY:      {decision}")
    print("="*50)
    
    print("\nGenerating AI Insight via local LLM... (This may take 10-20 seconds)\n")
    
    # generate_insight now returns a dictionary
    result = generate_insight(target_txn, decision)
    
    # 1. Print for the user
    print(" [ AI INSIGHT ]")
    print(f" {result['narrative']}")
    print("\n" + "="*50)
    
    # 2. Package and log the Product Metrics
    audit_record = {
        "transaction_index": target_txn["transaction_index"],
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
    print(f"Metrics logged to artifacts/audit/generation_logs.jsonl (Latency: {result['latency_ms']}ms)")

if __name__ == "__main__":
    main()