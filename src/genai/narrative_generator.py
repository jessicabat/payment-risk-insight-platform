"""
Phase 3: GenAI Narrative Generator
----------------------------------
Calls a local LLM (Ollama) to translate SHAP values into readable insights.
Enforces strict guardrails to prevent hallucination of non-PaySim facts.
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# --- GUARDRAILS ---
FORBIDDEN_TERMS = [
    "country", "cross-border", "merchant category", 
    "location", "user identity", "ip address", "device"
]

def validate_guardrails(text: str) -> bool:
    """Ensures the LLM did not hallucinate data outside our payload."""
    lower_text = text.lower()
    return not any(term in lower_text for term in FORBIDDEN_TERMS)

# --- PROMPT TEMPLATE ---
def build_prompt(transaction: dict, decision: str) -> str:
    drivers_text = "\n".join(
        f"- {d['feature']}: actual value is {d['actual_value']:.2f} (SHAP Risk Impact: {d['shap_impact']:.2f})"
        for d in transaction["top_risk_drivers"]
    )
    
    return f"""
    You are an expert fraud analyst assistant. 
    Write a brief, professional 2-3 sentence summary explaining why this transaction was flagged.
    
    RULES:
    - ONLY use the data provided below.
    - DO NOT mention geography, merchants, or device data.
    - Be objective and concise.
    
    DATA:
    Decision: {decision}
    Risk Score: {transaction['predicted_risk_score']:.2f}
    
    Top Risk Drivers (Features pushing the score higher):
    {drivers_text}
    """

# --- GENERATOR ---
def generate_insight(transaction: dict, decision: str) -> str:
    prompt = build_prompt(transaction, decision)
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120 # Prevent hanging
        )
        response.raise_for_status()
        narrative = response.json().get("response", "").strip()
        
        # Apply Guardrails
        if validate_guardrails(narrative):
            return narrative
        else:
            return "GUARDRAIL TRIGGERED: The LLM generated a narrative containing unverified assumptions. Please review the structured SHAP drivers manually."
            
    except requests.exceptions.RequestException as e:
        return f"LLM Connection Error: Ensure Ollama is running. ({e})"