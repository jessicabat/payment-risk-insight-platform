"""
Phase 3: Audit & Telemetry Logger
---------------------------------
Appends GenAI generation metrics to a JSONL file for product analytics.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

# We use JSONL (JSON Lines) because it is append-only and highly scalable
LOG_FILE = Path("artifacts/audit/generation_logs.jsonl")

def log_generation(audit_payload: dict):
    """Appends a single generation event to the audit log."""
    # Ensure directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Inject an absolute UTC timestamp
    audit_payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(audit_payload) + "\n")