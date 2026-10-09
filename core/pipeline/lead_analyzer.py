"""
lead_analyzer.py — Lead Analysis Helper
========================================
Derives AI analysis fields (intent, sentiment, interest, recommended action)
from CRM fields and call transcripts.
"""


def _derive_ai_analysis(crm_fields: dict, transcript: str, lead_status: str) -> dict:
    temp = crm_fields.get("lead_temperature", "warm")
    interest_val = temp.capitalize() if isinstance(temp, str) and temp else "Warm"
    
    positive_statuses = {
        "Counselling Scheduled",
        "Campus Visit Scheduled",
        "Interested",
        "Success",
        "Completed",
    }
    
    sentiment_val = "Positive" if lead_status in positive_statuses else "Neutral"
    
    prog = crm_fields.get("program_interest")
    if isinstance(prog, list):
        prog_str = ", ".join(prog)
    else:
        prog_str = str(prog or "General Inquiry")

    if lead_status in ("Call Later", "Interested"):
        action = "Follow up via phone/WhatsApp"
    elif lead_status in ("Counselling Scheduled", "Campus Visit Scheduled"):
        action = "Prepare counselling / visit materials"
    else:
        action = "No immediate follow-up required"

    return {
        "intent": prog_str if prog_str else "General Inquiry",
        "interest": interest_val,
        "sentiment": sentiment_val,
        "recommended_action": action,
        "recommendedAction": action,
    }
