import json
import yaml
from pathlib import Path

base = Path(r"d:\somya jain\Voice_platform\use_cases\zipnach")

# 1. system_prompt.txt
prompt = """You are a helpful voice assistant for Zipnach. Your job is to collect details and answer queries. Be polite, concise, and professional."""
(base / "system_prompt.txt").write_text(prompt, encoding="utf-8")

# 2. use_case.yaml
use_case = {
    "name": "zipnach",
    "display_name": "Zipnach Voice Bot",
    "server_port": 8768,
    "asterisk": {
        "app_name": "zipnach_agent"
    },
    "tts": {
        "preset_id": "my_speaker_saavi"
    },
    "llm": {
        "model": "qwen2.5:1.5b"
    }
}
with open(base / "use_case.yaml", "w", encoding="utf-8") as f:
    yaml.dump(use_case, f, sort_keys=False)

# 3. entity_schema.json
entity_schema = {
  "fields": {
    "customer_name": {
      "type": "text"
    },
    "loan_status": {
      "type": "enum",
      "values": ["approved", "rejected", "pending"]
    },
    "verification_status": {
      "type": "boolean",
      "positive_phrases": ["verified", "done", "yes"],
      "negative_phrases": ["not verified", "pending", "no"]
    }
  },
  "required_order": [
    "customer_name",
    "loan_status"
  ]
}
with open(base / "entity_schema.json", "w", encoding="utf-8") as f:
    json.dump(entity_schema, f, indent=2)

# 4. keywords.json (standard defaults since it wasn't externalized)
keywords = {
  "WRONG_NUMBER_KEYWORDS": ["wrong number", "galat number"],
  "WRONG_PERSON_KEYWORDS": ["wrong person", "not me"],
  "DND_KEYWORDS": ["do not call", "stop calling", "dnd"],
  "CALL_LATER_KEYWORDS": ["call later", "busy", "baad mein"],
  "HUMAN_AGENT_KEYWORDS": ["talk to human", "agent", "customer care"],
  "EXIT_KEYWORDS": ["bye", "exit", "quit"],
  "REPEAT_KEYWORDS": ["repeat", "say again", "pardon"],
  "WAIT_KEYWORDS": ["wait", "hold on", "ek minute"],
  "HELP_KEYWORDS": ["help", "assist"],
  "WEAK_POSITIVES": ["yes", "yeah", "haan", "correct", "right"],
  "STRONG_COMPLETION": ["done", "completed", "ho gaya"],
  "NEGATIVE_KEYWORDS": ["no", "nah", "nahi", "wrong", "incorrect"]
}
with open(base / "keywords.json", "w", encoding="utf-8") as f:
    json.dump(keywords, f, indent=2)

print("Zipnach files created.")
