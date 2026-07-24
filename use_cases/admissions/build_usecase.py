import json
import yaml
from pathlib import Path

base = Path(r"d:\somya jain\Voice_platform\use_cases\admissions")

# 1. system_prompt.txt
prompt = """You are an admissions assistant for Management Education & Research Institute (MERI), Janakpuri, Delhi.
Your job is to answer short queries about MBA and PGDM programs. Be polite, concise, and professional."""
(base / "system_prompt.txt").write_text(prompt, encoding="utf-8")

# 2. use_case.yaml
use_case = {
    "name": "admissions",
    "display_name": "MERI Admissions Bot",
    "server_port": 8767,
    "asterisk": {
        "app_name": "admissions_agent"
    },
    "tts": {
        "preset_id": "my_speaker_saavi"
    },
    "llm": {
        "model": "qwen2.5:1.5b"
    },
    "extra_kb_dirs": ["MERI_KB"]
}
with open(base / "use_case.yaml", "w", encoding="utf-8") as f:
    yaml.dump(use_case, f, sort_keys=False)

# 3. entity_schema.json
entity_schema = {
  "fields": {
    "graduation_status": {
      "type": "enum",
      "values": ["completed", "final_year", "early_year"],
      "allow_overwrite": False
    },
    "exam_status": {
      "type": "enum",
      "values": ["appeared", "not_appeared"],
      "allow_overwrite": False
    },
    "exam_name": {
      "type": "enum",
      "values": ["CAT", "MAT", "CMAT", "XAT", "GMAT", "SNAP", "NMAT"],
      "allow_overwrite": False
    },
    "program_interest": {
      "type": "enum",
      "values": ["mba", "pgdm", "both"]
    },
    "specialization": {
      "type": "enum",
      "values": ["finance", "marketing", "hr", "operations", "analytics"]
    },
    "counselling_mode": {
      "type": "enum",
      "values": ["campus_visit", "online", "info_only", "undecided"]
    }
  },
  "required_order": [
    "graduation_status",
    "exam_status",
    "counselling_mode"
  ]
}
with open(base / "entity_schema.json", "w", encoding="utf-8") as f:
    json.dump(entity_schema, f, indent=2)

# 4. flow.json
flow = {
  "initial_state": "CALL_CONNECTED",
  "llm_fallback_states": [
    "QUALIFICATION_Q1",
    "QUALIFICATION_Q2",
    "VISIT_OPTION",
    "PROGRAM_SELECTION"
  ],
  "states": {
    "CALL_CONNECTED": {
      "script_key": "outbound_intro",
      "transitions": {
        "affirm": "QUALIFICATION_Q1",
        "deny": "CLOSING",
        "default": "QUALIFICATION_Q1"
      }
    },
    "QUALIFICATION_Q1": {
      "script_key": "qualification_question_1",
      "llm_fallback": True,
      "transitions": {
        "grad_completed": "QUALIFICATION_Q2",
        "grad_final_year": "QUALIFICATION_Q2",
        "grad_early": "CLOSING",
        "default": "QUALIFICATION_Q1"
      }
    },
    "QUALIFICATION_Q2": {
      "script_key": "qualification_question_2",
      "llm_fallback": True,
      "transitions": {
        "exam_given": "PROGRAM_OVERVIEW",
        "exam_not_given": "PROGRAM_OVERVIEW",
        "default": "PROGRAM_OVERVIEW"
      }
    },
    "PROGRAM_OVERVIEW": {
      "script_key": "program_overview_intro",
      "transitions": {
        "default": "PROGRAM_SELECTION"
      }
    },
    "PROGRAM_SELECTION": {
      "script_key": "program_selection_question",
      "llm_fallback": True,
      "transitions": {
        "program_mba": "VISIT_OPTION",
        "program_pgdm": "VISIT_OPTION",
        "program_both": "VISIT_OPTION",
        "default": "VISIT_OPTION"
      }
    },
    "VISIT_OPTION": {
      "script_key": "visit_option",
      "llm_fallback": True,
      "transitions": {
        "visit_campus": "VISIT_SLOT_CAPTURE",
        "visit_online": "WHATSAPP_READBACK",
        "default": "WHATSAPP_READBACK"
      }
    },
    "VISIT_SLOT_CAPTURE": {
      "script_key": "visit_slot_campus",
      "transitions": {
        "default": "WHATSAPP_READBACK"
      }
    },
    "WHATSAPP_READBACK": {
      "script_key": "whatsapp_readback",
      "transitions": {
        "affirm": "WHATSAPP_CONFIRMATION",
        "deny": "CLOSING",
        "default": "WHATSAPP_CONFIRMATION"
      }
    },
    "WHATSAPP_CONFIRMATION": {
      "script_key": "whatsapp_confirmation",
      "transitions": {
        "default": "CLOSING"
      }
    },
    "CLOSING": {
      "script_key": "closing",
      "terminal": True
    }
  }
}
with open(base / "flow.json", "w", encoding="utf-8") as f:
    json.dump(flow, f, indent=2)

print("Admissions files created.")
