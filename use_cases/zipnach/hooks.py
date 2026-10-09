def set_language(session, intent_name, intent_data, memory):
    if intent_name == "choose_english":
        session.language = "en"
        print(f"[Hooks] User chose English for session {session.session_id}")
    elif intent_name == "choose_hindi":
        session.language = "hi"
        print(f"[Hooks] User chose Hindi for session {session.session_id}")
    return None
