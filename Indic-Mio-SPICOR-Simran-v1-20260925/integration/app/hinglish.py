from __future__ import annotations

import re


_NUMBER_WORDS = {
    "0": "शून्य",
    "1": "एक",
    "2": "दो",
    "3": "तीन",
    "4": "चार",
    "5": "पाँच",
    "6": "छह",
    "7": "सात",
    "8": "आठ",
    "9": "नौ",
}

_INDIAN_ENGLISH = {
    "payment": "पेमेंट",
    "receive": "रिसीव",
    "received": "रिसीव्ड",
    "successfully": "सक्सेसफुली",
    "help": "हेल्प",
    "account": "अकाउंट",
    "balance": "बैलेंस",
    "customer": "कस्टमर",
    "service": "सर्विस",
    "order": "ऑर्डर",
    "delivery": "डिलीवरी",
    "confirm": "कन्फर्म",
    "confirmed": "कन्फर्म्ड",
    "cancel": "कैंसल",
    "support": "सपोर्ट",
    "thank you": "थैंक यू",
    "okay": "ओके",
    "otp": "ओ टी पी",
}


def _speak_digits(match: re.Match[str]) -> str:
    return " ".join(_NUMBER_WORDS[digit] for digit in match.group(0))


def normalize_hinglish(text: str, language: str = "auto") -> str:
    """Normalize voice-bot Hinglish while preserving ordinary English input.

    Indic-Mio natively understands code mixing. This conservative pass focuses on
    the failure cases that matter most in calls: digits, OTPs, and frequent English
    loanwords embedded in Devanagari sentences.
    """

    text = re.sub(r"[ \t]+", " ", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\d+", _speak_digits, text)

    contains_devanagari = bool(re.search(r"[\u0900-\u097f]", text))
    if contains_devanagari or language.lower() in {"hi", "hindi", "hinglish"}:
        for source, target in sorted(
            _INDIAN_ENGLISH.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.I)
    return text
