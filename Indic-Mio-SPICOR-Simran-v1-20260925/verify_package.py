#!/usr/bin/env python3
"""Validate that the deployable adapter package is complete and coherent."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "adapter/adapter_config.json": 100,
    "adapter/adapter_model.safetensors": 1_000_000,
    "adapter/tokenizer.json": 1_000_000,
    "adapter/tokenizer_config.json": 1_000,
    "adapter/added_tokens.json": 1_000,
    "adapter/merges.txt": 1_000,
    "adapter/vocab.json": 1_000,
    "voices/simran_clone_indian.mp3": 10_000,
    "config/production-config.json": 100,
    "config/voices.json": 100,
    "integration/scripts/start-backend.sh": 100,
    "integration/scripts/start-finetuned-simran.sh": 100,
    "integration/scripts/start-finetuned-gateway.sh": 100,
    "integration/app/main.py": 1_000,
    "integration/app/hinglish.py": 100,
}

EXPECTED_SHA256 = {
    "adapter/adapter_model.safetensors": "400b0fc02378a28c78b464a63981f9a5340908a4434c8d6bb174d10b6847cbf4",
    "adapter/adapter_config.json": "871c06791bfb26cb598997ddfe91d506a0b4ee554a6fac96134fde1077700723",
    "voices/simran_clone_indian.mp3": "0a5ea578b8ab220a152b3043f4f1c82f7fe790e51d9e63c16c89950dd347d958",
    "config/production-config.json": "f2989976bcca80920db3419e97bc7be0e4d9729be1df9978a9af0ca4b0bfef2e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    for relative, minimum_size in REQUIRED.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif path.stat().st_size < minimum_size:
            errors.append(f"unexpectedly small: {relative}")

    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if path.is_file() and sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")

    try:
        adapter = json.loads((ROOT / "adapter/adapter_config.json").read_text())
        if adapter.get("peft_type") != "LORA":
            errors.append("adapter_config.json is not a LoRA adapter")
        if adapter.get("r") != 16 or adapter.get("lora_alpha") != 32:
            errors.append("adapter rank/alpha differs from the selected model")
    except Exception as exc:
        errors.append(f"invalid adapter config: {exc}")

    try:
        production = json.loads(
            (ROOT / "config/production-config.json").read_text(encoding="utf-8")
        )
        if production.get("adapter_name") != "indic-mio-spicor-fullshort-stage2":
            errors.append("production adapter name is incorrect")
        if production.get("voice", {}).get("id") != "simran_clone_indian":
            errors.append("production voice id is incorrect")
    except Exception as exc:
        errors.append(f"invalid production config: {exc}")

    if errors:
        print("PACKAGE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PACKAGE CHECK PASSED")
    print("Adapter: indic-mio-spicor-fullshort-stage2")
    print("Voice: simran_clone_indian")
    print("The package is complete. External base-model/runtime checks happen during installation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
