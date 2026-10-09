#!/usr/bin/env python3
"""Install this handoff package into a compatible MOSS-TTS checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ADAPTER_NAME = "indic-mio-spicor-fullshort-stage2"
VOICE_ID = "simran_clone_indian"


def copy_with_backup(source: Path, target: Path, backup_root: Path) -> None:
    if target.exists():
        backup = backup_root / target.name
        counter = 1
        while backup.exists():
            backup = backup_root / f"{target.stem}-{counter}{target.suffix}"
            counter += 1
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Recipient MOSS-TTS directory")
    args = parser.parse_args()
    target = Path(args.target).expanduser().resolve()

    required_target = [
        target / "models/Indic-Mio/config.json",
        target / "models/Indic-Mio/model.safetensors",
        target / "vendor/MioTTS-Inference/run_server.py",
        target / "scripts/start-backend.sh",
    ]
    missing = [str(path) for path in required_target if not path.exists()]
    if missing:
        print("TARGET CHECK FAILED. Set up MOSS-TTS and Indic-Mio first:")
        for path in missing:
            print(f"- missing: {path}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = target / "package-backup" / timestamp

    adapter_target = target / "finetuning/runs" / ADAPTER_NAME
    adapter_target.mkdir(parents=True, exist_ok=True)
    for source in (PACKAGE / "adapter").iterdir():
        if source.is_file():
            shutil.copy2(source, adapter_target / source.name)

    voice_target = target / "data/voices" / f"{VOICE_ID}.mp3"
    voice_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGE / "voices/simran_clone_indian.mp3", voice_target)

    # Install the Bhavyaa voice preset
    bhavyaa_preset = PACKAGE / "presets/bhavyaa_1_0.pt"
    if bhavyaa_preset.exists():
        preset_target = target / "vendor/MioTTS-Inference/presets/bhavyaa_1_0.pt"
        preset_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bhavyaa_preset, preset_target)

    voices_path = target / "data/voices.json"
    voices = {}
    if voices_path.exists():
        voices = json.loads(voices_path.read_text(encoding="utf-8"))
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(voices_path, backup / "voices.json")
    packaged_voice = json.loads(
        (PACKAGE / "config/voices.json").read_text(encoding="utf-8")
    )[VOICE_ID]
    voices[VOICE_ID] = packaged_voice
    voices_path.parent.mkdir(parents=True, exist_ok=True)
    voices_path.write_text(
        json.dumps(voices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    replacements = {
        PACKAGE / "integration/scripts/start-backend.sh": target / "scripts/start-backend.sh",
        PACKAGE / "integration/scripts/start-finetuned-simran.sh": target / "scripts/start-finetuned-simran.sh",
        PACKAGE / "integration/scripts/start-finetuned-gateway.sh": target / "scripts/start-finetuned-gateway.sh",
        PACKAGE / "integration/app/main.py": target / "app/main.py",
        PACKAGE / "integration/app/hinglish.py": target / "app/hinglish.py",
        PACKAGE / "integration/app/__init__.py": target / "app/__init__.py",
        PACKAGE / "integration/requirements.txt": target / "requirements.package.txt",
    }
    backup.mkdir(parents=True, exist_ok=True)
    for source, destination in replacements.items():
        copy_with_backup(source, destination, backup)

    installed_config = target / "data/indic-mio-spicor-production-config.json"
    shutil.copy2(PACKAGE / "config/production-config.json", installed_config)

    print("INSTALLATION COMPLETE")
    print(f"Target: {target}")
    print(f"Adapter: {adapter_target}")
    print(f"Voice: {voice_target}")
    print(f"Backups: {backup}")
    print("\nStart terminal 1:")
    print("  bash scripts/start-finetuned-simran.sh")
    print("Start terminal 2:")
    print("  bash scripts/start-finetuned-gateway.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
