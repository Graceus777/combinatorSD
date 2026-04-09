"""
LoRA Scanner - Scans a folder for .safetensors LoRA files
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRA:
    """Represents a LoRA file"""
    name: str
    filepath: str
    weight: float = 1.0
    prompt: str = ""  # Optional trigger words/prompt

    def to_tag(self) -> str:
        """Generate the LoRA tag for the prompt"""
        return f"<lora:{self.name}:{self.weight}>"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "filepath": self.filepath,
            "weight": self.weight,
            "prompt": self.prompt
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoRA":
        return cls(
            name=data["name"],
            filepath=data.get("filepath", ""),
            weight=data.get("weight", 1.0),
            prompt=data.get("prompt", "")
        )


def read_activation_text(lora_path: str) -> str:
    """
    Read activation/trigger text from companion .txt file.
    SD WebUI convention: lora_name.safetensors has lora_name.txt with trigger words.
    """
    txt_path = Path(lora_path).with_suffix('.txt')
    if txt_path.exists():
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def scan_lora_folder(folder_path: str) -> List[LoRA]:
    """
    Scan a folder for .safetensors files and return a list of LoRA objects.
    Also reads companion .txt files for activation/trigger words.

    Args:
        folder_path: Path to the LoRA folder (e.g., models/Lora in SD WebUI)
        Supports local paths and UNC paths (\\\\server\\share\\...)

    Returns:
        List of LoRA objects found in the folder
    """
    loras = []

    if not folder_path:
        return loras

    # Handle UNC paths and regular paths
    folder = Path(folder_path)

    if not folder.exists():
        return loras

    # Scan for .safetensors files (most common LoRA format)
    # Also check for .pt and .ckpt files (older formats)
    extensions = [".safetensors", ".pt", ".ckpt"]

    for ext in extensions:
        for filepath in folder.rglob(f"*{ext}"):
            # Extract name from filename (without extension)
            name = filepath.stem

            # Skip files that look like full models (usually much larger)
            # LoRAs are typically under 500MB
            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                if size_mb > 500:
                    continue
            except OSError:
                continue

            # Read activation text from companion .txt file
            activation_text = read_activation_text(str(filepath))

            lora = LoRA(
                name=name,
                filepath=str(filepath),
                weight=1.0,
                prompt=activation_text
            )
            loras.append(lora)

    # Sort by name
    loras.sort(key=lambda x: x.name.lower())

    return loras


def get_lora_names(loras: List[LoRA]) -> List[str]:
    """Get just the names from a list of LoRAs"""
    return [lora.name for lora in loras]


if __name__ == "__main__":
    # Test: Scan a folder if provided
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"Scanning: {folder}")
        loras = scan_lora_folder(folder)
        print(f"Found {len(loras)} LoRAs:")
        for lora in loras:
            print(f"  - {lora.name}")
    else:
        print("Usage: python lora_scanner.py <lora_folder_path>")
