"""
LoRA Combinator SD - Gradio Web Interface
Drag-and-drop style LoRA management for Stable Diffusion image generation
"""
import gradio as gr
import json
import os
import random
import hashlib
import base64
import io as _io
import itertools
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from lora_scanner import LoRA, scan_lora_folder
from api_client import A1111Client, build_payload


# --- Configuration ---
CONFIG_FILE = "config.json"
HISTORY_FILE = "generation_history.jsonl"
DEFAULT_CONFIG = {
    "api_url": "http://192.168.0.238:7860/",
    "lora_folder": "",
    "output_dir": "generated_images",
    "last_zone_config": "",
    "last_prompt_config": "",
    "defaults": {
        "model": "",
        "positive_prompt": "masterpiece, best quality, sharp focus, highres",
        "negative_prompt": "(low quality, worst quality:1.4)",
        "steps": 27,
        "sampler": "Euler a",
        "cfg_scale": 6.0,
        "width": 1024,
        "height": 1280,
        "batch_size": 1,
        "batch_count": 1,
        "random_count": 1,
        "cooldown": 0,
        "enable_hr": False,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "denoising_strength": 0.5,
        "enable_adetailer": False,
        "skip_exists": False
    }
}

# Local folder for LoRA activation text files
LORA_TEXTS_DIR = Path(__file__).parent / "lora_texts"


def get_activation_text(lora_name: str) -> str:
    """
    Read activation/trigger text from local file.
    Checks lora_texts/{lora_name}.json first, then .txt as fallback.
    """
    # Try .json (Civitai-style metadata)
    json_path = LORA_TEXTS_DIR / f"{lora_name}.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("activation text", "").strip()
        except Exception:
            pass

    # Fallback to plain .txt
    txt_path = LORA_TEXTS_DIR / f"{lora_name}.txt"
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding='utf-8').strip()
        except Exception:
            pass
    return ""


def load_config() -> dict:
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def save_session_settings(
    api_url, model_name, positive_prompt, negative_prompt,
    steps, sampler, cfg, width, height,
    enable_hr, hr_scale, hr_upscaler, denoising,
    enable_adetailer, random_count, batch_size, batch_count, cooldown,
    skip_exists=False
):
    """Save all current UI settings to config for next session"""
    global config
    config["api_url"] = api_url
    config["defaults"] = {
        "model": model_name or "",
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "steps": int(steps),
        "sampler": sampler,
        "cfg_scale": float(cfg),
        "width": int(width),
        "height": int(height),
        "batch_size": int(batch_size),
        "batch_count": int(batch_count),
        "random_count": int(random_count),
        "cooldown": int(cooldown),
        "enable_hr": bool(enable_hr),
        "hr_scale": float(hr_scale),
        "hr_upscaler": hr_upscaler,
        "denoising_strength": float(denoising),
        "enable_adetailer": bool(enable_adetailer),
        "skip_exists": bool(skip_exists),
    }
    save_config(config)


# --- Generation History ---
def load_history() -> List[Dict]:
    """Load generation history from JSONL file"""
    records = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def append_history_record(record: Dict):
    """Append a single generation record to history"""
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def compute_generation_hash(
    prompt: str, negative_prompt: str, steps: int, sampler: str,
    cfg: float, width: int, height: int, enable_hr: bool,
    hr_scale: float, hr_upscaler: str, denoising: float,
    enable_adetailer: bool, batch_size: int
) -> str:
    """Hash generation parameters for duplicate detection (excludes seed)"""
    key_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": int(steps),
        "sampler": sampler,
        "cfg": float(cfg),
        "width": int(width),
        "height": int(height),
        "enable_hr": bool(enable_hr),
        "batch_size": int(batch_size),
        "enable_adetailer": bool(enable_adetailer),
    }
    if enable_hr:
        key_data["hr_scale"] = float(hr_scale)
        key_data["hr_upscaler"] = hr_upscaler
        key_data["denoising"] = float(denoising)
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def build_frequency_map(history: List[Dict]) -> Dict[str, int]:
    """Count how many times each LoRA has been used across all history"""
    freq = {}
    for record in history:
        for lora in record.get("loras", []):
            freq[lora] = freq.get(lora, 0) + 1
    return freq


def weighted_sample(pool: list, weights: list, k: int) -> list:
    """Weighted random sample without replacement"""
    pool = list(pool)
    weights = list(weights)
    selected = []
    for _ in range(min(k, len(pool))):
        if not pool:
            break
        total = sum(weights)
        if total <= 0:
            idx = random.randint(0, len(pool) - 1)
        else:
            r = random.uniform(0, total)
            cumulative = 0
            idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r:
                    idx = i
                    break
        selected.append(pool[idx])
        pool.pop(idx)
        weights.pop(idx)
    return selected


# --- State Management ---
@dataclass
class LoRAEntry:
    """LoRA entry with weight and activation text"""
    weight: float = 1.0
    activation: str = ""


@dataclass
class AppState:
    """Application state"""
    all_loras: List[LoRA] = field(default_factory=list)
    bank_loras: List[str] = field(default_factory=list)  # Names in bank
    always_loras: Dict[str, LoRAEntry] = field(default_factory=dict)  # name -> LoRAEntry
    loop_loras: Dict[str, LoRAEntry] = field(default_factory=dict)
    random_loras: Dict[str, LoRAEntry] = field(default_factory=dict)
    is_running: bool = False
    should_stop: bool = False
    current_job: int = 0
    total_jobs: int = 0

    def get_lora_by_name(self, name: str) -> Optional[LoRA]:
        """Find a LoRA by name"""
        for lora in self.all_loras:
            if lora.name == name:
                return lora
        return None


# Global state
state = AppState()
config = load_config()
client = A1111Client(config["api_url"])


def refresh_loras(folder_path: str) -> Tuple[str, gr.update]:
    """Scan folder and refresh LoRA list"""
    global state, config

    if not folder_path:
        return "Please enter a LoRA folder path", gr.update(choices=[])

    state.all_loras = scan_lora_folder(folder_path)
    state.bank_loras = [lora.name for lora in state.all_loras]

    # Clear zones
    state.always_loras.clear()
    state.loop_loras.clear()
    state.random_loras.clear()

    # Save folder to config
    config["lora_folder"] = folder_path
    save_config(config)


def extract_activation_from_metadata(metadata: dict) -> str:
    """Extract activation/trigger words from LoRA metadata"""
    if not metadata:
        return ""

    # First check explicit trigger keys
    trigger_keys = [
        "ss_trigger_words",
        "trigger",
        "triggers",
        "activation text",
        "activation_text",
    ]

    for key in trigger_keys:
        if key in metadata and metadata[key]:
            val = metadata[key]
            if isinstance(val, list):
                return ", ".join(val)
            return str(val)

    # Try to extract from ss_tag_frequency (training tags)
    # The most common unique tags are likely triggers
    tag_freq = metadata.get("ss_tag_frequency", {})
    if tag_freq:
        # Common generic tags to filter out
        generic_tags = {
            "masterpiece", "best quality", "high quality", "highres",
            "1girl", "1boy", "solo", "looking at viewer", "smile",
            "long hair", "short hair", "breasts", "blush", "simple background",
            "white background", "nude", "nipples", "pussy", "penis",
        }

        # Flatten tag frequencies from all folders
        all_tags = {}
        for folder, tags in tag_freq.items():
            if isinstance(tags, dict):
                for tag, count in tags.items():
                    tag_lower = tag.lower()
                    if tag_lower not in generic_tags:
                        all_tags[tag] = all_tags.get(tag, 0) + count

        if all_tags:
            # Get the most common non-generic tag
            sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
            # Return top most common unique tags
            top_tags = [t[0] for t in sorted_tags[:7] if t[1] > 6]
            if top_tags:
                return ", ".join(top_tags)

    return ""


def fetch_loras_from_api(api_url: str) -> Tuple[str, gr.update]:
    """Fetch LoRAs from SD WebUI API instead of scanning folder"""
    global state, client

    if not api_url:
        return "Please enter API URL", gr.update(choices=[])

    client = A1111Client(api_url)

    # Test connection first
    success, msg = client.test_connection()
    if not success:
        return f"API Error: {msg}", gr.update(choices=[])

    # Fetch LoRAs
    api_loras = client.get_loras()
    if not api_loras:
        return "No LoRAs found (or API endpoint not available)", gr.update(choices=[])

    # Convert to our LoRA format, extracting metadata if available
    state.all_loras = []
    activations_found = 0

    for lora_data in api_loras:
        # Try to extract activation text from metadata
        metadata = lora_data.get("metadata", {})
        activation = extract_activation_from_metadata(metadata)
        if activation:
            activations_found += 1

        lora = LoRA(
            name=lora_data.get("name", lora_data.get("alias", "")),
            filepath=lora_data.get("path", ""),
            weight=1.0,
            prompt=activation
        )
        state.all_loras.append(lora)

    state.bank_loras = [lora.name for lora in state.all_loras]

    # Clear zones
    state.always_loras.clear()
    state.loop_loras.clear()
    state.random_loras.clear()

    count = len(state.all_loras)
    status_msg = f"Found {count} LoRAs"
    if activations_found > 0:
        status_msg += f" ({activations_found} with trigger words)"
    else:
        status_msg += " (no trigger words in metadata - check console for API response)"
    return (
        status_msg,
        gr.update(choices=state.bank_loras, value=[])
    )


def save_zone_config(config_name: str) -> str:
    """Save current zone configuration to a JSON file"""
    if not config_name:
        return "Please enter a config name"

    config_data = {
        "always": {name: {"weight": entry.weight, "activation": entry.activation}
                   for name, entry in state.always_loras.items()},
        "loop": {name: {"weight": entry.weight, "activation": entry.activation}
                 for name, entry in state.loop_loras.items()},
        "random": {name: {"weight": entry.weight, "activation": entry.activation}
                   for name, entry in state.random_loras.items()},
    }

    config_dir = "configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    filepath = os.path.join(config_dir, f"{config_name}.json")
    with open(filepath, 'w') as f:
        json.dump(config_data, f, indent=2)

    return f"Saved LoRA config: {filepath}"


def load_zone_config(config_name: str) -> Tuple[str, str, str, str]:
    """Load zone configuration from a JSON file"""
    global state, config

    if not config_name:
        return "(empty)", "(empty)", "(empty)", "Please enter a config name"

    # Remember last-used zone config
    config["last_zone_config"] = config_name
    save_config(config)

    filepath = os.path.join("configs", f"{config_name}.json")
    if not os.path.exists(filepath):
        return "(empty)", "(empty)", "(empty)", f"Config not found: {filepath}"

    try:
        with open(filepath, 'r') as f:
            config_data = json.load(f)

        # Clear current zones
        state.always_loras.clear()
        state.loop_loras.clear()
        state.random_loras.clear()

        # Load zones (check local activation files as fallback)
        for name, data in config_data.get("always", {}).items():
            activation = data.get("activation", "") or get_activation_text(name)
            state.always_loras[name] = LoRAEntry(
                weight=data.get("weight", 1.0),
                activation=activation
            )
        for name, data in config_data.get("loop", {}).items():
            activation = data.get("activation", "") or get_activation_text(name)
            state.loop_loras[name] = LoRAEntry(
                weight=data.get("weight", 1.0),
                activation=activation
            )
        for name, data in config_data.get("random", {}).items():
            activation = data.get("activation", "") or get_activation_text(name)
            state.random_loras[name] = LoRAEntry(
                weight=data.get("weight", 1.0),
                activation=activation
            )

        return (
            format_zone(state.always_loras),
            format_zone(state.loop_loras),
            format_zone(state.random_loras),
            f"Loaded LoRA config: {config_name}"
        )
    except Exception as e:
        return "(empty)", "(empty)", "(empty)", f"Error loading config: {e}"


def save_prompt_config(config_name: str, positive_prompt: str, negative_prompt: str) -> str:
    """Save prompt configuration to a JSON file"""
    if not config_name:
        return "Please enter a config name"

    config_data = {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
    }

    config_dir = "configs/prompts"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    filepath = os.path.join(config_dir, f"{config_name}.json")
    with open(filepath, 'w') as f:
        json.dump(config_data, f, indent=2)

    return f"Saved prompt config: {config_name}"


def load_prompt_config(config_name: str) -> Tuple[str, str, str]:
    """Load prompt configuration from a JSON file"""
    global config

    if not config_name:
        return "", "", "Please enter a config name"

    # Remember last-used prompt config
    config["last_prompt_config"] = config_name
    save_config(config)

    filepath = os.path.join("configs/prompts", f"{config_name}.json")
    if not os.path.exists(filepath):
        return "", "", f"Config not found: {config_name}"

    try:
        with open(filepath, 'r') as f:
            config_data = json.load(f)

        return (
            config_data.get("positive_prompt", ""),
            config_data.get("negative_prompt", ""),
            f"Loaded prompt config: {config_name}"
        )
    except Exception as e:
        return "", "", f"Error loading config: {e}"


def list_prompt_configs() -> List[str]:
    """List all saved prompt configuration files"""
    config_dir = "configs/prompts"
    if not os.path.exists(config_dir):
        return []
    configs = [f.replace(".json", "") for f in os.listdir(config_dir) if f.endswith(".json")]
    return configs


def list_saved_configs() -> List[str]:
    """List all saved configuration files"""
    config_dir = "configs"
    if not os.path.exists(config_dir):
        return []
    configs = [f.replace(".json", "") for f in os.listdir(config_dir) if f.endswith(".json")]
    return configs


def add_to_zone(
    selected: List[str],
    zone: str,
    default_weight: float,
    always_state: str,
    loop_state: str,
    random_state: str
) -> Tuple[gr.update, str, str, str, gr.update, gr.update, gr.update, str]:
    """Move selected LoRAs from bank to a zone"""
    global state

    if not selected:
        return (
            gr.update(),
            always_state,
            loop_state,
            random_state,
            gr.update(),
            gr.update(),
            gr.update(),
            "No LoRAs selected"
        )

    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone)

    if zone_dict is None:
        return (
            gr.update(),
            always_state,
            loop_state,
            random_state,
            gr.update(),
            gr.update(),
            gr.update(),
            "Invalid zone"
        )

    # Move selected to zone
    for name in selected:
        if name in state.bank_loras:
            # Get activation text: local files first, then API metadata
            lora = state.get_lora_by_name(name)
            activation = get_activation_text(name) or (lora.prompt if lora else "")
            zone_dict[name] = LoRAEntry(weight=default_weight, activation=activation)
            state.bank_loras.remove(name)

    return (
        gr.update(choices=state.bank_loras, value=[]),
        format_zone(state.always_loras),
        format_zone(state.loop_loras),
        format_zone(state.random_loras),
        gr.update(choices=get_zone_lora_choices("always")),
        gr.update(choices=get_zone_lora_choices("loop")),
        gr.update(choices=get_zone_lora_choices("random")),
        f"Added {len(selected)} LoRA(s) to {zone}"
    )


def remove_from_zone(
    zone: str,
    lora_name: str
) -> Tuple[gr.update, str, str, str, gr.update, gr.update, gr.update, str]:
    """Remove a LoRA from a zone back to bank"""
    global state

    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone)

    if zone_dict and lora_name in zone_dict:
        del zone_dict[lora_name]
        state.bank_loras.append(lora_name)
        state.bank_loras.sort()

    return (
        gr.update(choices=state.bank_loras),
        format_zone(state.always_loras),
        format_zone(state.loop_loras),
        format_zone(state.random_loras),
        gr.update(choices=get_zone_lora_choices("always")),
        gr.update(choices=get_zone_lora_choices("loop")),
        gr.update(choices=get_zone_lora_choices("random")),
        f"Removed {lora_name}"
    )


def clear_zone(zone: str) -> Tuple[gr.update, str, str, str, gr.update, gr.update, gr.update, str]:
    """Move all LoRAs from a single zone back to bank"""
    global state

    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone, {})

    count = len(zone_dict)
    for name in list(zone_dict.keys()):
        state.bank_loras.append(name)
    zone_dict.clear()

    state.bank_loras.sort()

    return (
        gr.update(choices=state.bank_loras, value=[]),
        format_zone(state.always_loras),
        format_zone(state.loop_loras),
        format_zone(state.random_loras),
        gr.update(choices=get_zone_lora_choices("always")),
        gr.update(choices=get_zone_lora_choices("loop")),
        gr.update(choices=get_zone_lora_choices("random")),
        f"Cleared {count} LoRA(s) from {zone}"
    )


def clear_all_zones() -> Tuple[gr.update, str, str, str, gr.update, gr.update, gr.update, str]:
    """Move all LoRAs from all zones back to bank"""
    global state

    count = len(state.always_loras) + len(state.loop_loras) + len(state.random_loras)

    for zone_dict in [state.always_loras, state.loop_loras, state.random_loras]:
        for name in list(zone_dict.keys()):
            state.bank_loras.append(name)
        zone_dict.clear()

    state.bank_loras.sort()

    return (
        gr.update(choices=state.bank_loras, value=[]),
        format_zone(state.always_loras),
        format_zone(state.loop_loras),
        format_zone(state.random_loras),
        gr.update(choices=[]),
        gr.update(choices=[]),
        gr.update(choices=[]),
        f"Cleared {count} LoRA(s) from all zones"
    )


def format_zone(zone_dict: Dict[str, LoRAEntry]) -> str:
    """Format zone contents for display"""
    if not zone_dict:
        return "(empty)"
    lines = []
    for name, entry in zone_dict.items():
        line = f"{name} (w={entry.weight})"
        if entry.activation:
            # Show truncated activation text
            act_preview = entry.activation[:30] + "..." if len(entry.activation) > 30 else entry.activation
            line += f" [{act_preview}]"
        lines.append(line)
    return "\n".join(lines)


def update_weight(zone: str, lora_name: str, weight: float) -> str:
    """Update the weight of a LoRA in a zone"""
    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone)

    if zone_dict and lora_name in zone_dict:
        zone_dict[lora_name].weight = weight
        return format_zone(zone_dict)
    return "(error)"


def get_zone_lora_choices(zone: str) -> list:
    """Get list of LoRA names in a zone for dropdown"""
    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone, {})
    return list(zone_dict.keys())


def get_lora_weight_in_zone(zone: str, lora_name: str) -> gr.update:
    """Get current weight of a LoRA in a zone, return as slider update"""
    if not lora_name:
        return gr.update()

    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone, {})

    if lora_name in zone_dict:
        return gr.update(value=zone_dict[lora_name].weight)
    return gr.update(value=0.8)  # default


def update_zone_weight(zone: str, lora_name: str, weight: float) -> str:
    """Update weight and return updated display"""
    if not lora_name:
        return format_zone({
            "always": state.always_loras,
            "loop": state.loop_loras,
            "random": state.random_loras
        }.get(zone, {})) or "(empty)"

    zone_dict = {
        "always": state.always_loras,
        "loop": state.loop_loras,
        "random": state.random_loras
    }.get(zone)

    if zone_dict and lora_name in zone_dict:
        zone_dict[lora_name].weight = weight

    return format_zone(zone_dict) if zone_dict else "(empty)"


def build_prompt_preview(
    base_prompt: str,
    negative_prompt: str,
    random_count: int = 1,
    batch_size: int = 1,
    batch_count: int = 1
) -> str:
    """Build a preview of all prompts that will be generated"""
    if not state.always_loras and not state.loop_loras and not state.random_loras:
        if base_prompt:
            return f"Prompt: {base_prompt}"
        return "(no LoRAs selected, add some to zones first)"

    previews = []

    # Calculate total images
    loop_count = max(1, len(state.loop_loras))
    variants = loop_count
    total_images = variants * int(batch_count) * int(batch_size)

    previews.append(f"=== Will generate {total_images} image(s) ===")
    previews.append(f"    ({variants} variants × {int(batch_count)} batches × {int(batch_size)} per batch)\n")

    # Show what's constant
    if state.always_loras:
        always_parts = [f"<lora:{n}:{e.weight}>" for n, e in state.always_loras.items()]
        always_activations = [e.activation for e in state.always_loras.values() if e.activation]
        previews.append(f"ALWAYS: {', '.join(always_parts)}")
        if always_activations:
            previews.append(f"  Triggers: {', '.join(always_activations)}")

    # Show loop variants
    if state.loop_loras:
        previews.append(f"\nLOOP ({len(state.loop_loras)} variants):")
        for name, entry in state.loop_loras.items():
            line = f"  - <lora:{name}:{entry.weight}>"
            if entry.activation:
                line += f" [{entry.activation[:40]}...]" if len(entry.activation) > 40 else f" [{entry.activation}]"
            previews.append(line)

    # Show random pool
    if state.random_loras:
        previews.append(f"\nRANDOM (pick {random_count} from {len(state.random_loras)}):")
        for name, entry in state.random_loras.items():
            previews.append(f"  - {name} (w={entry.weight})")

    # Show base prompts
    if base_prompt:
        previews.append(f"\nBASE PROMPT: {base_prompt}")
    if negative_prompt:
        previews.append(f"NEGATIVE: {negative_prompt}")

    return "\n".join(previews)


def calculate_total_images(random_count: int) -> int:
    """Calculate total images that will be generated"""
    loop_count = max(1, len(state.loop_loras))
    return loop_count  # Each loop variant = 1 image (with random picks applied)


def generate_payloads(
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    sampler: str,
    cfg: float,
    width: int,
    height: int,
    enable_hr: bool,
    hr_scale: float,
    hr_upscaler: str,
    denoising: float,
    enable_adetailer: bool,
    random_count: int,
    batch_size: int = 1
) -> List[Dict[str, Any]]:
    """Generate all payloads based on current zone configuration"""
    payloads = []

    # Get loop combinations
    if state.loop_loras:
        loop_items = list(state.loop_loras.items())
    else:
        loop_items = [(None, None)]  # Single iteration if no loop LoRAs

    for loop_name, loop_entry in loop_items:
        # Build LoRA tags and collect activation text
        lora_tags = []
        activation_texts = []

        # Always LoRAs
        for name, entry in state.always_loras.items():
            lora_tags.append(f"<lora:{name}:{entry.weight}>")
            activation = get_activation_text(name) or entry.activation
            if activation:
                activation_texts.append(activation)

        # Current loop LoRA
        if loop_name and loop_entry:
            lora_tags.append(f"<lora:{loop_name}:{loop_entry.weight}>")
            activation = get_activation_text(loop_name) or loop_entry.activation
            if activation:
                activation_texts.append(activation)

        # Random LoRAs
        if state.random_loras and random_count > 0:
            random_pool = list(state.random_loras.items())
            picks = random.sample(
                random_pool,
                min(random_count, len(random_pool))
            )
            for name, entry in picks:
                lora_tags.append(f"<lora:{name}:{entry.weight}>")
                activation = get_activation_text(name) or entry.activation
                if activation:
                    activation_texts.append(activation)

        # Build final prompt: LoRA tags + activation texts + base prompt
        prompt_parts = lora_tags.copy()
        if activation_texts:
            prompt_parts.extend(activation_texts)
        if positive_prompt:
            prompt_parts.append(positive_prompt)
        final_prompt = ", ".join(prompt_parts)

        # Create filename
        name_parts = []
        if loop_name:
            name_parts.append(loop_name)
        if state.always_loras:
            name_parts.append("always" + str(len(state.always_loras)))
        if not name_parts:
            name_parts.append("image")

        payload = build_payload(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            sampler_name=sampler,
            cfg_scale=cfg,
            width=width,
            height=height,
            enable_hr=enable_hr,
            hr_scale=hr_scale,
            hr_upscaler=hr_upscaler,
            denoising_strength=denoising,
            enable_adetailer=enable_adetailer,
            batch_size=batch_size,
            custom_filename="_".join(name_parts)
        )
        payloads.append(payload)

    return payloads


def run_generation(
    api_url: str,
    model_name: str,
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    sampler: str,
    cfg: float,
    width: int,
    height: int,
    enable_hr: bool,
    hr_scale: float,
    hr_upscaler: str,
    denoising: float,
    enable_adetailer: bool,
    random_count: int,
    batch_size: int,
    batch_count: int,
    cooldown_seconds: int,
    skip_exists: bool,
    gen_mode_val: str = "txt2img",
    img2img_image_val=None,
    img2img_denoising_val: float = 0.75,
    img2img_resize_mode_val: str = "Just resize",
    enable_controlnet_val=False,
    control_image_val=None,
    control_preprocessor_val="none",
    control_model_val="controlnet-lineart-anime-sdxl-fp16",
    control_weight_val=1.0,
    control_guidance_start_val=0.0,
    control_guidance_end_val=1.0,
    control_mode_val="Balanced",
    progress=gr.Progress()
):
    """Run the generation queue with live preview - yields after each image"""
    global state, client, config

    if state.is_running:
        yield "Generation already in progress!", []
        return

    # Save all settings for next session
    client = A1111Client(api_url)
    save_session_settings(
        api_url, model_name, positive_prompt, negative_prompt,
        steps, sampler, cfg, width, height,
        enable_hr, hr_scale, hr_upscaler, denoising,
        enable_adetailer, random_count, batch_size, batch_count, cooldown_seconds,
        skip_exists
    )

    # img2img mode: validate and prepare input image
    is_img2img = gen_mode_val == "img2img"
    img2img_b64 = None
    resize_mode_map = {"Just resize": 0, "Crop and resize": 1, "Resize and fill": 2}
    resize_mode_int = resize_mode_map.get(img2img_resize_mode_val, 0)

    if is_img2img:
        if img2img_image_val is None:
            yield "img2img mode requires an input image. Please upload one.", []
            return
        # Convert PIL image to base64
        buf = _io.BytesIO()
        img2img_image_val.save(buf, format="PNG")
        img2img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    control_b64 = None
    if enable_controlnet_val and control_image_val is not None:
        buf = _io.BytesIO()
        control_image_val.save(buf, format="PNG")
        control_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    # Test connection
    success, msg = client.test_connection()
    if not success:
        yield f"Connection failed: {msg}", []
        return

    # Set model if specified
    if model_name:
        current_model = client.get_current_model()
        if current_model != model_name:
            yield f"Switching model to {model_name}...", []
            if not client.set_model(model_name):
                yield f"Failed to set model: {model_name}", []
                return
            yield f"Model switched to {model_name}", []

    # Load generation history for frequency data and skip-exists
    history = load_history()
    existing_hashes = {r["gen_hash"] for r in history if "gen_hash" in r}
    freq = build_frequency_map(history)

    # Get loop variants, ordered by frequency (least generated first)
    if state.loop_loras:
        loop_items = list(state.loop_loras.items())
        loop_items.sort(key=lambda x: freq.get(x[0], 0))
    else:
        loop_items = [(None, None)]

    if not loop_items and not state.always_loras and not state.random_loras:
        yield "No LoRAs configured. Add some to zones first.", []
        return

    # Total jobs = loop_variants * batch_count
    num_variants = len(loop_items)
    total_jobs = num_variants * int(batch_count)

    state.is_running = True
    state.should_stop = False
    state.total_jobs = total_jobs
    state.current_job = 0

    mode_str = "img2img" if is_img2img else "txt2img"
    log_lines = [f"Starting {mode_str} generation: {num_variants} variants x {int(batch_count)} batches x {int(batch_size)} per batch = {total_jobs * int(batch_size)} images..."]
    if freq:
        log_lines.append(f"History: {len(history)} previous generations tracked")
    if skip_exists:
        log_lines.append(f"Skip Exists: ON ({len(existing_hashes)} unique combos in history)")

    # Show loop order with frequency info
    if state.loop_loras:
        order_info = [f"{name}({freq.get(name, 0)})" for name, _ in loop_items]
        log_lines.append(f"Loop order (uses): {', '.join(order_info)}")

    generated_images = []
    skipped_count = 0

    # Initial yield to show we're starting
    yield "\n".join(log_lines), []

    job_num = 0
    for i, (loop_name, loop_entry) in enumerate(loop_items):
        for batch_idx in range(int(batch_count)):
            if state.should_stop:
                log_lines.append("Generation stopped by user.")
                yield "\n".join(log_lines), list(generated_images)
                break

            job_num += 1
            state.current_job = job_num
            progress(job_num / total_jobs, desc=f"Generating {job_num}/{total_jobs}")

            # Build prompt fresh each time (re-picks random LoRAs)
            lora_tags = []
            activation_texts = []
            used_lora_names = []

            # Always LoRAs
            for name, entry in state.always_loras.items():
                lora_tags.append(f"<lora:{name}:{entry.weight}>")
                used_lora_names.append(name)
                activation = get_activation_text(name) or entry.activation
                if activation:
                    activation_texts.append(activation)

            # Current loop LoRA
            if loop_name and loop_entry:
                lora_tags.append(f"<lora:{loop_name}:{loop_entry.weight}>")
                used_lora_names.append(loop_name)
                activation = get_activation_text(loop_name) or loop_entry.activation
                if activation:
                    activation_texts.append(activation)

            # Random LoRAs - frequency-weighted picks (less used = more likely)
            random_picks_info = []
            if state.random_loras and random_count > 0:
                random_pool = list(state.random_loras.items())
                pick_count = min(int(random_count), len(random_pool))

                if freq:
                    max_freq_val = max((freq.get(n, 0) for n, _ in random_pool), default=0) + 1
                    weights = [max_freq_val - freq.get(n, 0) + 1 for n, _ in random_pool]
                    picks = weighted_sample(random_pool, weights, pick_count)
                else:
                    picks = random.sample(random_pool, pick_count)

                for name, entry in picks:
                    lora_tags.append(f"<lora:{name}:{entry.weight}>")
                    used_lora_names.append(name)
                    activation = get_activation_text(name) or entry.activation
                    if activation:
                        activation_texts.append(activation)
                    random_picks_info.append(f"{name}({freq.get(name, 0)})")

            # Build final prompt
            prompt_parts = lora_tags.copy()
            if activation_texts:
                prompt_parts.extend(activation_texts)
            if positive_prompt:
                prompt_parts.append(positive_prompt)
            final_prompt = ", ".join(prompt_parts)

            # Compute generation hash for history
            gen_hash = compute_generation_hash(
                final_prompt, negative_prompt, steps, sampler, cfg,
                width, height, enable_hr, hr_scale, hr_upscaler,
                denoising, enable_adetailer, int(batch_size)
            )

            # Check skip exists
            if skip_exists and gen_hash in existing_hashes:
                variant_name = loop_name if loop_name else "base"
                log_lines.append(f"\n--- Variant '{variant_name}' Batch {batch_idx+1} SKIPPED (already generated) ---")
                skipped_count += 1
                yield "\n".join(log_lines), list(generated_images)
                continue

            # Create filename
            name_parts = []
            if loop_name:
                name_parts.append(loop_name)
            if state.always_loras:
                name_parts.append(f"always{len(state.always_loras)}")
            name_parts.append(f"b{batch_idx+1}")
            if not name_parts:
                name_parts.append("image")

            payload = build_payload(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                sampler_name=sampler,
                cfg_scale=cfg,
                width=width,
                height=height,
                enable_hr=enable_hr,
                hr_scale=hr_scale,
                hr_upscaler=hr_upscaler,
                denoising_strength=denoising,
                enable_adetailer=enable_adetailer,
                batch_size=int(batch_size),
                custom_filename="_".join(name_parts)
            )
            
            if enable_controlnet_val and control_b64:
                payload["alwayson_scripts"] = payload.get("alwayson_scripts", {})

                payload["alwayson_scripts"]["controlnet"] = {
                    "args": [
                        {
                            "enabled": True,
                            "image": control_b64,
                            "module": control_preprocessor_val,
                            "model": control_model_val,
                            "weight": control_weight_val,
                            "guidance_start": control_guidance_start_val,
                            "guidance_end": control_guidance_end_val,
                            "pixel_perfect": True,
                            "control_mode": control_mode_val,
                            "resize_mode": "Just Resize"
                        }
                    ]
                }

            # Add img2img-specific fields
            if is_img2img:
                payload["init_images"] = [img2img_b64]
                payload["denoising_strength"] = float(img2img_denoising_val)
                payload["resize_mode"] = resize_mode_int

            variant_name = loop_name if loop_name else "base"
            mode_label = "img2img" if is_img2img else "txt2img"
            log_lines.append(f"\n--- [{mode_label}] Variant '{variant_name}' Batch {batch_idx+1}/{int(batch_count)} ---")
            if is_img2img:
                log_lines.append(f"Denoising: {img2img_denoising_val}, Resize: {img2img_resize_mode_val}")
            if random_picks_info:
                log_lines.append(f"Random picks (uses): {', '.join(random_picks_info)}")
            log_lines.append(f"Prompt: {final_prompt[:100]}...")

            # Yield to show we're working on this one
            yield "\n".join(log_lines), list(generated_images)
            
            
            
            if is_img2img:
                success, msg, filepaths = client.generate_img2img(
                    payload,
                    output_dir=config.get("output_dir", "generated_images")
                )
            else:
                success, msg, filepaths = client.generate_image(
                    payload,
                    output_dir=config.get("output_dir", "generated_images")
                )

            log_lines.append(msg)
            if success and filepaths:
                generated_images.extend(filepaths)
                # Record to history
                append_history_record({
                    "timestamp": datetime.now().isoformat(),
                    "loras": sorted(used_lora_names),
                    "gen_hash": gen_hash,
                    "prompt_preview": final_prompt[:200],
                    "files": [os.path.basename(f) for f in filepaths]
                })

            # Yield after each generation to update gallery
            yield "\n".join(log_lines), list(generated_images)

            # Cooldown between generations (skip after last job)
            if cooldown_seconds > 0 and job_num < total_jobs and not state.should_stop:
                log_lines.append(f"Cooling down for {cooldown_seconds} seconds...")
                yield "\n".join(log_lines), list(generated_images)
                time.sleep(cooldown_seconds)

        if state.should_stop:
            break

    state.is_running = False
    summary = f"\n=== Complete: {len(generated_images)} images generated"
    if skipped_count:
        summary += f", {skipped_count} skipped"
    summary += " ==="
    log_lines.append(summary)

    yield "\n".join(log_lines), list(generated_images)


def stop_generation():
    """Signal generation to stop"""
    global state
    state.should_stop = True
    return "Stop signal sent..."


def test_api_connection(api_url: str) -> str:
    """Test connection to the API"""
    test_client = A1111Client(api_url)
    success, msg = test_client.test_connection()
    if success:
        models = test_client.get_models()
        return f"Connected! Found {len(models)} models."
    return f"Failed: {msg}"


def get_api_samplers(api_url: str) -> gr.update:
    """Fetch samplers from API"""
    test_client = A1111Client(api_url)
    samplers = test_client.get_samplers()
    if samplers:
        return gr.update(choices=samplers, value=samplers[0])
    return gr.update()


def get_api_models(api_url: str) -> gr.update:
    """Fetch models from API and return current model selected"""
    test_client = A1111Client(api_url)
    models = test_client.get_models()
    current = test_client.get_current_model()
    if models:
        return gr.update(choices=models, value=current if current in models else models[0])
    return gr.update()


def sort_lora_bank(sort_method: str) -> gr.update:
    """Sort the LoRA bank by the specified method"""
    global state

    if not state.bank_loras:
        return gr.update()

    if sort_method == "A-Z":
        sorted_loras = sorted(state.bank_loras, key=lambda x: x.lower())
    elif sort_method == "Z-A":
        sorted_loras = sorted(state.bank_loras, key=lambda x: x.lower(), reverse=True)
    else:
        sorted_loras = state.bank_loras

    state.bank_loras = sorted_loras
    return gr.update(choices=sorted_loras, value=[])


# --- Build Gradio Interface ---
def create_app():
    """Create the Gradio application"""
    global config

    defaults = config.get("defaults", DEFAULT_CONFIG["defaults"])

    # Custom CSS for multi-column LoRA grid
    custom_css = """
    .lora-bank-grid {
        max-height: 400px;
        overflow-y: auto;
    }
    .lora-bank-grid > div {
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 4px !important;
    }
    .lora-bank-grid label {
        font-size: 0.85em !important;
        padding: 4px 8px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sort-note {
        font-size: 0.8em;
        color: #888;
        align-self: center;
    }
    """

    with gr.Blocks(title="LoRA Combinator SD", css=custom_css) as app:

        gr.Markdown("# LoRA Combinator SD")
        gr.Markdown("Manage LoRA combinations for Stable Diffusion batch generation")

        # --- LoRA Bank Section ---
        with gr.Accordion("LoRA Bank", open=True):
            with gr.Row():
                lora_folder_input = gr.Textbox(
                    label="LoRA Folder Path (local or network: \\\\server\\share\\...)",
                    value=config.get("lora_folder", ""),
                    placeholder="e.g., C:/SD/models/Lora or \\\\192.168.0.238\\share\\Lora",
                    scale=3
                )
                scan_btn = gr.Button("Scan Folder", variant="primary", scale=1)
                fetch_api_btn = gr.Button("Fetch from API", variant="secondary", scale=1)
                scan_status = gr.Textbox(label="Status", interactive=False, scale=1)

            with gr.Row():
                lora_sort_dropdown = gr.Dropdown(
                    label="Sort by",
                    choices=["A-Z", "Z-A"],
                    value="A-Z",
                    scale=1
                )
                gr.Markdown("*Time-based sorting requires folder access*", elem_classes=["sort-note"])

            # Multi-column checkbox group with custom CSS
            lora_bank = gr.CheckboxGroup(
                label="Available LoRAs (select to add to zones)",
                choices=[],
                interactive=True,
                elem_classes=["lora-bank-grid"]
            )

            with gr.Row():
                default_weight = gr.Slider(
                    label="Default LoRA Weight",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    scale=1
                )
                add_always_btn = gr.Button("Add to Always Use", variant="secondary")
                add_loop_btn = gr.Button("Add to Loop Through", variant="secondary")
                add_random_btn = gr.Button("Add to Random Pool", variant="secondary")
                clear_all_btn = gr.Button("Clear All Zones", variant="stop")

        # --- Save/Load Config Section ---
        with gr.Accordion("Save/Load Zone Config", open=False):
            with gr.Row():
                config_name_input = gr.Textbox(
                    label="Config Name",
                    value=config.get("last_zone_config", ""),
                    placeholder="my_config",
                    scale=2
                )
                save_config_btn = gr.Button("Save", variant="primary", scale=1)
                load_config_btn = gr.Button("Load", variant="secondary", scale=1)
            with gr.Row():
                config_dropdown = gr.Dropdown(
                    label="Saved Configs",
                    choices=list_saved_configs(),
                    scale=2
                )
                refresh_configs_btn = gr.Button("Refresh List", scale=1)

        # --- Zones Section ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Always Use")
                gr.Markdown("*These LoRAs are included in every generation*")
                always_display = gr.Textbox(
                    value="(empty)",
                    interactive=False,
                    lines=4,
                    show_label=False
                )
                with gr.Row():
                    always_weight_dropdown = gr.Dropdown(
                        label="Adjust weight",
                        choices=[],
                        scale=2
                    )
                    always_weight_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        show_label=False,
                        scale=1
                    )
                with gr.Row():
                    always_remove_name = gr.Textbox(
                        placeholder="LoRA name to remove",
                        show_label=False,
                        scale=2
                    )
                    always_remove_btn = gr.Button("Remove", scale=1)
                clear_always_btn = gr.Button("Clear Always", variant="stop", size="sm")

            with gr.Column():
                gr.Markdown("### Loop Through")
                gr.Markdown("*Generate one image for each of these*")
                loop_display = gr.Textbox(
                    value="(empty)",
                    interactive=False,
                    lines=4,
                    show_label=False
                )
                with gr.Row():
                    loop_weight_dropdown = gr.Dropdown(
                        label="Adjust weight",
                        choices=[],
                        scale=2
                    )
                    loop_weight_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        show_label=False,
                        scale=1
                    )
                with gr.Row():
                    loop_remove_name = gr.Textbox(
                        placeholder="LoRA name to remove",
                        show_label=False,
                        scale=2
                    )
                    loop_remove_btn = gr.Button("Remove", scale=1)
                clear_loop_btn = gr.Button("Clear Loop", variant="stop", size="sm")

            with gr.Column():
                gr.Markdown("### Random Pool")
                gr.Markdown("*Pick N random from this pool per image*")
                random_display = gr.Textbox(
                    value="(empty)",
                    interactive=False,
                    lines=4,
                    show_label=False
                )
                random_count = gr.Slider(
                    label="Pick N random",
                    minimum=0,
                    maximum=5,
                    value=defaults.get("random_count", 1),
                    step=1
                )
                with gr.Row():
                    random_weight_dropdown = gr.Dropdown(
                        label="Adjust weight",
                        choices=[],
                        scale=2
                    )
                    random_weight_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        show_label=False,
                        scale=1
                    )
                with gr.Row():
                    random_remove_name = gr.Textbox(
                        placeholder="LoRA name to remove",
                        show_label=False,
                        scale=2
                    )
                    random_remove_btn = gr.Button("Remove", scale=1)
                clear_random_btn = gr.Button("Clear Random", variant="stop", size="sm")

        # --- Settings Section ---
        with gr.Accordion("Generation Settings", open=True):
            with gr.Row():
                gen_mode = gr.Radio(
                    label="Generation Mode",
                    choices=["txt2img", "img2img"],
                    value="txt2img",
                    scale=2
                )

            # img2img controls (hidden by default)
            with gr.Group(visible=False) as img2img_controls:
                with gr.Row():
                    img2img_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        scale=2
                    )
                    with gr.Column(scale=1):
                        img2img_denoising = gr.Slider(
                            label="img2img Denoising Strength",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            info="How much to change the input (0=keep original, 1=ignore input)"
                        )
                        img2img_resize_mode = gr.Dropdown(
                            label="Resize Mode",
                            choices=["Just resize", "Crop and resize", "Resize and fill"],
                            value="Just resize"
                        )
                        
            with gr.Group() as controlnet_controls:
                enable_controlnet = gr.Checkbox(label="Enable ControlNet", value=False)

                with gr.Row():
                    control_image = gr.Image(label="Control Image", type="pil")
                    control_preprocessor = gr.Dropdown(
                        label="Preprocessor",
                        choices=["none", "canny", "depth", "openpose", "tile", "lineart_anime"],
                        value="lineart_anime"
                    )
                    control_model = gr.Textbox(
                        label="ControlNet Model",
                        placeholder="controlnet-lineart-anime-sdxl-fp16"
                    )

                with gr.Row():
                    control_weight = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Weight")
                    control_guidance_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Start")
                    control_guidance_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="End")
                    control_mode = gr.Dropdown(
                        label="Control Mode",
                        choices=["Balanced", "My prompt is more important", "ControlNet is more important"],
                        value="Balanced"
                    )

            with gr.Row():
                api_url_input = gr.Textbox(
                    label="A1111 API URL",
                    value=config.get("api_url", DEFAULT_CONFIG["api_url"]),
                    scale=2
                )
                test_api_btn = gr.Button("Test Connection", scale=1)
                api_status = gr.Textbox(label="API Status", interactive=False, scale=1)

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Checkpoint Model",
                    choices=[defaults.get("model", "")] if defaults.get("model") else [],
                    value=defaults.get("model") or None,
                    allow_custom_value=True,
                    scale=3
                )
                refresh_models_btn = gr.Button("Refresh Models", scale=1)

            with gr.Row():
                sampler_input = gr.Dropdown(
                    label="Sampler",
                    choices=["Euler a", "Euler", "DPM++ 2M Karras", "DPM++ SDE Karras"],
                    value=defaults.get("sampler", "Euler a"),
                    allow_custom_value=True
                )
                steps_input = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=150,
                    value=defaults.get("steps", 27),
                    step=1
                )
                cfg_input = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=30,
                    value=defaults.get("cfg_scale", 6.0),
                    step=0.5
                )

            with gr.Row():
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=1,
                    maximum=8,
                    value=defaults.get("batch_size", 1),
                    step=1,
                    info="Images per batch (VRAM intensive)"
                )
                batch_count = gr.Slider(
                    label="Batch Count",
                    minimum=1,
                    maximum=10,
                    value=defaults.get("batch_count", 1),
                    step=1,
                    info="Number of batches per variant"
                )

            with gr.Row():
                width_input = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=2048,
                    value=defaults.get("width", 1024),
                    step=64
                )
                swap_dims_btn = gr.Button("⇄ Swap", scale=0, min_width=80)
                height_input = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=2048,
                    value=defaults.get("height", 1280),
                    step=64
                )

            with gr.Row():
                gr.Markdown("**Presets:**")
                preset_1_1 = gr.Button("1:1", scale=0, min_width=60)
                preset_4_3 = gr.Button("4:3", scale=0, min_width=60)
                preset_3_4 = gr.Button("3:4", scale=0, min_width=60)
                preset_5_4 = gr.Button("5:4", scale=0, min_width=60)
                preset_4_5 = gr.Button("4:5", scale=0, min_width=60)
                preset_16_9 = gr.Button("16:9", scale=0, min_width=60)
                preset_9_16 = gr.Button("9:16", scale=0, min_width=60)
                preset_3_2 = gr.Button("3:2", scale=0, min_width=60)
                preset_2_3 = gr.Button("2:3", scale=0, min_width=60)

            with gr.Row():
                enable_hr = gr.Checkbox(
                    label="Enable Hires Fix",
                    value=defaults.get("enable_hr", False)
                )
                hr_scale = gr.Slider(
                    label="HR Scale",
                    minimum=1.0,
                    maximum=4.0,
                    value=defaults.get("hr_scale", 1.5),
                    step=0.1
                )
                hr_upscaler = gr.Dropdown(
                    label="HR Upscaler",
                    choices=["Latent", "Latent (nearest)", "ESRGAN_4x", "R-ESRGAN 4x+"],
                    value=defaults.get("hr_upscaler", "Latent")
                )
                denoising = gr.Slider(
                    label="Denoising Strength",
                    minimum=0.0,
                    maximum=1.0,
                    value=defaults.get("denoising_strength", 0.5),
                    step=0.05
                )

            with gr.Row():
                enable_adetailer = gr.Checkbox(
                    label="Enable ADetailer",
                    value=defaults.get("enable_adetailer", False)
                )
                skip_exists_checkbox = gr.Checkbox(
                    label="Skip Exists (skip combos already in history)",
                    value=defaults.get("skip_exists", False)
                )

            positive_prompt = gr.Textbox(
                label="Base Positive Prompt",
                value=defaults.get("positive_prompt", ""),
                lines=2
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=defaults.get("negative_prompt", ""),
                lines=2
            )

            # Prompt config save/load
            with gr.Row():
                prompt_config_name = gr.Textbox(
                    label="Prompt Config Name",
                    value=config.get("last_prompt_config", ""),
                    placeholder="my_prompts",
                    scale=2
                )
                save_prompt_btn = gr.Button("Save Prompts", scale=1)
                load_prompt_btn = gr.Button("Load Prompts", scale=1)
            with gr.Row():
                prompt_config_dropdown = gr.Dropdown(
                    label="Saved Prompt Configs",
                    choices=list_prompt_configs(),
                    scale=2
                )
                refresh_prompt_configs_btn = gr.Button("Refresh", scale=1)

        # --- Preview Section ---
        with gr.Accordion("Generation Preview", open=True):
            preview_display = gr.Textbox(
                label="What will be generated",
                interactive=False,
                lines=8
            )
            refresh_preview_btn = gr.Button("Refresh Preview")

        # --- Generation Controls ---
        with gr.Row():
            cooldown_slider = gr.Slider(
                label="Cooldown (seconds between images)",
                minimum=0,
                maximum=120,
                value=defaults.get("cooldown", 0),
                step=1,
                scale=2
            )
            generate_btn = gr.Button("Generate Images", variant="primary", scale=2)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)

        # --- Output Section ---
        with gr.Row():
            with gr.Column(scale=1):
                log_output = gr.Textbox(
                    label="Generation Log",
                    interactive=False,
                    lines=12
                )
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    columns=3,
                    height="auto",
                    object_fit="contain",
                    allow_preview=True
                )

        # --- Wire up events ---

        # Scan folder
        scan_btn.click(
            fn=refresh_loras,
            inputs=[lora_folder_input],
            outputs=[scan_status, lora_bank]
        )

        # Fetch from API
        fetch_api_btn.click(
            fn=fetch_loras_from_api,
            inputs=[api_url_input],
            outputs=[scan_status, lora_bank]
        )

        # Sort LoRA bank
        lora_sort_dropdown.change(
            fn=sort_lora_bank,
            inputs=[lora_sort_dropdown],
            outputs=[lora_bank]
        )

        # Add to zones
        add_always_btn.click(
            fn=lambda sel, w, a, l, r: add_to_zone(sel, "always", w, a, l, r),
            inputs=[lora_bank, default_weight, always_display, loop_display, random_display],
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )
        add_loop_btn.click(
            fn=lambda sel, w, a, l, r: add_to_zone(sel, "loop", w, a, l, r),
            inputs=[lora_bank, default_weight, always_display, loop_display, random_display],
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )
        add_random_btn.click(
            fn=lambda sel, w, a, l, r: add_to_zone(sel, "random", w, a, l, r),
            inputs=[lora_bank, default_weight, always_display, loop_display, random_display],
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )

        # Clear zones
        clear_all_btn.click(
            fn=clear_all_zones,
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )
        zone_clear_outputs = [lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        clear_always_btn.click(
            fn=lambda: clear_zone("always"),
            outputs=zone_clear_outputs
        )
        clear_loop_btn.click(
            fn=lambda: clear_zone("loop"),
            outputs=zone_clear_outputs
        )
        clear_random_btn.click(
            fn=lambda: clear_zone("random"),
            outputs=zone_clear_outputs
        )

        # Save/Load LoRA configs
        save_config_btn.click(
            fn=save_zone_config,
            inputs=[config_name_input],
            outputs=[scan_status]
        )
        load_config_btn.click(
            fn=lambda name: load_zone_config(name if name else ""),
            inputs=[config_name_input],
            outputs=[always_display, loop_display, random_display, scan_status]
        )
        config_dropdown.change(
            fn=lambda name: name,
            inputs=[config_dropdown],
            outputs=[config_name_input]
        )

        # Save/Load Prompt configs
        save_prompt_btn.click(
            fn=save_prompt_config,
            inputs=[prompt_config_name, positive_prompt, negative_prompt],
            outputs=[scan_status]
        )
        load_prompt_btn.click(
            fn=lambda name: load_prompt_config(name if name else ""),
            inputs=[prompt_config_name],
            outputs=[positive_prompt, negative_prompt, scan_status]
        )
        prompt_config_dropdown.change(
            fn=lambda name: name,
            inputs=[prompt_config_dropdown],
            outputs=[prompt_config_name]
        )
        refresh_prompt_configs_btn.click(
            fn=lambda: gr.update(choices=list_prompt_configs()),
            outputs=[prompt_config_dropdown]
        )
        refresh_configs_btn.click(
            fn=lambda: gr.update(choices=list_saved_configs()),
            outputs=[config_dropdown]
        )

        # Remove from zones
        always_remove_btn.click(
            fn=lambda name: remove_from_zone("always", name),
            inputs=[always_remove_name],
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )
        loop_remove_btn.click(
            fn=lambda name: remove_from_zone("loop", name),
            inputs=[loop_remove_name],
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )
        random_remove_btn.click(
            fn=lambda name: remove_from_zone("random", name),
            inputs=[random_remove_name],
            outputs=[lora_bank, always_display, loop_display, random_display, always_weight_dropdown, loop_weight_dropdown, random_weight_dropdown, scan_status]
        )

        # Weight adjustment - when dropdown changes, load current weight
        always_weight_dropdown.change(
            fn=lambda name: get_lora_weight_in_zone("always", name),
            inputs=[always_weight_dropdown],
            outputs=[always_weight_slider]
        )
        loop_weight_dropdown.change(
            fn=lambda name: get_lora_weight_in_zone("loop", name),
            inputs=[loop_weight_dropdown],
            outputs=[loop_weight_slider]
        )
        random_weight_dropdown.change(
            fn=lambda name: get_lora_weight_in_zone("random", name),
            inputs=[random_weight_dropdown],
            outputs=[random_weight_slider]
        )

        # Weight adjustment - when slider changes, update weight
        always_weight_slider.change(
            fn=lambda name, w: update_zone_weight("always", name, w),
            inputs=[always_weight_dropdown, always_weight_slider],
            outputs=[always_display]
        )
        loop_weight_slider.change(
            fn=lambda name, w: update_zone_weight("loop", name, w),
            inputs=[loop_weight_dropdown, loop_weight_slider],
            outputs=[loop_display]
        )
        random_weight_slider.change(
            fn=lambda name, w: update_zone_weight("random", name, w),
            inputs=[random_weight_dropdown, random_weight_slider],
            outputs=[random_display]
        )

        # Dimension swap and presets
        swap_dims_btn.click(
            fn=lambda w, h: (h, w),
            inputs=[width_input, height_input],
            outputs=[width_input, height_input]
        )
        preset_1_1.click(fn=lambda: (1024, 1024), outputs=[width_input, height_input])
        preset_4_3.click(fn=lambda: (1152, 896), outputs=[width_input, height_input])
        preset_3_4.click(fn=lambda: (896, 1152), outputs=[width_input, height_input])
        preset_5_4.click(fn=lambda: (1280, 1024), outputs=[width_input, height_input])
        preset_4_5.click(fn=lambda: (1024, 1280), outputs=[width_input, height_input])
        preset_16_9.click(fn=lambda: (1344, 768), outputs=[width_input, height_input])
        preset_9_16.click(fn=lambda: (768, 1344), outputs=[width_input, height_input])
        preset_3_2.click(fn=lambda: (1216, 832), outputs=[width_input, height_input])
        preset_2_3.click(fn=lambda: (832, 1216), outputs=[width_input, height_input])

        # Mode toggle - show/hide img2img controls
        gen_mode.change(
            fn=lambda mode: gr.update(visible=(mode == "img2img")),
            inputs=[gen_mode],
            outputs=[img2img_controls]
        )

        # Test API
        test_api_btn.click(
            fn=test_api_connection,
            inputs=[api_url_input],
            outputs=[api_status]
        )

        # Load samplers from API
        test_api_btn.click(
            fn=get_api_samplers,
            inputs=[api_url_input],
            outputs=[sampler_input]
        )

        # Load models from API (on test connection and refresh)
        test_api_btn.click(
            fn=get_api_models,
            inputs=[api_url_input],
            outputs=[model_dropdown]
        )
        refresh_models_btn.click(
            fn=get_api_models,
            inputs=[api_url_input],
            outputs=[model_dropdown]
        )

        # Preview
        refresh_preview_btn.click(
            fn=build_prompt_preview,
            inputs=[positive_prompt, negative_prompt, random_count, batch_size, batch_count],
            outputs=[preview_display]
        )

        # Generation
        generate_btn.click(
            fn=run_generation,
            inputs=[
                api_url_input,
                model_dropdown,
                positive_prompt,
                negative_prompt,
                steps_input,
                sampler_input,
                cfg_input,
                width_input,
                height_input,
                enable_hr,
                hr_scale,
                hr_upscaler,
                denoising,
                enable_adetailer,
                random_count,
                batch_size,
                batch_count,
                cooldown_slider,
                skip_exists_checkbox,
                gen_mode,
                img2img_image,
                img2img_denoising,
                img2img_resize_mode,
                enable_controlnet,
                control_image,
                control_preprocessor,
                control_model,
                control_weight,
                control_guidance_start,
                control_guidance_end,
                control_mode
            ],
            outputs=[log_output, gallery]
        )

        # Stop
        stop_btn.click(
            fn=stop_generation,
            outputs=[log_output]
        )

    return app


# --- Main ---
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",  # Allow access from other machines
        server_port=7861,  # Different from SD WebUI
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
