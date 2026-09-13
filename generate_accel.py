"""
Accelerated SD character generator with img2img prompt-similarity cache.
Cycles through a supplied local pose recipe; cached poses seed later img2img calls.

Usage:
  python generate_accel.py --recipe recipe.json
  python generate_accel.py --recipe recipe.json -n 40
  python generate_accel.py --list
"""
import sys, os, json, time, random, re, argparse, base64, io, hashlib, math
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_client import A1111Client, build_payload

import requests
from PIL import Image

LORA_TEXTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_texts")
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_cache.json")

def load_recipe(path):
    """Read a local recipe without embedding private model/prompt presets."""
    with open(path, encoding="utf-8-sig") as handle:
        recipe = json.load(handle)
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a JSON object")
    poses = recipe.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("recipe requires a nonempty poses list")
    for pose in poses:
        if not isinstance(pose, dict) or not isinstance(pose.get("lora"), str) or not pose["lora"].strip():
            raise ValueError("each pose requires a LoRA name")
        try:
            weight = float(pose.get("weight", 0.8))
        except (TypeError, ValueError) as error:
            raise ValueError("pose weight must be a number") from error
        if not math.isfinite(weight):
            raise ValueError("pose weight must be finite")
        if not isinstance(pose.get("activation", ""), str):
            raise ValueError("pose activation must be a string")
    if not isinstance(recipe.get("skip_keywords", []), list) or not all(isinstance(k, str) for k in recipe.get("skip_keywords", [])):
        raise ValueError("skip_keywords must be a list")
    return recipe


class PoseCache:
    """Cache that stores one image per pose index for img2img reuse."""

    def __init__(self, path: str = CACHE_FILE):
        self.path = path
        self.entries: Dict[int, Dict] = {}  # pose_idx -> {filepath, width, height}
        self._load()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Convert string keys back to int
                self.entries = {int(k): v for k, v in data.get("entries", {}).items()}
            except Exception:
                self.entries = {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": self.entries}, f)

    def validate(self) -> int:
        """Remove entries with missing files."""
        before = len(self.entries)
        self.entries = {k: v for k, v in self.entries.items()
                        if os.path.isfile(v.get("filepath", ""))}
        removed = before - len(self.entries)
        if removed > 0:
            self._save()
        return removed

    def get(self, pose_idx: int, width: int, height: int) -> Optional[str]:
        """Get cached image path for pose if resolution matches."""
        entry = self.entries.get(pose_idx)
        if entry and entry.get("width") == width and entry.get("height") == height:
            fp = entry.get("filepath")
            if fp and os.path.isfile(fp):
                return fp
        return None

    def put(self, pose_idx: int, filepath: str, width: int, height: int):
        """Store an image for a pose index."""
        self.entries[pose_idx] = {
            "filepath": filepath,
            "width": width,
            "height": height,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()


# ═══════════════════════════════════════════════════════════════════════════
# img2img generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_img2img(client: A1111Client, payload: Dict, init_path: str,
                     denoise: float, output_dir: str) -> Tuple[bool, str, List[str]]:
    """img2img generation using a cached init image."""
    try:
        with open(init_path, "rb") as f:
            init_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return False, f"Failed to read init image: {e}", []

    api_payload = {k: v for k, v in payload.items()
                   if k not in ["custom_filename", "enable_hr", "hr_scale",
                                "hr_upscaler", "enable_adetailer"]}
    api_payload["init_images"] = [init_b64]
    api_payload["denoising_strength"] = denoise

    if payload.get("enable_adetailer"):
        api_payload["alwayson_scripts"] = {"ADetailer": {"args": [True]}}

    try:
        url = f"{client.base_url}/sdapi/v1/img2img"
        response = requests.post(url, json=api_payload, timeout=300)
        if response.status_code != 200:
            return False, f"API error: {response.status_code}", []

        result = response.json()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        saved = []
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = payload.get("custom_filename", "img2img")
        for idx, img_b64 in enumerate(result["images"]):
            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGBA")
            fn = f"{ts}_{name}_{idx+1}.png" if len(result["images"]) > 1 else f"{ts}_{name}.png"
            fp = os.path.abspath(os.path.join(output_dir, fn)).replace("\\", "/")
            image.save(fp)
            saved.append(fp)
        return True, f"Saved: {len(saved)} image(s)", saved

    except requests.exceptions.Timeout:
        return False, "Generation timed out", []
    except Exception as e:
        return False, f"Error: {e}", []


# ═══════════════════════════════════════════════════════════════════════════
# Character discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_characters(lora_texts_dir=LORA_TEXTS_DIR, skip_keywords=()):
    chars = []
    if not os.path.isdir(lora_texts_dir):
        return chars
    for fname in os.listdir(lora_texts_dir):
        if not fname.endswith(".json"):
            continue
        lora_name = fname[:-5]
        if any(kw.lower() in lora_name.lower() for kw in skip_keywords):
            continue
        try:
            with open(os.path.join(lora_texts_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        act = data.get("activation text", "").strip()
        if not act:
            continue
        weight = data.get("preferred weight", 0)
        if weight <= 0:
            weight = 0.8
        chars.append({"lora": lora_name, "activation": act, "weight": weight})
    return chars


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(args):
    recipe = load_recipe(args.recipe) if args.recipe else None
    if not args.list and recipe is None:
        raise ValueError("--recipe is required for generation; see recipe.example.json")
    chars = discover_characters(args.lora_texts, (recipe or {}).get("skip_keywords", []))

    if args.list:
        print(f"Discovered {len(chars)} character LoRAs:")
        for c in sorted(chars, key=lambda x: x["lora"]):
            print(f"  {c['lora']:55s} w={c['weight']}")
        return

    if not chars:
        print("No characters found in lora_texts/")
        return

    random.shuffle(chars)
    n = min(args.n, len(chars)) if args.n else len(chars)
    chars = chars[:n]

    client = A1111Client(args.api)
    ok_conn, msg_conn = client.test_connection()
    if not ok_conn:
        print(f"API connection failed: {msg_conn}")
        return

    poses = recipe["poses"]
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = os.path.join(args.output, f"pose_cache_{recipe_hash}.json")
    os.makedirs(args.output, exist_ok=True)
    cache = PoseCache(cache_path)
    removed = cache.validate()

    print(f"Generating {n} characters x {args.batch} = {n * args.batch} images", flush=True)
    print(f"Cycling through {len(poses)} pose LoRAs", flush=True)
    print(f"Resolution: {args.width}x{args.height}, denoise for accel: {args.denoise}", flush=True)
    print(f"Cache: {len(cache.entries)} poses cached ({removed} stale removed)", flush=True)
    print("", flush=True)

    accel_hits, accel_misses = 0, 0

    for i, char in enumerate(chars):
        pose_idx = i % len(poses)
        pose = poses[pose_idx]
        pose_name, pose_weight, pose_tags = pose["lora"], float(pose.get("weight", 0.8)), pose.get("activation", "")

        char_lora = f"<lora:{char['lora']}:{char['weight']}>"
        pose_lora = f"<lora:{pose_name}:{pose_weight}>"
        prompt = f"{char_lora}, {pose_lora}, {char['activation']}, {pose_tags}, {recipe.get('positive_prompt', '')}"
        tag = char["lora"].replace(" ", "_")[:25]

        payload = build_payload(
            prompt=prompt, negative_prompt=recipe.get('negative_prompt', ''),
            steps=args.steps, sampler_name=args.sampler, cfg_scale=args.cfg,
            width=args.width, height=args.height,
            enable_hr=False, enable_adetailer=args.adetailer,
            seed=-1, batch_size=args.batch,
            custom_filename=f"{tag}_p{pose_idx+1}",
        )

        # Check cache for this pose
        init_path = cache.get(pose_idx, args.width, args.height)

        print(f"[{i+1}/{n}] {char['lora']} + pose{pose_idx+1}", flush=True)

        if init_path:
            accel_hits += 1
            print(f"  ACCEL: denoise={args.denoise}, init={os.path.basename(init_path)}", flush=True)
            ok, msg, files = generate_img2img(client, payload, init_path, args.denoise, args.output)
        else:
            accel_misses += 1
            print(f"  txt2img (seeding cache)", flush=True)
            ok, msg, files = client.generate_image(payload, output_dir=args.output)

        status = "OK" if ok else "FAIL"
        print(f"  {status} - {msg}", flush=True)

        # Cache first successful image for this pose
        if ok and files and not cache.get(pose_idx, args.width, args.height):
            cache.put(pose_idx, files[0], args.width, args.height)

        if i < n - 1:
            time.sleep(args.cooldown)

    print("", flush=True)
    print(f"DONE - accel hits: {accel_hits}, misses: {accel_misses}", flush=True)
    hit_rate = accel_hits / (accel_hits + accel_misses) * 100 if (accel_hits + accel_misses) > 0 else 0
    print(f"Hit rate: {hit_rate:.1f}%", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=None, help="Number of characters (default: all)")
    p.add_argument("--batch", "-b", type=int, default=2)
    p.add_argument("--width", "-W", type=int, default=1024)
    p.add_argument("--height", "-H", type=int, default=1280)
    p.add_argument("--steps", type=int, default=27)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--sampler", default="Euler a")
    p.add_argument("--adetailer", action="store_true", default=False)
    p.add_argument("--denoise", type=float, default=0.8, help="Denoising strength for img2img accel (pose-only, needs high denoise)")
    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--api", default="http://127.0.0.1:7860")
    p.add_argument("--output", "-o", default="generated_images")
    p.add_argument("--list", action="store_true", help="List discovered characters")
    p.add_argument("--recipe", help="Local recipe JSON; required unless --list")
    p.add_argument("--lora-texts", default=LORA_TEXTS_DIR, help="Character sidecar directory")
    try:
        run(p.parse_args())
    except (ValueError, OSError) as error:
        p.error(str(error))
