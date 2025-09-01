import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import requests
import base64
import io
import os
import time
import itertools
from PIL import Image
from collections import Counter
import threading
from datetime import datetime
import copy
import random
import re

# --- Configuration ---
TEMPLATES_FILE = 'templates.json'
OUTPUT_DIR = 'generated_images'
PAYLOAD_AUDIT_FILE = 'payload_audit.txt'
DEFAULT_API_URL = "http://127.0.0.1:7860"

# --- Utility ---
def safe_filename(s: str) -> str:
    s = re.sub(r'[^\w\-_\. ]', '_', s)
    s = re.sub(r'\s+', '-', s)
    return s[:200]

# --- Core Logic Classes ---
class PromptGenerator:
    def __init__(self, templates_file):
        self.templates = self.load_templates(templates_file)
        self.lora_stats = Counter()

    def load_templates(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("templates.json must contain a JSON object/dictionary at top level.")
                return data
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Error", f"Could not load or parse {filepath}:\n{e}")
            return None

    def generate_payload_queue(self, selections, base_payload, skip_interval=1):
        """
        selections: dict mapping category -> list of selected items (or empty list for "All")
        """
        if not self.templates:
            return [], Counter()
        self.lora_stats.clear()
        # Build lists per category for Cartesian product
        category_lists = []
        for category, selected_items in selections.items():
            items = self.templates.get(category, [])
            if selected_items and "All" not in selected_items:
                # Only keep items matching selected names
                items = [item for item in items if item.get('name') in selected_items]
            if not items:
                # Fallback to at least one empty item to prevent product() from failing
                items = [{}]
            category_lists.append(items)

        payload_queue = []
        all_combos = list(itertools.product(*category_lists))
        for idx, combo in enumerate(all_combos):
            if skip_interval > 1 and (idx % skip_interval) != 0:
                continue
            prompt_parts = []
            lora_tags = []
            for item in combo:
                if not item:
                    continue
                if item.get('prompt'):
                    prompt_parts.append(item['prompt'])
                lora = item.get('lora')
                if lora and lora.get('name'):
                    tag = f"<lora:{lora['name']}:{lora.get('weight', 1.0)}>"
                    lora_tags.append(tag)
                    self.lora_stats[lora['name']] += 1
            final_prompt = ", ".join(filter(None, lora_tags + prompt_parts))
            payload = copy.deepcopy(base_payload)
            payload['prompt'] = final_prompt
            # descriptive filename
            name_parts = [item.get('name', 'item') for item in combo if item.get('name')]
            if not name_parts:
                name_parts = ['image']
            payload['custom_filename'] = safe_filename("_".join(name_parts))
            payload_queue.append(payload)
        return payload_queue, self.lora_stats.copy()

class A1111Client:
    def __init__(self, api_url):
        self.api_url = f"{api_url.rstrip('/')}/sdapi/v1/txt2img"

    def generate_image(self, payload, output_dir=OUTPUT_DIR):
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            r = response.json()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_data = base64.b64decode(r['images'][0])
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{timestamp}_{payload.get('custom_filename', 'image')}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            return True, f"Image saved to {filepath}"
        except requests.exceptions.RequestException as e:
            return False, f"API Error: {e}"
        except Exception as e:
            return False, f"Error processing image: {e}"

# --- GUI ---
class App(tk.Tk):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.title("A1111 Prompt Sequence Generator")
        self.geometry("950x820")
        self.payload_queue = []
        self.is_running = False
        self.stop_event = threading.Event()
        self.selection_vars = {}  # category -> ttk.Combobox variable
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        # API & parameters
        params_frame = ttk.LabelFrame(main_frame, text="API & Generation Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        params_frame.columnconfigure(1, weight=1)
        ttk.Label(params_frame, text="A1111 API URL:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.api_url_var = tk.StringVar(value=DEFAULT_API_URL)
        ttk.Entry(params_frame, textvariable=self.api_url_var).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, text="Base Positive Prompt:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.pos_prompt_var = tk.StringVar(value="best quality, masterpiece")
        ttk.Entry(params_frame, textvariable=self.pos_prompt_var).grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, text="Negative Prompt:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.neg_prompt_var = tk.StringVar(value="low quality, worst quality, blurry, deformed")
        ttk.Entry(params_frame, textvariable=self.neg_prompt_var).grid(row=2, column=1, sticky=tk.EW, padx=5)
        # Other parameters
        other_params_frame = ttk.Frame(params_frame)
        other_params_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(other_params_frame, text="Steps:").pack(side=tk.LEFT, padx=5)
        self.steps_var = tk.StringVar(value="25")
        ttk.Entry(other_params_frame, textvariable=self.steps_var, width=5).pack(side=tk.LEFT)
        ttk.Label(other_params_frame, text="Sampler:").pack(side=tk.LEFT, padx=5)
        self.sampler_var = tk.StringVar(value="DPM++ 2M Karras")
        ttk.Entry(other_params_frame, textvariable=self.sampler_var, width=20).pack(side=tk.LEFT)
        ttk.Label(other_params_frame, text="CFG:").pack(side=tk.LEFT, padx=5)
        self.cfg_var = tk.StringVar(value="7.0")
        ttk.Entry(other_params_frame, textvariable=self.cfg_var, width=5).pack(side=tk.LEFT)
        
        ttk.Label(other_params_frame, text="Width:").pack(side=tk.LEFT, padx=5)
        self.width_var = tk.StringVar(value="512")
        ttk.Entry(other_params_frame, textvariable=self.width_var, width=5).pack(side=tk.LEFT)

        ttk.Label(other_params_frame, text="Height:").pack(side=tk.LEFT, padx=5)
        self.height_var = tk.StringVar(value="512")
        ttk.Entry(other_params_frame, textvariable=self.height_var, width=5).pack(side=tk.LEFT)

        self.enable_hr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(other_params_frame, text="Enable HR", variable=self.enable_hr_var).pack(side=tk.LEFT, padx=5)

        # Prompt Components frame
        component_frame = ttk.LabelFrame(main_frame, text="Prompt Components / LoRA Selection", padding="10")
        component_frame.pack(fill=tk.X, pady=5)
        for i, (category, items) in enumerate(self.generator.templates.items()):
            ttk.Label(component_frame, text=category.title()+":").grid(row=0, column=i, sticky=tk.W, padx=5)
            var = tk.StringVar(value="All")
            cb = ttk.Combobox(component_frame, textvariable=var, state="readonly", width=20)
            options = ["All"] + [item.get('name','') for item in items]
            cb['values'] = options
            cb.current(0)
            cb.grid(row=1, column=i, padx=5, pady=2)
            self.selection_vars[category] = var
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)
        skip_frame = ttk.Frame(controls_frame)
        skip_frame.pack(fill=tk.X, pady=2)
        ttk.Label(skip_frame, text="Skip interval (1 = none):").pack(side=tk.LEFT, padx=5)
        self.skip_var = tk.IntVar(value=1)
        ttk.Spinbox(skip_frame, from_=1, to=50, textvariable=self.skip_var, width=5).pack(side=tk.LEFT)
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.X)
        self.btn_gen_payloads = ttk.Button(action_frame, text="Generate & Audit Payloads", command=self.generate_and_audit)
        self.btn_gen_payloads.pack(side=tk.LEFT, padx=5)
        self.btn_preview = ttk.Button(action_frame, text="Preview Random Prompt", command=self.preview_random_prompt)
        self.btn_preview.pack(side=tk.LEFT, padx=5)
        self.btn_run = ttk.Button(action_frame, text="RUN GENERATION", command=self.start_generation)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(action_frame, text="STOP", command=self.stop_generation, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=18)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def get_base_payload(self):
        try:
            return {
                "prompt": self.pos_prompt_var.get(),
                "negative_prompt": self.neg_prompt_var.get(),
                "steps": int(self.steps_var.get()),
                "sampler_name": self.sampler_var.get(),
                "cfg_scale": float(self.cfg_var.get()),
                "width": int(self.width_var.get()),
                "height": int(self.height_var.get()),
                "enable_hr": self.enable_hr_var.get()  # new boolean
            }
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return {}
    def get_current_selections(self):
        selections = {}
        for cat, var in self.selection_vars.items():
            val = var.get()
            if val == "All":
                selections[cat] = []
            else:
                selections[cat] = [val]
        return selections

    def generate_and_audit(self):
        base_payload = self.get_base_payload()
        selections = self.get_current_selections()
        skip_interval = self.skip_var.get()
        self.payload_queue, lora_stats = self.generator.generate_payload_queue(selections, base_payload, skip_interval)
        if not self.payload_queue:
            self.log("No payloads generated. Check your selections.")
            return
        self.log(f"Generated {len(self.payload_queue)} payloads.")
        stats_str = "\n--- LoRA Usage Statistics ---\n"
        if lora_stats:
            for name, count in lora_stats.items():
                stats_str += f"{name}: {count} uses\n"
        else:
            stats_str += "No LoRAs used in this batch.\n"
        stats_str += "-----------------------------"
        self.log(stats_str)
        # Write to audit
        try:
            with open(PAYLOAD_AUDIT_FILE, 'w', encoding='utf-8') as f:
                f.write(f"--- Generated {len(self.payload_queue)} Payloads ---\n\n")
                f.write(stats_str + "\n\n--- Full Payloads ---\n\n")
                for i, payload in enumerate(self.payload_queue):
                    f.write(f"--- Payload {i+1} ---\n")
                    f.write(json.dumps(payload, indent=2))
                    f.write("\n\n")
            self.log(f"Full payload list saved to {PAYLOAD_AUDIT_FILE}")
        except Exception as e:
            self.log(f"Error writing audit file: {e}")

    def preview_random_prompt(self):
        if not self.payload_queue:
            self.generate_and_audit()
        if not self.payload_queue:
            return
        payload = random.choice(self.payload_queue)
        self.log(f"\n--- Random Preview ---\nPrompt: {payload['prompt']}\nFilename: {payload['custom_filename']}\n--------------------")

    def start_generation(self):
        self.generate_and_audit()
        if not self.payload_queue:
            self.log("Generation cancelled: No payloads in queue.")
            return
        self.is_running = True
        self.stop_event.clear()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_gen_payloads.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.log("\n--- STARTING GENERATION ---")
        self.generation_thread = threading.Thread(target=self.run_generation_loop, daemon=True)
        self.generation_thread.start()

    def stop_generation(self):
        if self.is_running:
            self.stop_event.set()
            self.log("--- STOP SIGNAL SENT ---")
            self.log("Waiting for current job to finish...")
            self.btn_stop.config(state=tk.DISABLED)

    def run_generation_loop(self):
        client = A1111Client(self.api_url_var.get())
        cooldown = int(self.skip_var.get())  # reuse skip_var for simplicity
        total_jobs = len(self.payload_queue)
        for i, payload in enumerate(self.payload_queue):
            if self.stop_event.is_set():
                self.log("Generation stopped by user.")
                break
            self.log(f"--- Job {i+1}/{total_jobs} ---")
            self.log(f"Prompt: {payload['prompt']}")
            success, message = client.generate_image(payload)
            self.log(f"Status: {message}")
            if i < total_jobs - 1:
                self.log(f"Cooling down for {cooldown} seconds...")
                time.sleep(cooldown)
        self.generation_finished()

    def generation_finished(self):
        self.log("\n--- GENERATION FINISHED ---")
        self.is_running = False
        self.btn_run.config(state=tk.NORMAL)
        self.btn_gen_payloads.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.stop_event.clear()

if __name__ == "__main__":
    if not os.path.exists(TEMPLATES_FILE):
        messagebox.showerror("Fatal Error", f"'{TEMPLATES_FILE}' not found. Please create it next to the script.")
    else:
        prompt_generator = PromptGenerator(TEMPLATES_FILE)
        app = App(prompt_generator)
        app.mainloop()
