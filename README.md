# combinatorSD

Standalone Gradio app for LoRA combinations through an existing Automatic1111 HTTP API. It manages Always, Loop and Random zones, prompt variants, txt2img/img2img batches, optional hires and ADetailer, and local generation history.

## Choose the right tool

| Tool | Runs against | Use it for |
| --- | --- | --- |
| This repository | A1111 HTTP API | A separate browser UI or remote A1111 server |
| [sd-combinator-ext](https://github.com/Graceus777/sd-combinator-ext) | A1111 internals | The LoRA Combinator tab inside WebUI |
| [comfyui_wrapper](https://github.com/Graceus777/comfyui_wrapper) | ComfyUI API | Workflow graphs, editable batch jobs and video |
| [sd-comic-ext](https://github.com/Graceus777/sd-comic-ext) | A1111 or ComfyUI | Scripted panels, pages, movies and project review |

## Install and launch

Use Python 3.10+ and start A1111 with `--api`. The app does not install or start A1111, models, ADetailer or ControlNet.

```powershell
git clone https://github.com/Graceus777/combinatorSD.git
cd combinatorSD
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python app.py
```

On Linux/macOS use `.venv/bin/python`. Open `http://127.0.0.1:7861`; A1111 normally uses port 7860. The app currently listens on all interfaces. Set the API URL in the UI to your A1111 server, then test the connection and refresh models/samplers.

## First batch

1. Fetch LoRAs from the API, or scan a folder readable by the machine running this app. A remote server's filesystem paths are not necessarily local paths.
2. Add shared style LoRAs to **Always**, character LoRAs to **Loop**, and optional variations to **Random**. Choose weights and random count.
3. Add activation text. Local `lora_texts/NAME.json` uses `{"activation text": "your trigger"}`; `NAME.txt` is also supported by the app. Use model names that A1111 recognizes.
4. Set the positive/negative prompt, dimensions, sampler, steps, CFG, seed and batch settings. Preview before generating.
5. Run a small batch and inspect images. Save your zone and prompt configurations for reuse.

`config.json` holds UI settings; `configs/` holds saved zones/prompts; `generation_history.jsonl` supports deduplication and frequency weighting. Images are written to the chosen output folder (default `generated_images/`). The API client also requests server-side saves. Keep these runtime files local.

## Optional pose-cache runner

`generate_accel.py` cycles character sidecars through an explicit pose recipe. The first successful image for a pose seeds later img2img generations; this changes the image-conditioning behavior and is not a promise of equal quality or faster execution.

Copy `recipe.example.json` to `recipe.json`, replace the placeholder with an installed pose LoRA and its trigger, and place character JSON sidecars in `lora_texts/`. Sidecars use `activation text` and optional `preferred weight`.

```powershell
python generate_accel.py --list
python generate_accel.py --recipe recipe.json -n 4 --batch 1 --api http://127.0.0.1:7860
python generate_accel.py --help
```

ADetailer is opt-in with `--adetailer`. Cache files live under the output directory and are separated by recipe hash. Change output directories for a different checkpoint or conditioning setup: the cache is an image reuse aid, not a full generation fingerprint. Generation requires your own model files; none are bundled.

## History and troubleshooting

```powershell
python gen_dashboard.py --top 20
python gen_dashboard.py --file generation_history.jsonl --since 2026-09-01
```

- Connection failure: confirm A1111 is running with `--api` and the configured URL reaches it.
- Missing LoRAs: refresh A1111's registry and check names and local sidecars.
- Missing ADetailer: install it in A1111 before enabling it here.
- Skipped jobs: inspect skip-exists settings and history; deleting history changes deduplication and frequency weighting.
- Timeout: inspect the server queue before retrying; an HTTP timeout does not prove the GPU job stopped.

For offline checks: `python -m pip install pytest` then `python -m pytest -q`. These checks do not perform GPU generation.
