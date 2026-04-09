"""
A1111 API Client - Handles communication with Stable Diffusion WebUI API
"""
import requests
import base64
import io
import os
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image


class A1111Client:
    """Client for Automatic1111 Stable Diffusion WebUI API"""

    def __init__(self, api_url: str = "http://127.0.0.1:7860"):
        self.base_url = api_url.rstrip('/')
        self.txt2img_url = f"{self.base_url}/sdapi/v1/txt2img"
        self.img2img_url = f"{self.base_url}/sdapi/v1/img2img"
        self.models_url = f"{self.base_url}/sdapi/v1/sd-models"
        self.samplers_url = f"{self.base_url}/sdapi/v1/samplers"
        self.options_url = f"{self.base_url}/sdapi/v1/options"
        self.progress_url = f"{self.base_url}/sdapi/v1/progress"

    def test_connection(self) -> Tuple[bool, str]:
        """Test if the API is reachable"""
        try:
            response = requests.get(self.options_url, timeout=5)
            if response.status_code == 200:
                return True, "Connected successfully"
            return False, f"API returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to API. Is SD WebUI running?"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def get_models(self) -> List[str]:
        """Get list of available checkpoint models"""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                models = response.json()
                return [m.get("model_name", m.get("title", "")) for m in models]
            return []
        except Exception:
            return []

    def get_samplers(self) -> List[str]:
        """Get list of available samplers"""
        try:
            response = requests.get(self.samplers_url, timeout=10)
            if response.status_code == 200:
                samplers = response.json()
                return [s.get("name", "") for s in samplers]
            return []
        except Exception:
            return ["Euler a", "Euler", "DPM++ 2M Karras", "DPM++ SDE Karras"]

    def get_current_model(self) -> str:
        """Get currently loaded model"""
        try:
            response = requests.get(self.options_url, timeout=10)
            if response.status_code == 200:
                options = response.json()
                return options.get("sd_model_checkpoint", "")
            return ""
        except Exception:
            return ""

    def get_loras(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of available LoRAs from SD WebUI API.
        Returns list of dicts with 'name', 'alias', 'path', and possibly 'metadata' keys.
        """
        try:
            response = requests.get(
                f"{self.base_url}/sdapi/v1/loras",
                timeout=30
            )
            if response.status_code == 200:
                loras = response.json()
                # Debug: print first lora to see structure
                if loras:
                    print(f"[DEBUG] First LoRA from API: {loras[0]}")
                return loras
            return []
        except Exception as e:
            print(f"[DEBUG] Error fetching loras: {e}")
            return []

    def refresh_loras(self) -> bool:
        """Tell SD WebUI to refresh its LoRA list"""
        try:
            response = requests.post(
                f"{self.base_url}/sdapi/v1/refresh-loras",
                timeout=30
            )
            return response.status_code == 200
        except Exception:
            return False

    def set_model(self, model_name: str) -> bool:
        """Set the active model"""
        try:
            payload = {"sd_model_checkpoint": model_name}
            response = requests.post(self.options_url, json=payload, timeout=120)
            return response.status_code == 200
        except Exception:
            return False

    def get_progress(self) -> Dict[str, Any]:
        """Get current generation progress"""
        try:
            response = requests.get(self.progress_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"progress": 0, "eta_relative": 0}
        except Exception:
            return {"progress": 0, "eta_relative": 0}

    def generate_image(
        self,
        payload: Dict[str, Any],
        output_dir: str = "generated_images"
    ) -> Tuple[bool, str, List[str]]:
        """
        Generate image(s) using txt2img.

        Args:
            payload: The generation payload
            output_dir: Directory to save generated images

        Returns:
            Tuple of (success, message, list of filepaths)
        """
        try:
            # Remove custom fields that aren't part of the API
            api_payload = {k: v for k, v in payload.items()
                         if k not in ["custom_filename", "enable_adetailer"]}

            # Handle ADetailer if enabled
            if payload.get("enable_adetailer"):
                api_payload.setdefault("alwayson_scripts", {})["ADetailer"] = {
                    "args": [
                        True,   # ad_enable
                        False,  # skip_img2img (not used in txt2img)
                        {
                            "ad_model": "face_yolov8n.pt",
                            "ad_mask_k_largest": 1,
                        }
                    ]
                }

            # Save images on the server too (with full metadata for debugging)
            api_payload["save_images"] = True

            # Use longer timeout when hires fix is enabled (upscale + ADetailer)
            timeout = 600 if api_payload.get("enable_hr") else 300

            response = requests.post(
                self.txt2img_url,
                json=api_payload,
                timeout=timeout
            )

            if response.status_code != 200:
                return False, f"API error: {response.status_code}", []

            result = response.json()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # A1111 returns: [grid, img1, img2, ..., adetailer_extras...]
            # when batch_size > 1.  With batch_size == 1 there is no grid.
            # Skip the grid and ADetailer extras — keep only real images.
            batch_size = api_payload.get("batch_size", 1)
            all_images = result['images']
            if batch_size > 1:
                # first image is the grid — skip it, take next batch_size
                images_to_save = all_images[1:batch_size + 1]
            else:
                images_to_save = all_images[:1]

            saved_files = []
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            custom_name = payload.get('custom_filename', 'image')

            for idx, img_b64 in enumerate(images_to_save):
                try:
                    image_data = base64.b64decode(img_b64)

                    if len(images_to_save) > 1:
                        filename = f"{timestamp}_{custom_name}_{idx+1}.png"
                    else:
                        filename = f"{timestamp}_{custom_name}.png"

                    filepath = os.path.abspath(os.path.join(output_dir, filename))
                    filepath = filepath.replace("\\", "/")  # Normalize for Gradio
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    saved_files.append(filepath)
                except Exception as e:
                    print(f"[WARNING] Failed to save image {idx}: {e}")
                    continue

            if not saved_files:
                return False, "No images could be saved from response", []

            return True, f"Saved: {len(saved_files)} image(s)", saved_files

        except requests.exceptions.Timeout:
            return False, "Generation timed out", []
        except requests.exceptions.ConnectionError:
            return False, "Lost connection to API", []
        except KeyError:
            return False, "Invalid API response (no images)", []
        except Exception as e:
            return False, f"Error: {str(e)}", []

    def generate_img2img(
        self,
        payload: Dict[str, Any],
        output_dir: str = "generated_images"
    ) -> Tuple[bool, str, List[str]]:
        """
        Generate image(s) using img2img.

        Args:
            payload: The generation payload (must include 'init_images')
            output_dir: Directory to save generated images

        Returns:
            Tuple of (success, message, list of filepaths)
        """
        try:
            # Remove custom fields that aren't part of the API
            api_payload = {k: v for k, v in payload.items()
                         if k not in ["custom_filename", "enable_adetailer"]}

            # Handle ADetailer if enabled
            if payload.get("enable_adetailer"):
                api_payload.setdefault("alwayson_scripts", {})["ADetailer"] = {
                    "args": [
                        True,   # ad_enable
                        False,  # skip_img2img
                        {
                            "ad_model": "face_yolov8n.pt",
                            "ad_mask_k_largest": 1,
                        }
                    ]
                }

            # Save images on the server too (with full metadata for debugging)
            api_payload["save_images"] = True

            timeout = 600 if api_payload.get("enable_hr") else 300

            response = requests.post(
                self.img2img_url,
                json=api_payload,
                timeout=timeout
            )

            if response.status_code != 200:
                return False, f"API error: {response.status_code}", []

            result = response.json()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            batch_size = api_payload.get("batch_size", 1)
            all_images = result['images']
            if batch_size > 1:
                images_to_save = all_images[1:batch_size + 1]
            else:
                images_to_save = all_images[:1]

            saved_files = []
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            custom_name = payload.get('custom_filename', 'img2img')

            for idx, img_b64 in enumerate(images_to_save):
                try:
                    image_data = base64.b64decode(img_b64)

                    if len(images_to_save) > 1:
                        filename = f"{timestamp}_{custom_name}_{idx+1}.png"
                    else:
                        filename = f"{timestamp}_{custom_name}.png"

                    filepath = os.path.abspath(os.path.join(output_dir, filename))
                    filepath = filepath.replace("\\", "/")
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    saved_files.append(filepath)
                except Exception as e:
                    print(f"[WARNING] Failed to save image {idx}: {e}")
                    continue

            if not saved_files:
                return False, "No images could be saved from response", []

            return True, f"Saved: {len(saved_files)} image(s)", saved_files

        except requests.exceptions.Timeout:
            return False, "Generation timed out", []
        except requests.exceptions.ConnectionError:
            return False, "Lost connection to API", []
        except KeyError:
            return False, "Invalid API response (no images)", []
        except Exception as e:
            return False, f"Error: {str(e)}", []

    def interrogate(self, image_path: str, model: str = "clip") -> Optional[str]:
        """Get CLIP or DeepDanbooru tags for an image.

        Returns the caption string, or None on failure.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = requests.post(
                f"{self.base_url}/sdapi/v1/interrogate",
                json={"image": f"data:image/png;base64,{b64}", "model": model},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("caption", "")
            return None
        except Exception:
            return None

    def interrupt(self) -> bool:
        """Interrupt current generation"""
        try:
            response = requests.post(
                f"{self.base_url}/sdapi/v1/interrupt",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


def build_payload(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 27,
    sampler_name: str = "Euler a",
    cfg_scale: float = 6.0,
    width: int = 512,
    height: int = 768,
    enable_hr: bool = False,
    hr_scale: float = 1.5,
    hr_upscaler: str = "Latent",
    denoising_strength: float = 0.5,
    enable_adetailer: bool = False,
    seed: int = -1,
    batch_size: int = 1,
    custom_filename: str = "image"
) -> Dict[str, Any]:
    """Build a generation payload with all parameters"""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler_name,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "batch_size": batch_size,
        "custom_filename": custom_filename
    }

    if enable_hr:
        payload.update({
            "enable_hr": True,
            "hr_scale": hr_scale,
            "hr_upscaler": hr_upscaler,
            "denoising_strength": denoising_strength
        })

    if enable_adetailer:
        payload["enable_adetailer"] = True

    return payload


if __name__ == "__main__":
    # Test connection
    client = A1111Client()
    success, msg = client.test_connection()
    print(f"Connection test: {msg}")

    if success:
        print(f"Models: {client.get_models()[:5]}...")  # First 5
        print(f"Samplers: {client.get_samplers()}")
        print(f"Current model: {client.get_current_model()}")
