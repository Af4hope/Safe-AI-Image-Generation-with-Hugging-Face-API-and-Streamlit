from huggingface_hub import InferenceClient

import config

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
FILTER_API_URL = "https://filters-zeta.vercel.app/api/filter"

ENHANCE_SYS = (
    "Improve prompts for text-to-image. Return ONLY the enhanced prompt."

    "Add subject, style, lighting, camera angle, background,colors. Keep it safe."

)


# This is only for image quality guidance, not safety filtering
NEGATIVE = "low quality, blurry, distorted, watermark, text,cropped"

img_client = InferenceClient(provider="hf-inference",
api_key=config.HF_API_KEY)

def check_prompt_with_filter_api(prompt: str):
    try:
        response = requests.post(
            FILTER_API_URL,
            json={"prompt": prompt},
            timeout=10,
        )

        response.raise_for_status()
        
        data = response.json()

        if not isinstance(data, dict):
            return {"ok": False, "reason": "Invalid filter API response. "}

        return data