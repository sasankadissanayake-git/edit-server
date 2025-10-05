from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import QwenImageEditPlusPipeline
import torch
import requests
import logging
from io import BytesIO
from typing import Optional
from PIL import Image
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pipeline: Optional[QwenImageEditPlusPipeline] = None

class ImageEditInput(BaseModel):
    image_url: str
    prompt: str

app = FastAPI()

# --- Model Loading on Startup ---
@app.on_event("startup")
async def load_model():
    """
    Load the Qwen model only once when the application starts.
    This is critical for performance and cloud deployment cost efficiency.
    """
    global pipeline
    model_id = "Qwen/Qwen-Image-Edit-2509"
    logging.info(f"Attempting to load model: {model_id}")

    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id, 
        )
        pipeline.set_progress_bar_config(disable=True)
        logging.info(f"Pipeline successfully loaded.")

    except Exception as e:
        logging.error(f"FATAL: Failed to load Qwen pipeline. Error: {e}")
        pipeline = None 




# --- API Endpoints ---

@app.get("/")
def read_root():
    """Simple health check endpoint."""
    status = "Ready" if pipeline else "Loading (or Failed)"
    return {"message": "Qwen Image Edit API", "model_status": status}


@app.post("/edit_image")
async def edit_image(input: ImageEditInput):

    if pipeline is None:
        logging.error("Model not available. Server startup failed or model is still loading.")
        raise HTTPException(status_code=503, detail="Image editing model is not yet loaded or failed to load. Please try again in a moment.")

    logging.info(f"Starting image edit for prompt: {input.prompt[:50]}...")

    try:
        image = Image.open(BytesIO(requests.get(input.image_url).content))
        
        inputs = {
            "image": image,
            "prompt": input.prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image: Image.Image = output.images[0]

        img_byte_arr = BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

        logging.info("Image generation complete. Returning base64 response.")

        return {"image": img_base64}

    except HTTPException:
        # Re-raise explicit HTTP exceptions (e.g., from download_image)
        raise
    except Exception as e:
        logging.error(f"Model inference failed: {e}", exc_info=True)
        # Raise a general internal server error for model-related failures
        raise HTTPException(status_code=500, detail=f"Image processing failed during model inference: {e}")

