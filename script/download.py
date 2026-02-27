from diffusers import StableDiffusionPipeline
import torch
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.save_pretrained("./sd15_local")

