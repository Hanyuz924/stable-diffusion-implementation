
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from models.VAE import Decoder
from models.VAE import Encoder
from models.diffusion import Diffusion
from safetensors.torch import load_file
import models.param_namecheck as pn
from tqdm import tqdm 
import json



def load_model(load_nuet = True, load_decoder = True, load_encoder =True, load_tokenizer = True):
    if load_nuet:
        unet_path = "../sd15_local/unet/diffusion_pytorch_model.safetensors"
        state_dict = load_file(unet_path)
        my_unet = Diffusion()
        pn.load_diffusion(my_unet, state_dict)
    else:
        my_unet = None

    if load_decoder or load_encoder : 
        official_vae = AutoencoderKL.from_pretrained(
                "../sd15_local", 
                subfolder="vae"
            )
    if load_decoder:
        vae_path = "../sd15_local/vae/diffusion_pytorch_model.safetensors"
        vae_config_path = "../sd15_local/vae/config.json"
        with open(vae_config_path, 'r', encoding='utf-8') as f:
                vae_config = json.load(f)
        state_dict = load_file(vae_path)
        decoder = Decoder(in_channels=vae_config["latent_channels"],
                            out_channels=vae_config["out_channels"],
                            down_block_channels=vae_config["block_out_channels"][::-1],
                            layer_per_block= vae_config["layers_per_block"])
        pn.load_decoder(decoder, state_dict)

        decoder_post_quant = torch.nn.Conv2d(4 ,4 , kernel_size=1)

        post_quant_weights = official_vae.post_quant_conv.state_dict()
        pn.load_post_quant_conv(decoder_post_quant, post_quant_weights)
    else:
        decoder = None
        decoder_post_quant = None
    
    if load_encoder:
        vae_path = "../sd15_local/vae/diffusion_pytorch_model.safetensors"
        vae_config_path = "../sd15_local/vae/config.json"
        with open(vae_config_path, 'r', encoding='utf-8') as f:
                vae_config = json.load(f)
        state_dict = load_file(vae_path)
        encoder = Encoder(in_channels=vae_config["in_channels"],
                          out_channels=vae_config["latent_channels"],
                          down_block_channels=vae_config["block_out_channels"],
                          layer_per_block= vae_config["layers_per_block"]
                        )
        pn.load_encoder(encoder, state_dict)

        encoder_post_quant_conv = torch.nn.Conv2d(8, 8, kernel_size = 1)
        pre_quant_conv_weights = official_vae.quant_conv.state_dict()
        pn.load_pre_quant_conv(encoder_post_quant_conv, pre_quant_conv_weights)
    else:
        encoder = None
    
    if load_tokenizer:
        tokenizer = CLIPTokenizer.from_pretrained(
            "../sd15_local", 
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "../sd15_local",
            subfolder="text_encoder"
        )
    else:
        tokenizer = None
        text_encoder = None

    return {
            "Decoder": decoder,
            "Unet":  my_unet,
            "post_quant_conv": decoder_post_quant,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "encoder":encoder,
            "encoder_post_quant_conv":encoder_post_quant_conv
        }

def DDIM():
    pass


def sample(text_prompt:str, guidance_scale:float, inference_step:int, sample_number:int = 1 ):
    pass


def main():
    device = "cuda"
    models = load_model(load_nuet = True, load_decoder = True, load_encoder =False, load_tokenizer = True)
    my_unet = models["Unet"].to(device)
    decoder = models["Decoder"].to(device)
    post_quant_conv = models["post_quant_conv"].to(device)
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"].to(device)
    
    scheduler = DDIMScheduler.from_pretrained(
        "../sd15_local", 
        subfolder="scheduler",
        local_files_only=True
    )
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    latents = torch.randn((1, 4, 64, 64), device=device)
    latents = latents * scheduler.init_noise_sigma
    prompt = "a photo of cute dog"

    guidance_scale = 7.0
    with torch.no_grad():
        text_input = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",device = device)
        cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in tqdm(scheduler.timesteps):
            timestep_val = t.item() if isinstance(t, torch.Tensor) else t
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep_val)
            t_tensor = torch.tensor([timestep_val], dtype=torch.float32, device=device)
            noise_pred = my_unet(
                latent_imgae=latent_model_input, 
                hidden_contex=text_embeddings, 
                time_step=t_tensor                     
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, timestep_val, latents).prev_sample

    with torch.no_grad():
        off_z = post_quant_conv(latents / 0.18215)
        image_tensor = decoder(off_z)

    image_tensor = image_tensor.clamp(-1, 1)
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_numpy = image_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
    image_pil = Image.fromarray((image_numpy * 255).round().astype(np.uint8))
    image_pil.save("final_cat.png")
    
if __name__ == "__main__":
    main()