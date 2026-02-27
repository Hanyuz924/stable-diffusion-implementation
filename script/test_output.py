import json
import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
import torch.nn.functional as F
from models.VAE import Decoder
from models.VAE import Encoder
from models.diffusion import Diffusion
from safetensors.torch import load_file
import models.param_namecheck as pn

device = "cuda"
def test_vae_error():
    official_vae = AutoencoderKL.from_pretrained(
        "../sd15_local", 
        subfolder="vae"
    ).to(device)
    official_vae.eval()
    vae_path = "../sd15_local/vae/diffusion_pytorch_model.safetensors"
    vae_config_path = "../sd15_local/vae/config.json"
    with open(vae_config_path, 'r', encoding='utf-8') as f:
        vae_config = json.load(f)
    state_dict = load_file(vae_path)
    encoder = Encoder(in_channels = vae_config["in_channels"],
                    out_channels = vae_config["latent_channels"],
                    down_block_channels= vae_config["block_out_channels"],
                    layer_per_block= vae_config["layers_per_block"]
                    ).to(device)
    pn.load_encoder(encoder, state_dict)

    encoder.eval()

    dummy_image = torch.randn(1, 3, 512, 512, device=official_vae.device)
    with torch.no_grad():
        official_enc_out = official_vae.encoder(dummy_image)
        my_enc_out = encoder(dummy_image)

    mse_diff = F.mse_loss(official_enc_out, my_enc_out).item()

    print(f"Official Encoder :  {list(official_enc_out.shape)}")
    print(f"My Encoder :        {list(my_enc_out.shape)}")
    print(f"Encoder MSE:        {mse_diff:.8f}")
    decoder = Decoder(in_channels=vae_config["latent_channels"],
                    out_channels=vae_config["out_channels"],
                    down_block_channels=vae_config["block_out_channels"][::-1],
                    layer_per_block= vae_config["layers_per_block"]).to(device)
    pn.load_decoder(decoder, state_dict)
    decoder.eval()

    with torch.no_grad():
        z = torch.randn(1, 4, 64, 64).to(device)
        h = z / 0.18215 
        official_dec_out = official_vae.decoder(h)
        my_dec_out = decoder(h)
        print(f"Max: {my_dec_out.max().item()}, Min: {my_dec_out.min().item()}")
    mse_diff = F.mse_loss(official_dec_out, my_dec_out).item()

    print(f"Official Decoder :  {list(official_dec_out.shape)}")
    print(f"My decoder :        {list(my_dec_out.shape)}")
    print(f"Decoder MSE:        {mse_diff:.8f}")

def test_diffusion():
    official_unet = UNet2DConditionModel.from_pretrained(
        "../sd15_local", 
        subfolder="unet",
        local_files_only=True 
    ).to(device)
    official_unet.eval()
    unet_path = "../sd15_local/unet/diffusion_pytorch_model.safetensors"
    state_dict = load_file(unet_path)

    my_unet = Diffusion()
    pn.load_diffusion(my_unet, state_dict)
    dummy_latent = torch.randn(1, 4, 64, 64, device=device)
    dummy_timestep = torch.tensor([500], dtype=torch.long, device=device)
    dummy_context = torch.randn(1, 77, 768, device=device)
    my_unet.eval().to(device)
    with torch.no_grad():
        official_output = official_unet(
            sample=dummy_latent, 
            timestep=dummy_timestep, 
            encoder_hidden_states=dummy_context
        ).sample
        my_output = my_unet(dummy_latent, dummy_context, dummy_timestep)
    
    unet_max_diff = torch.max(torch.abs(official_output - my_output)).item()
    unet_mse_diff = F.mse_loss(official_output, my_output).item()
    print(f"Officail U-Net shape:   {list(official_output.shape)}")
    print(f"My U-Net shape:         {list(my_output.shape)}")
    print(f"U-Net Max Diff:         {unet_max_diff:.10f}")
    print(f"U-Net (MSE):            {unet_mse_diff:.10f}")








