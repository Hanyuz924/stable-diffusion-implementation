import sys
import os
from tqdm import tqdm 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets
import models.pipleline as pipleline
from torchvision import transforms
import torch
from safetensors.torch import save_file
from diffusers.optimization import get_scheduler
import finetuen.lora as lora
from diffusers import DDIMScheduler
import wandb
from torch.utils.data import Dataset, DataLoader



class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.num_samples = len(data_dict["image"])
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.data_dict["image"][idx],
            "input_ids": self.data_dict["input_ids"][idx]
        }

def check_dataset(data):
    print(type(data))
    print(data["train"].column_names)
    print(data["train"]["Image"][0:2])

def save_weights(models):
    models_dict = models.state_dict()
    save_file(models_dict, "lora.safetensors")

def preprocess_data(transforms, tokenizer, data):

    images = [transforms(image.convert("RGB")) for image in data["train"]["Image"]]
    tokens = tokenizer(
                    data["train"]["Caption"][:],
                    max_length=tokenizer.model_max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
    return {
        "image": images,
        "input_ids": tokens.input_ids
    }

def inject_lora(models:torch.nn.Module, target_modules=["to_k", "to_q", "to_v", "to_out.0"], rank = 8, lora_alpha = 16, init_weights_method = "gaussian"):
    replaced_module = []
    for name, module in models.named_modules():
        if isinstance(module, torch.nn.Linear) and any(target in name for target in target_modules):
            replaced_module.append((name, module))

    for name, module in replaced_module:
        name_splits = name.split(".")
        child_name = name_splits[-1]
        parent_name = ".".join(name_splits[:-1])
        parent_module = models if parent_name == "" else models.get_submodule(parent_name)
        lora_layer = lora.LoraLayer(base_layer= module,
                                    rank= rank,
                                    lora_alpha=lora_alpha,
                                    init_method=init_weights_method
                                    )
        setattr(parent_module, child_name, lora_layer)
    return models
    
  
def main():
    #hyper params
    bs = 4
    lr = 1e-5
    data_loader_workers = 4
    train_epoch = 20
    #get models and weights 
    dataset = datasets.load_dataset("Dhiraj45/Anime-Caption")
    models = pipleline.load_model(load_nuet=True,
                                  load_decoder=False,
                                  load_encoder=True,
                                  load_tokenizer=True)
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    encoder = models["encoder"]
    scheduler = DDIMScheduler.from_pretrained(
        "../sd15_local", 
        subfolder="scheduler",
        local_files_only=True
    )

    unet = models["Unet"]
    unet = inject_lora(unet)


    for name, parameters in unet.named_parameters():
        if "lora" in name:
            parameters.requires_grad = True
        else:
            parameters.requires_grad = False
    TRAIN_W, TRAIN_H = 768, 448
    train_transforms = transforms.Compose([
        transforms.Resize((TRAIN_H, TRAIN_W), transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    processed_data = preprocess_data(train_transforms, tokenizer, dataset)
    train_dataset = DictDataset(processed_data)

    #dataloader:
    train_dataloader = torch.utils.data.DataLoader(
        dataset= train_dataset,
        batch_size= bs,
        shuffle= True,
        num_workers=data_loader_workers
    )
    # prepare optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    scheduler_traning_step = train_epoch * len(train_dataloader)
    scheduler_warm_up_step = min(int(scheduler_traning_step * 0.1), 500)
    lr_scheduler = get_scheduler(
        "cosine", 
        optimizer=optimizer,
        num_warmup_steps=scheduler_warm_up_step,
        num_training_steps=scheduler_traning_step,
    )
    wandb.init(
            project="sd-1-5-lora-training", 
            name="lora_training",           
            config={                       
                "learning_rate": lr,
                "epochs": train_epoch,
                "batch_size": bs,
        }
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(processed_data)}")
    print(f"  Num Epochs = {train_epoch}")
    print(f"  batch size = {bs}")
    global_step = 0
    total_steps = train_epoch * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training Steps")
    for epoch in range(train_epoch):
        unet.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            pxiel_value = batch["image"].to(device， dtype=torch.float32)
            caption_encode = batch["input_ids"].to(deivce)
            latent = encoder(pxiel_value) 
            bs = latent.shape[0]
            time_step = torch.randint(0, scheduler.num_train_timesteps, size=(bs, ), device=latent.device).long()
            noise = torch.randn_like(latent)
            noisy_latents = scheduler.add_noise(latent, noise, time_step) # type: ignore
            text_hiddent = text_encoder(caption_encode, return_dict=False, device=latent.device)[0]
            model_pre = unet(noisy_latents, text_hiddent, time_step)
            loss = torch.nn.functional.mse_loss(model_pre.float(), noise.float(), reduction="mean")
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": lr_scheduler.get_last_lr()[0]})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            wandb.log({
                "train/step_loss": loss.item(),
                "global_step": global_step
            })
            wandb.log({
                "LR":  lr_scheduler.get_last_lr()[0],
                "global_step": global_step
            })
            global_step += 1
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        wandb.log({
            "train/epoch_avg_loss": avg_epoch_loss,
            "epoch": epoch
        })
    
if __name__ == "__main__":
    main()



