import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CustomMotorcycleDataset(Dataset):
    def __init__(self, folder_path, tokenizer, size=512):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.image_paths = []
        self.captions = []
        
        valid_extensions = (".png")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                base_name = os.path.splitext(filename)[0]
                img_path = os.path.join(folder_path, filename)
                txt_path = os.path.join(folder_path, f"{base_name}.txt")
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    self.image_paths.append(img_path)
                    self.captions.append(caption)
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        caption = self.captions[idx]
        inputs = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze(0) 
        
        return {"pixel_values": pixel_values, "input_ids": input_ids}
