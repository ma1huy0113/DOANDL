import os
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        for file in os.listdir(root_dir):
            if file.endswith(".png"):
                txt = file.replace(".png", ".txt")
                self.samples.append((file, txt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, txt_name = self.samples[idx]

        img_path = os.path.join(self.root_dir, img_name)
        txt_path = os.path.join(self.root_dir, txt_name)

        image = Image.open(img_path).convert("RGB")

        with open(txt_path, "r", encoding="utf-8") as f:
            label = f.read().strip()

        if self.transform:
            image = self.transform(image)

        return image, label