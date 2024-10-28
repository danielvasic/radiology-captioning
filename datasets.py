import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class RadiologyDataset(Dataset):
    def __init__(self, caption_file, image_dir, vocab, transform=None):
        """
        Args:
            caption_file (str): Path to the caption file.
            image_dir (str): Directory where images are stored.
            vocab (Vocabulary): Vocabulary object to tokenize the captions.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.data = []

        with open(caption_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_name, caption = parts
                    image_path = os.path.join(image_dir, image_name + ".jpg")
                    if os.path.exists(image_path):
                        self.data.append((image_path, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Tokenize the caption using the vocabulary
        tokens = [self.vocab.stoi["<start>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<end>"]]


        return image, torch.tensor(tokens)


# Collate Function
def collate_fn(batch):
    images, captions = zip(*batch)
    lengths = [len(cap) for cap in captions]
    padded_captions = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]

    return torch.stack(images, 0), padded_captions, lengths

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
