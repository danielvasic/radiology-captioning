from datasets import RadiologyDataset, transform
from vocab import build

vocab = build()
dataset_path = "data"
train_dataset = RadiologyDataset(
    f"{dataset_path}/train/radiology/captions.txt", 
    f"{dataset_path}/train/radiology/images", 
    vocab, 
    transform
)

for cap, img in train_dataset:
    print(cap)
    print(img)
    break