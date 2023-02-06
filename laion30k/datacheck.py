import os
import re
import torch
import torchvision
import webdataset as wds
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from webdataset.handlers import warn_and_continue


def clean_caption(caption):
    caption = re.sub(" +", " ", caption)
    if caption[0] == "\"" or caption[0] == "'":
        caption = caption[1:]
    if caption[-1] == "\"" or caption[-1] == "'":
        caption = caption[:-1]
    return caption


def filter_captions(caption):
    possible_url_hints = ["www.", ".com", "http"]
    forbidden_characters = ["-", "_", ":", ";", "(", ")", "/", "%", "|", "?"]
    forbidden_words = ["download", "interior", "kitchen", "chair", "getty", "how", "what", "when", "why", "laminate", "furniture", "hair", "dress", "clothing"]
    if len(caption.split(" ")) < 2:
        print(False)
        return False
    if not all([False if i in caption else True for i in forbidden_characters]):
        print(False)
        return False
    if len(caption) > 150:
        print(False)
        return False
    if not all(ord(c) < 128 for c in caption):
        return False
    if not all([False if i in caption else True for i in possible_url_hints]):
        return False
    if any(char.isdigit() for char in caption):
        return False
    if not all([False if i in caption.lower() else True for i in forbidden_words]):
        return False
    return True


class ProcessDataV2:
    def __init__(self,):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
        ])

    def __call__(self, data):
        data["jpg"] = self.transforms(data["jpg"])
        return data


def collate(batch):
    images = torch.stack([i[0] for i in batch], dim=0)
    json_file = [i[1] for i in batch]
    captions = [i[2] for i in batch]
    return [images, json_file, captions]


def get_dataloader_new(path):
    dataset = wds.WebDataset(path, resampled=True, handler=warn_and_continue).decode("rgb", handler=warn_and_continue).map(
        ProcessDataV2(), handler=warn_and_continue).to_tuple("jpg", "json", "txt", handler=warn_and_continue).shuffle(1000, handler=warn_and_continue)
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate)
    return dataloader

dataset_length = 30_000
dataset_path = "30k"
print(os.getcwd())
os.makedirs(dataset_path, exist_ok=True)
# path = "file:000069.tar"
path = "path_to_laion_aesthetic_dataset"
dataloader = get_dataloader_new(path)
idx = 0


for _, (images, json_files, captions) in enumerate(dataloader):
    if idx < dataset_length:
        f = [i for i, json_file in enumerate(json_files) if json_file["AESTHETIC_SCORE"] > 6.0]
        if f:
            print(f)
            f = [i for i in f if filter_captions(captions[i])]
            captions = [clean_caption(captions[i]) for i in f if filter_captions(captions[i])]
            if f:
                print(captions)
                aesthetic_images = images[f]
                for image, caption in zip(aesthetic_images, captions):
                    torchvision.utils.save_image(image, os.path.join(dataset_path, f"{idx}.jpg"))
                    open(os.path.join(dataset_path, f"{idx}.txt"), "w").write(caption)
                    idx += 1
                # plt.figure(figsize=(32, 32))
                # plt.imshow(torch.cat([
                #     torch.cat([i for i in aesthetic_images.cpu()], dim=-1),
                # ], dim=-2).permute(1, 2, 0).cpu())
                # plt.show()

