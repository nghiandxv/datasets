from functools import partial
from pathlib import Path
from zipfile import ZipFile

import einops as ei
import httpx
import numpy as np
from PIL import Image
from tqdm import tqdm

tqdm = partial(tqdm, ncols=0, leave=False)

download_dir = Path.cwd() / 'download'
data_dir = Path.cwd() / 'data'
download_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

tiny_imagenet_file_name = 'tiny-imagenet-200.zip'
tiny_imagenet_url = f'https://cs231n.stanford.edu/{tiny_imagenet_file_name}'

with httpx.stream('GET', tiny_imagenet_url) as response, open(download_dir / tiny_imagenet_file_name, 'wb') as file:
    response.raise_for_status()
    print(f'Downloading {tiny_imagenet_file_name}...')
    for chunk in tqdm(response.iter_bytes(), unit=' chunks'):
        file.write(chunk)

with ZipFile(download_dir / tiny_imagenet_file_name) as archive:
    archive.extractall(download_dir)

extracted_dir = download_dir / 'tiny-imagenet-200'

with open(extracted_dir / 'wnids.txt', 'r') as file:
    wnids = [line.strip() for line in file]
wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}


def read_image(file_path: Path):
    data = np.asarray(Image.open(file_path))
    if len(data.shape) == 2:
        return ei.repeat(data, 'h w -> h w c', c=3)
    return data


train_images, train_labels = [], []
for wnid, label in wnid_to_label.items():
    class_dir = extracted_dir / 'train' / wnid
    for file_path in sorted((class_dir / 'images').glob('*.JPEG')):
        train_image = read_image(file_path)
        train_images.append(train_image)
        train_labels.append(label)

val_images, val_labels = [], []
with open(extracted_dir / 'val' / 'val_annotations.txt', 'r') as file:
    for line in file:
        val_file_name, val_wnid = line.split('\t')[:2]
        val_image = read_image(extracted_dir / 'val' / 'images' / val_file_name)
        val_images.append(val_image)
        val_labels.append(wnid_to_label[val_wnid])


def save(images: np.ndarray, labels: np.ndarray, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    id_length = len(str(len(images)))
    label_length = len(str(max(labels)))
    print(f'Saving {len(images)} images to {save_dir}')
    for i, (image, label) in tqdm(enumerate(zip(images, labels)), unit=' images'):
        image = Image.fromarray(image)
        save_path = save_dir / f'{str(i).zfill(id_length)}.{str(label).zfill(label_length)}.webp'
        image.save(save_path, lossless=True, quality=100)


save(ei.pack(train_images, '* h w c')[0], np.asarray(train_labels), data_dir / 'tiny_imagenet_200' / 'train')
save(ei.pack(val_images, '* h w c')[0], np.asarray(val_labels), data_dir / 'tiny_imagenet_200' / 'test')
