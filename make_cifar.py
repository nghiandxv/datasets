import pickle
import tarfile
from functools import partial
from pathlib import Path

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

cifar10_file_name = 'cifar-10-python.tar.gz'
cifar10_url = f'https://www.cs.toronto.edu/~kriz/{cifar10_file_name}'
cifar100_file_name = 'cifar-100-python.tar.gz'
cifar100_url = f'https://www.cs.toronto.edu/~kriz/{cifar100_file_name}'

cifar10_train_paths = [download_dir / 'cifar-10-batches-py' / f'data_batch_{i}' for i in range(1, 6)]
cifar10_test_paths = [download_dir / 'cifar-10-batches-py' / 'test_batch']
cifar100_train_paths = [download_dir / 'cifar-100-python' / 'train']
cifar100_test_paths = [download_dir / 'cifar-100-python' / 'test']


def download_and_extract(url: str, download_dir: Path, file_name: str):
    download_path = download_dir / file_name
    with httpx.stream('GET', url) as response, open(download_path, 'wb') as file:
        response.raise_for_status()
        print(f'Downloading {file_name}...')
        for chunk in tqdm(response.iter_bytes(), unit=' chunks'):
            file.write(chunk)
    with tarfile.open(download_path) as archive:
        archive.extractall(download_dir)


def load_and_save(save_dir: Path, paths: list[Path]):
    batched_images, batched_labels = [], []
    for path in paths:
        with open(path, 'rb') as file:
            entry = pickle.load(file, encoding='latin1')
            images = ei.rearrange(entry['data'], 'n (c h w) -> n h w c', c=3, h=32, w=32)
            labels = np.asarray(entry['labels' if 'labels' in entry else 'fine_labels'])
            batched_images.append(images)
            batched_labels.append(labels)

    all_images = ei.pack(batched_images, '* h w c')[0]
    all_labels = ei.pack(batched_labels, '*')[0]
    id_length = len(str(len(all_images)))
    label_length = len(str(max(all_labels)))

    print(f'Saving {len(all_images)} images to {save_dir}')
    for i, (image, label) in tqdm(enumerate(zip(all_images, all_labels)), unit=' images'):
        image = Image.fromarray(image)
        save_path = save_dir / f'{str(i).zfill(id_length)}.{str(label).zfill(label_length)}.webp'
        image.save(save_path, lossless=True, quality=100)


cifar10_train_save_dir = data_dir / 'cifar10' / 'train'
cifar10_test_save_dir = data_dir / 'cifar10' / 'test'
cifar10_train_save_dir.mkdir(parents=True, exist_ok=True)
cifar10_test_save_dir.mkdir(parents=True, exist_ok=True)
download_and_extract(cifar10_url, download_dir, cifar10_file_name)
load_and_save(cifar10_train_save_dir, cifar10_train_paths)
load_and_save(cifar10_test_save_dir, cifar10_test_paths)

cifar100_train_save_dir = data_dir / 'cifar100' / 'train'
cifar100_test_save_dir = data_dir / 'cifar100' / 'test'
cifar100_train_save_dir.mkdir(parents=True, exist_ok=True)
cifar100_test_save_dir.mkdir(parents=True, exist_ok=True)
download_and_extract(cifar100_url, download_dir, cifar100_file_name)
load_and_save(cifar100_train_save_dir, cifar100_train_paths)
load_and_save(cifar100_test_save_dir, cifar100_test_paths)
