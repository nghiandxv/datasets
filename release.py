import hashlib
from pathlib import Path
from zipfile import ZIP_LZMA, ZipFile

from tqdm import tqdm

DATA_DIR = Path.cwd() / 'data'
RELEASE_DIR = Path.cwd() / 'release'
DATA_DIR.mkdir(parents=True, exist_ok=True)
RELEASE_DIR.mkdir(parents=True, exist_ok=True)


def zip_dir(directory: Path, output_path: Path):
    with ZipFile(output_path, 'w', compression=ZIP_LZMA) as zip_file:
        for file_path in tqdm(sorted(directory.glob('*')), desc=f'Zipping to {output_path.name}', unit=' files'):
            if file_path.is_file():
                zip_file.writestr(file_path.name, file_path.read_bytes())


for dataset_dir in DATA_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue
    for split_dir in dataset_dir.iterdir():
        if not split_dir.is_dir():
            continue
        save_path = RELEASE_DIR / f'{dataset_dir.name}_{split_dir.name}.zip'
        zip_dir(split_dir, save_path)
        with open(save_path, 'rb') as file, open(save_path.with_suffix('.md5'), 'w') as hash_file:
            hash_file.write(hashlib.file_digest(file, 'md5').hexdigest())
