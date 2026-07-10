from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
import random
from multiprocessing import Pool, cpu_count


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L126
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.JPEG'}

def get_all_files(root_dir):
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1] in IMG_EXTENSIONS:
                all_files.append(os.path.join(root, file))
    random.shuffle(all_files)
    return all_files


def _load_and_crop(args):
    """Worker function for multiprocessing: load one image and center crop."""
    file_path, image_size = args
    try:
        pil_image = Image.open(file_path).convert('RGB')
        arr = center_crop_arr(pil_image, image_size)
        arr = np.asarray(arr).astype(np.uint8)
        if arr.shape == (image_size, image_size, 3):
            return arr
    except Exception:
        pass
    return None


def create_npz_from_sample_folder(sample_dir, num=50_000, image_size=256, npz_path=None, num_workers=None):
    """
    Builds a single .npz file from a folder of images (multiprocessing).
    """
    files = get_all_files(sample_dir)
    assert len(files) >= num, f"Found {len(files)} samples, but need at least {num} samples."
    files = files[:num]  # only process needed files

    if num_workers is None:
        num_workers = min(cpu_count(), 32)
    print(f"Loading {num} images with {num_workers} workers...")

    work_args = [(f, image_size) for f in files]
    samples = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(_load_and_crop, work_args), total=len(work_args), desc="Building .npz file"):
            if result is not None:
                samples.append(result)

    samples = np.stack(samples[:num])
    print(samples.shape)
    assert samples.shape == (num, image_size, image_size, 3)

    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_folder', type=str, required=True, help='Path to ImageNet val folder')
    parser.add_argument('--output', type=str, required=True, help='Output .npz path')
    parser.add_argument('--num', type=int, default=50_000)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=None, help='number of parallel workers (default: cpu_count)')
    args = parser.parse_args()
    create_npz_from_sample_folder(args.ref_folder, num=args.num, image_size=args.image_size, npz_path=args.output, num_workers=args.num_workers)
    print("Done.")