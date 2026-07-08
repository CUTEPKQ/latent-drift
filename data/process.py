import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from typing import Optional, Tuple, List
import logging
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import math
import glob


def save_nii_paths_to_txt(folder_path, output_txt_path):
    """
    Scan a folder for all .nii.gz files and save their paths to a txt file.

    Args:
        folder_path (str): folder to scan
        output_txt_path (str): output txt file path
    """
    nii_files = glob.glob(os.path.join(folder_path, '**', '*.nii.gz'), recursive=True)
    with open(output_txt_path, 'w') as f:
        for file_path in nii_files:
            f.write(file_path + '\n')
    print(f"Saved {len(nii_files)} .nii.gz file paths to {output_txt_path}")

class CTVideoDataset():
    """
    CT video dataset class for loading and processing .nii.gz files.
    """

    def __init__(self, config):
        """
        Initialize the CT video dataset.

        Args:
            config: config object with the following fields:
                - txt_file: str, txt file listing .nii.gz file paths
                - input_root: str, root dir of the input .nii.gz files
                - output_dir: str, output dir for the processed .npy files
                - pad_target: tuple, target padding size, default (192, 224, 185)
                - out_shape: tuple, output shape, default (96, 112, 93)
                - rotate90_hw: bool, whether to rotate 90 degrees in the H-W plane, default True
                - hu_clip: tuple, HU clipping range, default (-1000.0, 1000.0)
        """
        self.config = config
        self.txt_file = config.txt_file
        self.input_root = getattr(config, 'input_root', 'data/raw')
        self.output_dir = getattr(config, 'output_dir', 'data/processed_np_case')
        self.pad_target = getattr(config, 'pad_target', (192, 224, 185))
        self.out_shape = getattr(config, 'out_shape', (96, 112, 93))
        self.rotate90_hw = getattr(config, 'rotate90_hw', True)
        self.hu_clip = getattr(config, 'hu_clip', (-1000.0, 1000.0))
        self.num_threads = getattr(config, 'num_threads', 16)  # number of worker threads

        # Load the list of file paths
        self.file_paths = self._load_file_paths()

        # Preload all data
        self.loaded_data = []
        self.load_data()

        logging.info(f"Loaded {len(self.file_paths)} CT files from {self.txt_file}")

    def load_data(self):
        """
        Preload all CT data into memory using multiple threads.
        """
        n_files = len(self.file_paths)
        logging.info(f"Starting to load {n_files} CT files using multi-threading (unordered, no chunk)...")

        def process_single_file(file_path):
            try:
                ct_tensor = self.nii_to_tensor_pytorch(
                    path=file_path,
                    pad_target=self.pad_target,
                    out_shape=self.out_shape,
                    rotate90_hw=self.rotate90_hw,
                    hu_clip=self.hu_clip,
                )
                return {
                    'data': ct_tensor,                 # shape: (1, D, H, W)
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'shape': ct_tensor.shape,
                }
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                return None

        # Collect results with append (unordered); no [None] * n preallocation
        self.loaded_data = []

        num_threads = min(self.num_threads, n_files)
        with ThreadPool(processes=num_threads) as pool:
            with tqdm(total=n_files, desc="Loading CT files") as pbar:
                # unordered, no chunksize
                for data_item in pool.imap_unordered(process_single_file, self.file_paths):
                    if data_item is not None:
                        self.loaded_data.append(data_item)
                    pbar.update(1)

        successful_count = len(self.loaded_data)
        logging.info(f"Finished loading all CT files. Successfully loaded: {successful_count}/{n_files}")

        if successful_count == 0:
            raise ValueError("No files were successfully loaded!")

    def _load_file_paths(self) -> List[str]:
        """
        Read .nii.gz file paths from the txt file.

        Returns:
            List[str]: list of file paths
        """
        file_paths = []

        if not os.path.exists(self.txt_file):
            raise FileNotFoundError(f"Text file not found: {self.txt_file}")

        with open(self.txt_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    file_paths.append(path)
                elif path:
                    logging.warning(f"File not found: {path}")

        if not file_paths:
            raise ValueError(f"No valid files found in {self.txt_file}")

        return file_paths

    @torch.no_grad()
    def nii_to_tensor_pytorch(
        self,
        path: str,
        pad_target: Tuple[int, int, int] = (192, 224, 185),
        out_shape: Tuple[int, int, int] = (96, 112, 93),
        rotate90_hw: bool = True,
        hu_clip: Tuple[float, float] = (-1000.0, 1000.0)
    ) -> torch.Tensor:
        """
        Convert a NIfTI file into a PyTorch tensor.

        Args:
            path: .nii.gz file path
            pad_target: target padding size (H, W, D)
            out_shape: output shape (H, W, D)
            rotate90_hw: whether to rotate 90 degrees in the H-W plane
            hu_clip: HU clipping range

        Returns:
            torch.Tensor: tensor of shape (1, D, H, W)
        """
        try:
            # 1) Read data -> HU -> clip
            nii = nib.load(str(path))
            vol = nii.get_fdata().astype(np.float32) - 1024.0
            vol = np.clip(vol, hu_clip[0], hu_clip[1])

            # # 2) Normalize to [0, 1]
            # vol = (vol - hu_clip[0]) / (hu_clip[1] - hu_clip[0])  # [-1000,1000] -> [0,1]
            # 2) Normalize to [-1, 1]
            vol = vol/1000  # [-1000,1000] -> [-1,1]
            H, W, D = vol.shape

            # 3) numpy -> torch, shape (1,1,D,H,W)
            t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

            # 4) Center pad
            tH, tW, tD = pad_target
            d_pad = max(tD - D, 0)
            h_pad = max(tH - H, 0)
            w_pad = max(tW - W, 0)
            pads = (w_pad//2, w_pad - w_pad//2,
                    h_pad//2, h_pad - h_pad//2,
                    d_pad//2, d_pad - d_pad//2)
            t = F.pad(t, pads, mode='constant', value=-1.0)  # pad background with -1 (air = -1)

            # 5) Trilinear downsampling
            outH, outW, outD = out_shape
            t = F.interpolate(t, size=(outD, outH, outW),
                              mode='trilinear', align_corners=False)

            # 6) Optional rotation, enabled by default
            if rotate90_hw:
                t = torch.rot90(t, k=1, dims=(3, 4))

            t = t.squeeze(0)  # (1, D, H, W)

            # Mirror the input sub-path under output_dir, replacing the .nii.gz suffix with .npy
            rel_path = os.path.relpath(str(path), self.input_root)
            output_path = os.path.join(self.output_dir, rel_path)
            output_path = output_path.replace(".nii.gz", ".npy")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, t.cpu().numpy())

            return t

        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            raise


# Example config class
class CTVideoConfig:
    """Config class for the CT video dataset."""

    def __init__(
        self,
        txt_file: str,
        input_root: str = 'data/raw',
        output_dir: str = 'data/processed_np_case',
        pad_target: Tuple[int, int, int] = (192, 224, 185),
        out_shape: Tuple[int, int, int] = (96, 112, 93),
        rotate90_hw: bool = True,
        hu_clip: Tuple[float, float] = (-1000.0, 1000.0),
        num_threads: int = 8  # number of worker threads
    ):
        self.txt_file = txt_file
        self.input_root = input_root
        self.output_dir = output_dir
        self.pad_target = pad_target
        self.out_shape = out_shape
        self.rotate90_hw = rotate90_hw
        self.hu_clip = hu_clip
        self.num_threads = num_threads




if __name__ == "__main__":
    # Test code
    import time
    logging.basicConfig(level=logging.INFO)

    # Example config
    config = CTVideoConfig(
        txt_file="data/my-dataset.txt",
        input_root="data/raw",
        output_dir="data/processed_np_case",
        pad_target=(192, 224, 185),
        out_shape=(96, 112, 93),
        rotate90_hw=True,
        hu_clip=(-1000.0, 1000.0),
        num_threads=8
    )

    # save_nii_paths_to_txt(
    #     folder_path="data/raw",
    #     output_txt_path="data/my-dataset.txt"
    # )
    try:
        # Create the dataset (this preloads all data)
        print("Creating dataset and loading all data...")
        dataset = CTVideoDataset(config)
        print(f"Dataset size: {len(dataset.file_paths)}")
        print(f"Successfully loaded: {len(dataset.loaded_data)}/{len(dataset.file_paths)} files")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please make sure the txt file exists and contains valid .nii.gz file paths")
    except Exception as e:
        print(f"Error: {e}")
