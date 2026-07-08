import os
import torch
import numpy as np
from torch.utils.data import Dataset
import logging
from omegaconf import OmegaConf
import pandas as pd


class CTVideoDataset(Dataset):
    """CT generation dataset (ADNI, AIBL) for LDT."""

    def __init__(self, config=None):
        """
        Args:
            config: config object with the following fields:
                - csv_file: str, path to the csv with pre/post DX information
                - token_dir: str, token directory (default 'data/processed_np_case')
        """
        super().__init__()
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)
        self.config = config
        self.csv_file = config['csv_file']
        self.token_dir = config.get('token_dir', 'data/processed_np_case')

        self.subjects_data = []
        self.load_data()

        logging.info(f"Loaded {len(self.subjects_data)} subjects from {self.csv_file}")

    def load_data(self):
        # csv columns: pre_path, post_path, time_dif, pre_dx, post_dx, dx_change
        data = pd.read_csv(self.csv_file)
        for index, row in data.iterrows():
            pre_ct_path = self._to_token_path(row['pre_path'])
            post_ct_path = self._to_token_path(row['post_path'])

            pre_token = self._load_token(pre_ct_path)
            post_token = self._load_token(post_ct_path)

            self.subjects_data.append({
                'pre_token': pre_token,
                'post_token': post_token,
                'post_dx': row['post_dx'],
                'time_diff': row['time_dif']
            })

    def _to_token_path(self, npy_name: str) -> str:
        # csv stores the flattened, anonymized npy filename; join it with token_dir
        return os.path.join(self.token_dir, npy_name)

    def _load_token(self, token_path: str) -> torch.Tensor:
        try:
            if os.path.exists(token_path):
                token_data = torch.from_numpy(np.load(token_path))
                return token_data
            else:
                logging.warning(f"Token file not found: {token_path}")
                return None
        except Exception as e:
            logging.error(f"Error loading token {token_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.subjects_data)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self.subjects_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.subjects_data)}")

        subject_data = self.subjects_data[idx]

        pretoken = subject_data['pre_token']
        posttoken = subject_data['post_token']
        post_dx = subject_data['post_dx']
        # bucket month gap into bins: 1-6 -> 0; 7-12 -> 1; 13-18 -> 2 ...
        time_diff = (subject_data['time_diff'] - 1) // 6

        return {
            'pre_token': pretoken,
            'post_token': posttoken,
            'post_dx': post_dx,
            'time_diff': time_diff
        }
