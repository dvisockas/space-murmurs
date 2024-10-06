import torch
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import glob
from torch.utils.data.dataset import Dataset
import pdb
import json
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

DRIVE_PATH = '/content/space'
BASE_PATH = DRIVE_PATH + '/data/lunar'
CATALOG_PATH = BASE_PATH + '/training/catalogs/apollo12_catalog_GradeA_final_final.csv'

class MurmurDataset(Dataset):
    def __init__(
        self,
        catalog_path = CATALOG_PATH,
        window_size = 10000,
        overlap = 3000
    ):
        version = '-2'
        self.non_event_cache_path = DRIVE_PATH + '/non_event_windows-' + str(window_size) + '-' + str(overlap) + version
        self.event_cache_path = DRIVE_PATH + '/event_windows-' + str(window_size) + '-' + str(overlap) + version
        print(self.non_event_cache_path, self.event_cache_path)
        print(os.path.isfile(self.non_event_cache_path), os.path.isfile(self.event_cache_path))
        if os.path.isfile(self.non_event_cache_path) and os.path.isfile(self.event_cache_path):
            self.non_event_windows = np.load(self.non_event_cache_path, allow_pickle=True)
            self.event_windows = np.load(self.event_cache_path, allow_pickle=True)
            return

        self.window_size = window_size
        self.overlap = overlap
        self.non_event_windows = []
        self.event_windows = []
        timestamp_format = '%Y-%m-%dT%H:%M:%S.%f'

        for _idx, row in tqdm(pd.read_csv(catalog_path).iterrows()):
            filename = BASE_PATH + '/training/data/S12_GradeA/' + row.filename + '.csv'
            event_timestamp = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], timestamp_format)
            shift = 1
            i = 0

            content = pd.read_csv(filename, skiprows=1, names=('timestamp', '_rel', 'velocity'))

            while True:
                offset = i*overlap + shift
                chunk = content.iloc[offset:(offset + window_size)]
                if len(chunk) == 0:
                    break

                first_timestamp = datetime.strptime(chunk.iloc[0]['timestamp'], timestamp_format)
                last_timestamp = datetime.strptime(chunk.iloc[-1]['timestamp'], timestamp_format)
                has_event = first_timestamp <= event_timestamp and last_timestamp >= event_timestamp

                delta_seconds = (last_timestamp - first_timestamp).total_seconds()
                seconds_after_event = (last_timestamp - event_timestamp).total_seconds()
                # don't include windows where an event starts at the very end of the window
                # if we get such a window, shift a bit to the right
                if (has_event and (seconds_after_event / delta_seconds < 0.1)):
                    print(delta_seconds, seconds_after_event)
                    shift = shift + 500
                    continue

                padded_signal = np.pad(chunk['velocity'], (0, self.window_size - len(chunk))) / 1.5e-07
                if (first_timestamp <= event_timestamp and last_timestamp >= event_timestamp):
                    self.event_windows.append(padded_signal)
                else:
                    self.non_event_windows.append(padded_signal)

                i = i + 1

        self.event_windows = np.array(self.event_windows)
        self.non_event_windows = np.array(self.non_event_windows)
        self.dump_to_cache()

    def __getitem__(self, index: int) -> list:
        mod = 5
        if (index % mod) == 0:
            return [self.non_event_windows[math.floor(index / mod) % len(self.non_event_windows)], 0]
        else:
            return [self.event_windows[index % mod - 1 + math.floor(index / mod) * (mod - 1) % len(self.event_windows)], 1]

    def __len__(self):
        return len(self.non_event_windows) * 5

    def dump_to_cache(self):
        self.non_event_windows.dump(self.non_event_cache_path)
        self.event_windows.dump(self.event_cache_path)
