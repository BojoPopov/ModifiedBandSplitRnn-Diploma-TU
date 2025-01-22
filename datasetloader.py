import os
import random
from random import randint

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchaudio
import torch
from  time_stretch_modified import time_stretch_torch

class MultiSourceDataset(Dataset):
    def __init__(self, root_dir, segment_duration=4.0, sample_rate=44100, augmentations=True):
        self.root_dir = root_dir
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.do_augmentations = augmentations

        # List all files in each subfolder
        self.bass_files = self._get_files(os.path.join(root_dir, "bass"))
        self.drums_files = self._get_files(os.path.join(root_dir, "drums"))
        self.other_files = self._get_files(os.path.join(root_dir, "other"))
        self.vocals_files = self._get_files(os.path.join(root_dir, "vocals"))

    def _get_files(self, folder):
        return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".wav")]
    def augmentations(self, x, is_vocal, target_length=44100*4):
        if  not is_vocal:
            # print("noise")
            x = x+torch.randn_like(x)*random.uniform(0.0, 0.02)
        # # print("pitch")
        # step = random.randint(-2,2)
        #
        # pitch_shift_transform = torchaudio.transforms.PitchShift(44100, step,12,525).cuda()
        # x= pitch_shift_transform(x)
        # # print("stretch")
        # x = time_stretch_torch(x, random.uniform(0.8,1.2), hop_length=525, n_fft=2048)
        if random.random()>0.5:
            # print("flip cahnnels")
            x = x.flip(0)
        # print("flip sign")
        x = x * random.choice([1, -1])
        # print("gain")
        x = x * random.uniform(0.8,1.2)
        # print("shift")
        x = torch.roll(x, random.randint((-x.size(dim=-1))//2, (x.size(dim=-1))//2), dims=-1)
        # Ensure x matches the target length
        if x.shape[-1] > target_length:
            # If longer, truncate to target length
            x = x[..., :target_length]
        elif x.shape[-1] < target_length:
            # If shorter, pad with zeros to target length
            padding_size = target_length - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, padding_size), mode='constant', value=0)

        return x

    def __len__(self):
        return len(self.vocals_files)
        # return 10


    def _load_random_file(self, files):
        file_path = random.choice(files)
        waveform, _ = torchaudio.load(file_path)
        start_sample = randint(0,176400)
        end_sample = start_sample+176400

        # Slice the waveform
        waveform = waveform[:, start_sample:end_sample]
        return waveform

    def __getitem__(self, idx):

        bass = self._load_random_file(self.bass_files)
        drums = self._load_random_file(self.drums_files)
        other = self._load_random_file(self.other_files)
        vocals = self._load_random_file(self.vocals_files)
        bass = self.augmentations(bass, False)
        drums = self.augmentations(drums, False)
        other = self.augmentations(other, False)
        vocals = self.augmentations(vocals, True)
        # Stack the sources into a single tensor with shape (4, 2, segment_length)
        sources = torch.stack([bass, drums, other, vocals], dim=0)
        return sources

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=16, segment_duration=4.0, sample_rate=44100, augmentations=True):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.augmentations = augmentations

    def setup(self, stage=None):
        self.train_dataset = MultiSourceDataset(
            self.root_dir,
            segment_duration=self.segment_duration,
            sample_rate=self.sample_rate,
            augmentations=self.augmentations
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True)

    def val_dataloader(self):
        # Optionally, use a different dataset or the same with different parameters
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
