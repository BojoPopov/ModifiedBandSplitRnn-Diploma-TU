import os

import torch
import typing as tp

from sympy.stats.sampling.sample_numpy import numpy


class SAD:
    "SAD(Source Activity Detector)"
    def __init__(
            self,
            sr: int,
            window_size_in_sec: int = 8,
            overlap_ratio: float = 0.5,
            n_chunks_per_segment: int = 10,
            eps: float = 1e-5,
            gamma: float = 1e-3,
            threshold_max_quantile: float = 0.15,
            threshold_segment: float = 0.5,
    ):
        self.sr = sr
        self.n_chunks_per_segment = n_chunks_per_segment
        self.eps = eps
        self.gamma = gamma
        self.threshold_max_quantile = threshold_max_quantile
        self.threshold_segment = threshold_segment

        self.window_size = sr * window_size_in_sec
        self.step_size = int(self.window_size * overlap_ratio)
    def chunk(self, y: torch.Tensor):
        y = y.unfold(-1, self.window_size, self.step_size)
        y = y.chunk(self.n_chunks_per_segment, dim=-1)
        y = torch.stack(y, dim=-2)
        return y
    @staticmethod
    def calculate_rms(y: torch.Tensor):
        y_squared = torch.pow(y, 2) # need to square signal before mean and sqrt
        y_mean = torch.mean(torch.abs(y_squared), dim=-1, keepdim=True)
        y_rms = torch.sqrt(y_mean)
        return y_rms
    def calculate_thresholds(self, rms: torch.Tensor):
        rms[rms == 0.] = self.eps
        rms_threshold = torch.quantile(
            rms,
            self.threshold_max_quantile,
            dim=-2,
            keepdim=True,
        )
        rms_threshold[rms_threshold < self.gamma] = self.gamma
        rms_percentage = torch.mean(
            (rms > rms_threshold).float(),
            dim=-2,
            keepdim=True,
        )
        rms_mask = torch.all(rms_percentage > self.threshold_segment, dim=0).squeeze()
        return rms_mask
    def calculate_salient(self, y: torch.Tensor, mask: torch.Tensor):
        y = y[:, mask, ...]
        C, D1, D2, D3 = y.shape
        y = y.view(C, D1, D2*D3)
        return y
    def __call__(self,y: torch.Tensor,segment_saliency_mask: tp.Optional[torch.Tensor] = None):
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        if segment_saliency_mask is None:
            segment_saliency_mask = self.calculate_thresholds(rms)
        y_salient = self.calculate_salient(y, segment_saliency_mask)
        return y_salient, segment_saliency_mask

    def calculate_salient_indices(self, y: torch.Tensor):
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        mask = self.calculate_thresholds(rms)
        indices = torch.arange(mask.shape[-1])[mask] * self.step_size
        return indices.tolist()
if __name__ == "__main__":
    import torchaudio
    import os

    # Set sample rate and root directory path
    sr = 44100
    root_dir = r"D:\User\Desktop\musdb_normal\train_files"
    output_dir = r"D:\User\Desktop\musdb_normal\eight_seconds"

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    sad = SAD(sr=sr)

    # Iterate over all files in the directory
    for filename in os.listdir(root_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(root_dir, filename)
            y, sr = torchaudio.load(file_path)
            y_salient, saliency_mask = sad(y)  # Apply SAD to get salient segments and mask
            for i, (chunk, mask_value) in enumerate(zip(y_salient.permute(1, 0, 2), saliency_mask)):
                if mask_value:  # Only save if saliency mask is 1
                    # Remove the last 4 characters from the filename (i.e., ".wav")
                    filename_without_extension = filename[:-4]

                    # Construct the new filepath
                    chunk_filepath = os.path.join(output_dir, f"{filename_without_extension}_salient_chunk_{i}.wav")

                    # Save the chunk
                    torchaudio.save(chunk_filepath, chunk, sr)
                    print(f"Saved salient chunk {i} for {filename} with shape {chunk.shape}")
                else:
                    print(f"Skipping non-salient chunk {i} for {filename}")
