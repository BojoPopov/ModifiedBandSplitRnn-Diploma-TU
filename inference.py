import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import pytorch_lightning as pl
from ema_lightning_bandsplit_no_learnableSTFT_no_NormOnBand import BandSplitRNN
splits_v7 = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
def pad_audio(audio, target_length):
    """Pad audio to make its length divisible by the target length."""
    num_samples = len(audio[1])
    pad_amount = (target_length - (num_samples % target_length)) % target_length
    print(pad_amount)
    if pad_amount > 0:
        print(audio.shape)
        audio = torch.nn.functional.pad(audio, (0,pad_amount))
        print(audio.shape)
    return audio
def overlap_add(chunks, step_size, original_length):
    """Reconstruct the audio from overlapping chunks using weighted overlap-add with Hann window."""
    step_length = int(step_size)
    chunk_length = chunks.shape[2]  # Time dimension of a chunk
    num_chunks = chunks.shape[0]    # Number of chunks
    # Initialize an empty output buffer and weight accumulator
    output = torch.zeros(2, original_length)
    weight_sum = torch.zeros(2, original_length)
    # Generate a Hann window
    hann_window = torch.hann_window(chunk_length, periodic=True).to(chunks.device)
    hann_window = hann_window.unsqueeze(0).repeat(2, 1)  # Shape: [2, chunk_length]
    for i in range(num_chunks):
        start = i * step_length
        end = start + chunk_length
        # Apply Hann window to the current chunk
        windowed_chunk = chunks[i] * hann_window
        # Add windowed chunk to the output buffer
        output[:, start:end] += windowed_chunk
        weight_sum[:, start:end] += hann_window
    # Avoid division by zero where weights are zero
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    # Normalize the output by the accumulated weights
    output = output / weight_sum
    return output
def split_into_chunks(audio, sample_rate, chunk_size, step_size):
    """Split audio into overlapping chunks."""
    chunk_length = int(chunk_size * sample_rate)
    step_length = int(step_size * sample_rate)
    chunks = []
    for i in range(0, len(audio[1]) - chunk_length +1, step_length):
        print('hello')
        chunks.append(audio[:, i:i + chunk_length])
    return torch.stack(chunks)
def process_audio_with_model(model, audio, sample_rate, chunk_size=4, step_size=2):
    """Process the audio file with a PyTorch Lightning model."""
    # Pad the audio

    audio = pad_audio(audio, chunk_size * sample_rate)
    original_length = len(audio[1])
    # Split into overlapping chunks
    chunks = split_into_chunks(audio, sample_rate, chunk_size, step_size)

    # Run each chunk through the model
    bass = []
    drums = []
    other = []
    vocals = []
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.unsqueeze(0) # Add batch dimension
            bass_chunk, drums_chunk, other_chunk, vocals_chunk = model(chunk)  # Remove batch dimension
            bass.append(bass_chunk.squeeze(0))
            drums.append(drums_chunk.squeeze(0))
            other.append(other_chunk.squeeze(0))
            vocals.append(vocals_chunk.squeeze(0))

    # Reconstruct the audio
    processed_bass = overlap_add(torch.stack(bass),step_size * sample_rate,original_length)
    processed_drums = overlap_add(torch.stack(drums),step_size * sample_rate,original_length)
    processed_other= overlap_add(torch.stack(other),step_size * sample_rate,original_length)
    processed_vocals= overlap_add(torch.stack(vocals),step_size * sample_rate,original_length)
    return processed_bass, processed_drums, processed_other,  processed_vocals


def main(audio_path, model_checkpoint, output_path, sample_rate=44100):
    model = BandSplitRNN.load_from_checkpoint(model_checkpoint,bandsplits=splits_v7, num_layers=6, map_location='cpu')
    audio, sr = torchaudio.load(audio_path)
    if sr != 44100:
        audio = Resample(orig_freq=sr, new_freq=44100).forward(audio)
    bass, drums, other, vocals = process_audio_with_model(model, audio, 44100)
    torchaudio.save(r"D:\User\Desktop\musdb_normal\train\Aimee Norwich - Child\b.wav", bass, 44100)
    torchaudio.save(r"D:\User\Desktop\musdb_normal\train\Aimee Norwich - Child\d.wav", drums, 44100)
    torchaudio.save(r"D:\User\Desktop\musdb_normal\train\Aimee Norwich - Child\o.wav", other, 44100)
    torchaudio.save(r"D:\User\Desktop\musdb_normal\train\Aimee Norwich - Child\v_lili.wav", vocals, 44100)


# Example usage
audio_path = r"D:\User\Desktop\New folder (5)\ЛИЛИ ИВАНОВА_ ВЕТРОВЕ  LILI IVANOVA_ WINDS (OFFICIAL VIDEO).mp3"
# model_checkpoint = r"D:\Users\User\PycharmProjects\Hypertuning\lightning_logs\version_2350\checkpoints\epoch=9-step=7180.ckpt"
# model_checkpoint = r'D:\Users\User\PycharmProjects\Hypertuning\lightning_logs\version_2376\checkpoints\epoch=49-step=35900.ckpt'
# model_checkpoint = r"D:\Users\User\PycharmProjects\Hypertuning\lightning_logs\version_3639\checkpoints\epoch=21-step=15796.ckpt"
model_checkpoint = r"D:\Users\User\PycharmProjects\Hypertuning\lightning_logs\version_3640\checkpoints\epoch=49-step=35900.ckpt"
output_path = r"D:\User\Desktop\musdb_normal\train\Aimee Norwich - Child\test.wav"
main(audio_path, model_checkpoint, output_path)
