import librosa
import numpy as np
from torchaudio.transforms import MFCC
import torch


def split_into_frames(
    audio: np.ndarray,
    frame_size: float,
    frame_shift: float,
    sample_rate: int,
):
    """Split audio into overlapping frames."""
    frame_len = int(frame_size * sample_rate)
    hop_len = int(frame_shift * sample_rate)
    return librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len).T


def extract_mfcc(audio: torch.Tensor, mfcc_transform: MFCC):
    """Extract MFCC features."""
    # return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    audio = audio.to(device)
    mfcc_transform = mfcc_transform.to(device)

    return mfcc_transform(audio).squeeze().T  # (frames, n_mfcc)
