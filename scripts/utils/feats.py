import librosa
import numpy as np


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


def extract_mfcc(audio, sr, n_mfcc):
    """Extract MFCC features."""
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
