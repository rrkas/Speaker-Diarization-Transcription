"""
For Diarization:

- You need short, fixed-length overlapping frames of audio to:
    - Extract MFCCs or spectrograms for each frame.
    - Feed into speaker embedding models (e.g., Siamese/Triplet).
    - Use these embeddings for clustering and speaker labeling.

- That's exactly what the provided code does:
    - Converts audio to mono, 16kHz.
    - Splits into small overlapping frames (25ms with 10ms hop).
    - Extracts MFCC features per frame → used for speaker recognition and clustering.
"""

import multiprocessing
import os
from pathlib import Path
import sys
import threading
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
from torchaudio.transforms import MFCC

import warnings

warnings.filterwarnings("ignore")

root_dir = (Path(__file__).parent.parent.parent).resolve()
print(root_dir)
sys.path.insert(0, str(root_dir))

from scripts.utils.feats import extract_mfcc  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_dir = root_dir / "data" / "amicorpus"
features_dir = root_dir / "features" / "amicorpus"


# Audio config
SAMPLE_RATE = 16000
N_MFCC = 13

mfcc_transform = MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 40,
    },
).to(device)

batch_size = int(multiprocessing.cpu_count() // 2)
threads = {}


def process_file(fp: Path):
    try:
        split = fp.parent.parent.name
        data_name = fp.parent.name
        feats_fp = (
            features_dir / split / data_name / fp.name.replace(".wav", ".mfcc.npy")
        )

        if os.path.exists(feats_fp):
            try:
                print("checking", feats_fp)
                np.load(feats_fp)
                threads.pop(fp, None)
                return
            except Exception as err:
                print(fp, feats_fp, err)

        # print("loading audio")
        audio, sr = sf.read(str(fp))
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        audio = torch.tensor(audio).float().to(device)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # make it (1, N)

        # print(audio.shape)

        mfcc_feat = extract_mfcc(audio, mfcc_transform)

        os.makedirs(feats_fp.parent, exist_ok=True)

        np.save(str(feats_fp), mfcc_feat.cpu().numpy())
    except Exception as err:
        print(fp, err)
        raise err
    finally:
        threads.pop(fp, None)


def waiter(threads: dict, wait_till):
    while len(threads) > wait_till:
        pass


for fp in tqdm(sorted(data_dir.glob("**/*.wav"))):
    # print(fp)
    t = threading.Thread(target=process_file, args=(fp,))
    threads[fp] = t
    t.start()

    # process_file(fp)
    # break

    waiter(threads, batch_size + 1)

waiter(threads, 0)
