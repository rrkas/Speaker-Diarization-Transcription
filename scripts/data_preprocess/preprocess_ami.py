import multiprocessing
import os
from pathlib import Path
import sys
import threading
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

root_dir = (Path(__file__).parent.parent.parent).resolve()
print(root_dir)
sys.path.insert(0, str(root_dir))

from scripts.utils.feats import extract_mfcc, split_into_frames  # noqa: E402


data_dir = root_dir / "data" / "amicorpus"
features_dir = root_dir / "features" / "amicorpus"


# Audio config
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025  # 25 ms
FRAME_SHIFT = 0.010  # 10 ms
N_MFCC = 13


batch_size = int(multiprocessing.cpu_count() // 2)
threads = {}


def process_file(fp: Path):
    try:
        split = fp.parent.parent.name
        data_name = fp.parent.name
        feats_fp = (
            features_dir / split / data_name / fp.name.replace(".wav", ".feats.npy")
        )

        if os.path.exists(feats_fp):
            try:
                np.load(feats_fp)
                threads.pop(fp, None)
                return
            except Exception as err:
                print(fp, feats_fp, err)

        audio, _ = librosa.load(str(fp), sr=SAMPLE_RATE)
        frames = split_into_frames(audio, FRAME_SIZE, FRAME_SHIFT, SAMPLE_RATE)

        mfccs = [extract_mfcc(frame, SAMPLE_RATE, N_MFCC) for frame in frames]
        mfcc_array = np.vstack([m for m in mfccs if m.shape[0] > 0])

        os.makedirs(feats_fp.parent, exist_ok=True)

        np.save(str(feats_fp), mfcc_array)
    except Exception as err:
        print(fp, err)
    finally:
        threads.pop(fp, None)


def waiter(threads: dict, wait_till):
    while len(threads) > wait_till:
        pass


for fp in tqdm(sorted(data_dir.glob("**/*.wav"))):
    t = threading.Thread(target=process_file, args=(fp,))
    threads[fp] = t
    t.start()

    # process_file(fp)
    # break

    waiter(threads, batch_size + 1)

waiter(threads, 0)
