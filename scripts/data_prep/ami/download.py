import os
import json
import sys
import uuid
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from scipy.io.wavfile import write
from pathlib import Path

root_dir = (Path(__file__).parent.parent.parent.parent).resolve()
sys.path.insert(0, str(root_dir))

from scripts.utils.convert import sox_convert_file  # noqa: E402

temp_dir = root_dir / "temp"
data_dir = root_dir / "data" / "amicorpus"
cache_dir = data_dir / "cache"

# os.system(f"rm -rf {cache_dir}")

# validate existing data
data_validated = True
for split in ["train", "test", "validation"]:
    if not os.path.exists(data_dir / split):
        data_validated = False
        break

    if len(list((data_dir / split).glob("**/*.wav"))) != len(
        list((data_dir / split).glob("**/*.json"))
    ):
        data_validated = False
        break


if data_validated:
    print("data already downloaded & validated")
    exit()


def save_row(split: str, data_name: str, row: dict):
    fname = os.path.basename(row["audio"]["path"])
    audio_array = row["audio"]["array"]
    sampling_rate = row["audio"]["sampling_rate"]

    recs = [
        {
            "start": start,
            "end": end,
            "speaker": speaker,
        }
        for start, end, speaker in zip(
            row["timestamps_start"], row["timestamps_end"], row["speakers"]
        )
    ]

    audio_fp = data_dir / split / data_name / fname
    temp_fp = temp_dir / f"{uuid.uuid4().hex}.wav"
    info_fp = data_dir / split / data_name / fname.replace(".wav", ".json")

    os.makedirs(temp_fp.parent, exist_ok=True)
    os.makedirs(audio_fp.parent, exist_ok=True)

    write(temp_fp, sampling_rate, audio_array.astype(np.float32))
    sox_convert_file(temp_fp, audio_fp)

    os.remove(temp_fp)

    with open(info_fp, "w") as f:
        json.dump(recs, f, indent=4)


# ihm: Individual Headset Microphone
# sdm: Single Distant Microphone
for data_name in ["ihm", "sdm"]:
    ds = load_dataset("diarizers-community/ami", data_name, cache_dir=str(cache_dir))
    print(ds)

    for split in ds:
        for row in tqdm(ds[split], desc=split):
            save_row(split, data_name, row)

    del ds


os.system(f"rm -rf {cache_dir}")
