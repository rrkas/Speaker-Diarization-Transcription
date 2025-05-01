import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from scipy.io.wavfile import write
from pathlib import Path

rootdir = (Path(__file__).parent.parent.parent.parent).resolve()
print(rootdir)

cache_dir = (rootdir / "data" / "amicorpus").resolve()
print(cache_dir)

os.system(f"rm -rf {cache_dir / 'diarizers-community___ami'}")
os.system(f"rm -rf {cache_dir / '*.lock'}")

# validate existing data
data_validated = True
for split in ["train", "test", "validation"]:
    if not os.path.exists(cache_dir / split):
        data_validated = False
        break

    if len(list((cache_dir / split).glob("**/*.wav"))) != len(
        list((cache_dir / split).glob("**/*.json"))
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

    audio_filepath = cache_dir / split / data_name / fname
    info_filepath = cache_dir / split / data_name / fname.replace(".wav", ".json")

    os.makedirs(audio_filepath.parent, exist_ok=True)
    write(audio_filepath, sampling_rate, audio_array.astype(np.float32))

    with open(info_filepath, "w") as f:
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


os.system(f"rm -rf {cache_dir / 'diarizers-community___ami'}")
os.system(f"rm -rf {cache_dir / '*.lock'}")
