import os
from datasets import load_dataset
from pathlib import Path

rootdir = (Path(__file__).parent.parent.parent.parent).resolve()
cache_dir = (rootdir / "data" / "amicorpus").resolve()


os.makedirs(cache_dir, exist_ok=True)
ds = load_dataset("diarizers-community/ami", "ihm", cache_dir=cache_dir)
print(ds)

ds = load_dataset("diarizers-community/ami", "sdm", cache_dir=cache_dir)
print(ds)
