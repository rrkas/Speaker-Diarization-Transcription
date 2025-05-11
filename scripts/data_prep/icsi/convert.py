from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import sys
import uuid

from tqdm import tqdm

root_dir = Path(__file__).parent.parent.parent.parent.resolve()

temp_dir = root_dir / "temp"
org_data_dir = root_dir / "data" / "icsi-corpus"
rev_data_dir = root_dir / "data" / "icsi-corpus_rev"

sph2pipe_path = root_dir / "sph2pipe_v2.5/sph2pipe"

if not os.path.exists(org_data_dir):
    print("no data to process!")
    exit()

sph_files = sorted(org_data_dir.glob("**/*.sph"))
wav_files = sorted(org_data_dir.glob("**/*.wav"))

print("sph files found:", len(sph_files))
print("wav files found:", len(wav_files))

if len(sph_files) == 0 and len(wav_files) == 0:
    print("no files found to process!")


os.makedirs(rev_data_dir, exist_ok=True)


def run_cmd(cmd):
    start = datetime.now()
    # print(cmd)
    status = os.system(cmd)
    end = datetime.now()
    # print((end - start).total_seconds())
    return status


def sox_convert_file(src: Path, dst: Path, channels=1, bitrate=16, sample_rate=16000):
    run_cmd(
        f'sox -V0 "{src}" -b {bitrate} -c {channels} -r {sample_rate} -e signed "{dst}"'
    )


def process_file(fp: Path):
    if fp.name.endswith(".sph"):
        rel_fp = str(fp.relative_to(org_data_dir))
        tgt_fp = rev_data_dir / rel_fp.replace(".sph", ".wav")
        temp_fp = temp_dir / f"{uuid.uuid4().hex}.wav"
        os.makedirs(temp_fp.parent, exist_ok=True)
        run_cmd(f"{sph2pipe_path} -f rif {fp} {temp_fp}")
        os.makedirs(tgt_fp.parent, exist_ok=True)
        sox_convert_file(temp_fp, tgt_fp)
        os.remove(temp_fp)
    elif fp.name.endswith(".wav"):
        rel_fp = str(fp.relative_to(org_data_dir))
        tgt_fp = rev_data_dir / rel_fp
        os.makedirs(tgt_fp.parent, exist_ok=True)
        sox_convert_file(fp, tgt_fp)


def batchify(_lst: list, batch_size: int):
    batch_size = int(batch_size)

    for i in range(0, len(_lst), batch_size):
        yield _lst[i : i + batch_size]


batch_size = multiprocessing.cpu_count()

for batch in tqdm(list(batchify(sph_files, batch_size)), desc="sph"):
    with multiprocessing.Pool(len(batch)) as pool:
        pool.map(process_file, batch)


for fp in tqdm(list(batchify(wav_files, batch_size)), desc="wav"):
    with multiprocessing.Pool(len(batch)) as pool:
        pool.map(process_file, batch)


#  ./sph2pipe -f rif /home/alienwarenew/rohnak/mtech_diarz/data/icsi-corpus/SPH/Bdb001/chan1.sph ./sample.wav
