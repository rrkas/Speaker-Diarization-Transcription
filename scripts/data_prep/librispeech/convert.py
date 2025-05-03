import os
import sys
from tqdm import tqdm
from pathlib import Path


root_dir = (Path(__file__).parent.parent.parent.parent).resolve()
sys.path.insert(0, str(root_dir))

from scripts.utils.convert import sox_convert_file  # noqa: E402

temp_dir = root_dir / "temp"
data_dir = root_dir / "data" / "LibriSpeech"
tgt_dir = root_dir / "data" / "LibriSpeech-formatted"


audio_fps = sorted(data_dir.glob("**/*.flac"))
audio_dirs = sorted(set(e.parent for e in audio_fps))

# print(*audio_fps[:10], sep="\n")


for audio_dir in tqdm(audio_dirs, desc="Audio_dirs"):
    curr_audio_fps = sorted(audio_dir.glob("*.flac"))
    tr_fp = list(audio_dir.glob("*.txt"))
    if len(tr_fp) == 0:
        print(tr_fp, "not found")
        continue

    tr_fp = tr_fp[0]
    with open(tr_fp, encoding="utf-8") as f:
        tr_lines = f.read().splitlines()

    if len(tr_lines) != len(curr_audio_fps):
        print(audio_dir, len(tr_lines), len(curr_audio_fps))

    split = audio_dir.parent.parent.name
    dir_id = f"{audio_dir.parent.name}__{audio_dir.name}"

    curr_tgt_dir = tgt_dir / split / dir_id
    print(curr_tgt_dir)

    tgt_tr_fp = curr_tgt_dir / "transcript.txt"
    os.makedirs(tgt_tr_fp.parent, exist_ok=True)

    tgt_tr_f = open(tgt_tr_fp, "w", encoding="utf-8")

    for idx, tr_line in enumerate(tr_lines):
        audio_name = tr_line.strip().split()[0]
        tr_line = " ".join(tr_line.strip().split()[1:])
        tgt_tr_f.write(tr_line + "\n")

        audio_fp = audio_dir / f"{audio_name}.flac"

        if not os.path.exists(audio_fp):
            print(audio_fp, "not found")
            continue

        fp_parts = str(audio_fp).split(os.sep)
        fname = audio_fp.name.split(".")[0].split("-")[-1].rjust(8, "0")
        tgt_fp = curr_tgt_dir / "train_audios" / f"{fname}.wav"

        os.makedirs(tgt_fp.parent, exist_ok=True)
        sox_convert_file(audio_fp, tgt_fp)

    tgt_tr_f.flush()
    tgt_tr_f.close()

    # break
