import os
from pathlib import Path


def sox_convert_file(src: Path, dst: Path, channels=1, bitrate=16, sample_rate=16000):
    os.system(
        f'sox -V0 "{src}" -b {bitrate} -c {channels} -r {sample_rate} -e signed "{dst}"'
    )
