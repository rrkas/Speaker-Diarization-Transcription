from datetime import datetime
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.parent
data_dir = root_dir / "data" / "librispeech"

os.makedirs(data_dir, exist_ok=True)


def run_cmd(cmd):
    start = datetime.now()
    print(cmd)
    status = os.system(cmd)
    end = datetime.now()
    print((end - start).total_seconds())
    return status


for url in [
    "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
]:
    status = run_cmd(f"wget -c --tries=0 --read-timeout=20 {url} -P {data_dir}")
    if status == 0:
        tar_fp = data_dir / url.split("/")[-1]
        run_cmd(f"tar -xzf {tar_fp} -C {data_dir}")


run_cmd(f"rm {data_dir}/*.tar.gz")
