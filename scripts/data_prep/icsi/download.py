from datetime import datetime
import os
from pathlib import Path

from tqdm import tqdm

root_dir = Path(__file__).parent.parent.parent.parent
data_dir = root_dir / "data" / "icsi-corpus"
print(root_dir)
os.makedirs(data_dir, exist_ok=True)


def run_cmd(cmd):
    start = datetime.now()
    print(cmd)
    status = os.system(cmd)
    end = datetime.now()
    print((end - start).total_seconds())
    return status


with open("./urls.txt") as f:
    urls = f.read().splitlines()

for url in tqdm(sorted(urls)):
    if "/NXT/" in url:
        # folder_name = url.split("/")[-1].split(".")[0]
        folder_name = "NXT/"
    elif "/SPH/" in url:
        # folder_name = url.split("/")[-2]
        folder_name = "SPH/" + url.split("/")[-2]

    tgt_dir = data_dir / folder_name.strip("/")
    run_cmd(f"wget -c --tries=0 --read-timeout=20 {url} -P {tgt_dir}")


# for url in ["https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip"]:
#     run_cmd(f"")
