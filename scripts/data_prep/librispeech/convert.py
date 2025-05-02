from pathlib import Path
import sys


root_dir = (Path(__file__).parent.parent.parent.parent).resolve()
sys.path.insert(0, str(root_dir))

from scripts.utils.convert import sox_convert_file  # noqa: E402

temp_dir = root_dir / "temp"
data_dir = root_dir / "data" / "amicorpus"
cache_dir = data_dir / "cache"
