import os
from pathlib import Path

DATA_DOWNLOAD_ROOT = Path(os.getenv("DATA_DOWNLOAD_ROOT") or "~/.cache")
