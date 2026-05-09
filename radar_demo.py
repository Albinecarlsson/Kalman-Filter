from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kalman_filter.radar_demo import *


if __name__ == "__main__":
    main()
