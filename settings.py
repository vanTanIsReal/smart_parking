import csv
import logging
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
CHECK_FOLDER = "static/check/processed"
MODEL_DIR = "models/trocr"

MIN_PLATE_WIDTH = 50
MIN_PLATE_HEIGHT = 20
ASPECT_RATIO_THRESHOLD = 1.3
ONE_LINE_RATIO_THRESHOLD = 2.5

CSV_FILE = "plate_recognition.csv"
CSV_WRITE_INTERVAL = 3

DEFAULT_EXIT_SOURCE_RAW = os.environ.get("EXIT_CAMERA_SOURCE", os.environ.get("EXIT_CAMERA_URL", "1"))
DEFAULT_ENTRY_INDEX = int(os.environ.get("ENTRY_CAMERA_INDEX", "0"))
FORCE_GPU = os.environ.get("FORCE_GPU", "1").strip().lower() not in ("0", "false", "no")
PROCESS_EVERY_N_FRAMES = int(os.environ.get("PROCESS_EVERY_N_FRAMES", "2"))
DEDUP_SECONDS = float(os.environ.get("DEDUP_SECONDS", "4.0"))


def ensure_runtime_files() -> None:
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CHECK_FOLDER, MODEL_DIR]:
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Thời gian", "Văn bản biển số", "Độ tin cậy", "Tọa độ"])
