import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2

import state
from recognition import draw_results_on_frame, extract_plates_from_frame, process_plates_batch, write_to_csv
from settings import CSV_WRITE_INTERVAL, DEDUP_SECONDS, PROCESS_EVERY_N_FRAMES

logger = logging.getLogger(__name__)


def now() -> float:
    return time.time()


def normalize_plate(text: str) -> str:
    return "".join(ch for ch in (text or "").upper().strip() if ch.isalnum())


def parse_camera_source(value: Any) -> Any:
    if isinstance(value, int):
        return value
    s = str(value).strip()
    return int(s) if s.isdigit() else s


@dataclass
class CameraPipeline:
    name: str
    source: Any
    running: bool = False
    cap: Optional[cv2.VideoCapture] = None
    thread: Optional[threading.Thread] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame_jpeg: Optional[bytes] = None
    last_plates: List[Dict[str, Any]] = None
    last_error: Optional[str] = None
    last_seen_plate_ts: Dict[str, float] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.last_plates = []
        self.last_seen_plate_ts = {}
        self.last_error = None

        parsed_source = parse_camera_source(self.source)
        self.cap = cv2.VideoCapture(parsed_source)
        self.source = parsed_source

        if (not self.cap or not self.cap.isOpened()) and isinstance(parsed_source, str):
            for fallback_idx in (1, 2, 3, 4):
                cap_try = cv2.VideoCapture(fallback_idx)
                if cap_try and cap_try.isOpened():
                    logger.warning(
                        f"{self.name}: URL camera không mở được ({parsed_source}), fallback sang camera index {fallback_idx}"
                    )
                    self.cap = cap_try
                    self.source = fallback_idx
                    break
                if cap_try:
                    cap_try.release()

        if not self.cap or not self.cap.isOpened():
            self.running = False
            self.last_error = (
                f"Không thể mở camera nguồn={parsed_source}. "
                "Nếu dùng DroidCam Client, nhập source là 1 hoặc 2 thay vì URL."
            )
            raise RuntimeError(self.last_error)

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
        except Exception:
            pass

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.cap = None
            self.thread = None

    def _loop(self):
        frame_skip = 0
        last_csv_write = now()

        while self.running:
            cap = self.cap
            if cap is None:
                time.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self.last_error = "Không thể đọc frame"
                time.sleep(0.1)
                continue

            frame_skip = (frame_skip + 1) % max(1, PROCESS_EVERY_N_FRAMES)
            plates: List[Dict[str, Any]] = []
            if frame_skip == 0:
                try:
                    plates_data = extract_plates_from_frame(frame)
                    plates = process_plates_batch(plates_data)
                    frame = draw_results_on_frame(frame, plates)
                    if plates and now() - last_csv_write >= CSV_WRITE_INTERVAL:
                        write_to_csv(plates)
                        last_csv_write = now()
                    self._smart_parking_update(plates)
                    self.last_error = None
                except Exception as exc:
                    self.last_error = str(exc)

            try:
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    with self.lock:
                        self.frame_jpeg = buffer.tobytes()
                        if plates:
                            self.last_plates = plates
            except Exception as exc:
                self.last_error = str(exc)

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def _smart_parking_update(self, plates: List[Dict[str, Any]]):
        now_ts = now()
        for p in plates:
            plate_raw = normalize_plate(p.get("text", ""))
            if not plate_raw or plate_raw == normalize_plate("Không nhận dạng"):
                continue

            last_ts = self.last_seen_plate_ts.get(plate_raw, 0.0)
            if now_ts - last_ts < DEDUP_SECONDS:
                continue
            self.last_seen_plate_ts[plate_raw] = now_ts

            if self.name == "entry":
                with state.parking_lock:
                    state.parking_inside[plate_raw] = now_ts
                state.push_event({"ts": now_ts, "type": "IN", "plate": plate_raw, "camera": "entry"})
            elif self.name == "exit":
                removed = False
                with state.parking_lock:
                    if plate_raw in state.parking_inside:
                        state.parking_inside.pop(plate_raw, None)
                        removed = True
                state.push_event(
                    {"ts": now_ts, "type": "OUT" if removed else "OUT_UNK", "plate": plate_raw, "camera": "exit"}
                )

    def mjpeg_stream(self):
        while True:
            if not self.running:
                time.sleep(0.1)
                continue
            with self.lock:
                frame = self.frame_jpeg
            if not frame:
                time.sleep(0.05)
                continue
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)
