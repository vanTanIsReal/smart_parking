from flask import Flask, Response, jsonify, render_template, request
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
import os

import state
from pipeline import CameraPipeline, parse_camera_source
from recognition import (
    DEVICE_STR,
    CUDA_AVAILABLE,
    extract_plates_from_frame,
    model,
    process_plates_batch,
    warmup_models,
    write_to_csv,
)
from settings import (
    DEFAULT_ENTRY_INDEX,
    DEFAULT_EXIT_SOURCE_RAW,
    FORCE_GPU,
    MIN_PLATE_HEIGHT,
    MIN_PLATE_WIDTH,
    RESULT_FOLDER,
    UPLOAD_FOLDER,
    ensure_runtime_files,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
ensure_runtime_files()

if FORCE_GPU and not CUDA_AVAILABLE:
    raise RuntimeError(
        "FORCE_GPU=1 nhưng không phát hiện CUDA. "
        "Hãy cài đúng PyTorch CUDA + driver NVIDIA, hoặc set FORCE_GPU=0 để chạy CPU."
    )
logger.info(f"Device inference: {DEVICE_STR} (cuda_available={CUDA_AVAILABLE})")
warmup_models()

state.set_exit_camera_source(DEFAULT_EXIT_SOURCE_RAW)
entry_pipeline = CameraPipeline(name="entry", source=DEFAULT_ENTRY_INDEX)
exit_pipeline = CameraPipeline(name="exit", source=parse_camera_source(DEFAULT_EXIT_SOURCE_RAW))


@app.route('/')
def home():
    source = state.get_exit_camera_source()
    return render_template("index.html", exit_camera_source=source, force_gpu=FORCE_GPU, device=DEVICE_STR)

@app.route('/upload', methods=['POST'])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Không có tệp được tải lên"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Không có tệp được chọn"}), 400

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({"error": "Tệp ảnh không hợp lệ"}), 400

        results = model(img)
        result_img = results[0].plot()
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result_img)

        plates_data = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width < MIN_PLATE_WIDTH or box_height < MIN_PLATE_HEIGHT:
                    continue
                aspect_ratio = box_width / box_height
                plate_img = img[y1:y2, x1:x2]
                plates_data.append((plate_img, aspect_ratio, filename, x1, y1, x2, y2, box.conf))

        plates = process_plates_batch(plates_data)
        if plates:
            write_to_csv(plates)

        return jsonify(
            {
                "status": "thành công",
                "original_image": f"/static/uploads/{filename}",
                "result_image": f"/static/results/{filename}",
                "plates": plates,
                "plate_count": len(plates),
            }
        )
    except Exception as exc:
        logger.error(f"Lỗi xử lý: {exc}")
        return jsonify({"error": str(exc)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        raw_source = str(data.get("exit_camera_source", data.get("exit_camera_url", ""))).strip()
        if not raw_source:
            return jsonify({"error": "exit_camera_source trống"}), 400

        state.set_exit_camera_source(raw_source)
        exit_pipeline.source = parse_camera_source(raw_source)
        return jsonify({"status": "ok", "exit_camera_source": raw_source})

    source = state.get_exit_camera_source()
    return jsonify({"exit_camera_source": source, "force_gpu": FORCE_GPU, "device": DEVICE_STR})


@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.get_json(silent=True) or {}
    which = (data.get("which") or "").strip().lower()
    if which not in ("entry", "exit", "both"):
        return jsonify({"error": "which phải là entry|exit|both"}), 400

    try:
        if which in ("entry", "both"):
            entry_pipeline.start()
        if which in ("exit", "both"):
            source = state.get_exit_camera_source()
            exit_pipeline.source = parse_camera_source(source)
            exit_pipeline.start()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def api_stop():
    data = request.get_json(silent=True) or {}
    which = (data.get("which") or "").strip().lower()
    if which not in ("entry", "exit", "both"):
        return jsonify({"error": "which phải là entry|exit|both"}), 400

    if which in ("entry", "both"):
        entry_pipeline.stop()
    if which in ("exit", "both"):
        exit_pipeline.stop()
    return jsonify({"status": "ok"})


@app.route('/stream/entry')
def stream_entry():
    return Response(entry_pipeline.mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/exit')
def stream_exit():
    return Response(exit_pipeline.mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with state.events_lock:
        events = list(state.recent_events[:50])
    with state.parking_lock:
        inside = sorted(list(state.parking_inside.keys()))

    def _fmt(ev):
        return {
            **ev,
            "ts": datetime.fromtimestamp(ev["ts"]).strftime("%Y-%m-%d %H:%M:%S"),
        }

    return jsonify(
        {
            "device": DEVICE_STR,
            "force_gpu": FORCE_GPU,
            "entry": {"running": entry_pipeline.running, "error": entry_pipeline.last_error},
            "exit": {"running": exit_pipeline.running, "error": exit_pipeline.last_error},
            "inside": inside,
            "inside_count": len(inside),
            "events": [_fmt(e) for e in events],
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)