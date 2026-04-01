import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from settings import (
    ASPECT_RATIO_THRESHOLD,
    CHECK_FOLDER,
    CSV_FILE,
    MIN_PLATE_HEIGHT,
    MIN_PLATE_WIDTH,
    MODEL_DIR,
    ONE_LINE_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)

model = YOLO("best.pt")
logger.info("Mô hình YOLO được tải thành công")

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_STR = "cuda:0" if CUDA_AVAILABLE else "cpu"


@torch.no_grad()
def load_trocr_model():
    model_name = "microsoft/trocr-base-printed"
    processor_path = os.path.join(MODEL_DIR, "processor")
    model_path = os.path.join(MODEL_DIR, "model")

    if os.path.exists(processor_path) and os.path.exists(model_path):
        processor = TrOCRProcessor.from_pretrained(processor_path)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
    else:
        processor = TrOCRProcessor.from_pretrained(model_name)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor.save_pretrained(processor_path)
        trocr_model.save_pretrained(model_path)

    trocr_model.eval()
    if CUDA_AVAILABLE:
        trocr_model = trocr_model.to(DEVICE_STR)
        logger.info("Mô hình TrOCR được tải trên GPU")
    else:
        logger.info("Mô hình TrOCR được tải trên CPU")
    return processor, trocr_model


processor, trocr_model = load_trocr_model()


def deskew_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        angle = 0
        if lines is not None:
            angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in lines[:, 0]]
            angles = [a for a in angles if abs(a) < 30]
            angle = np.median(angles) if angles else 0

        if abs(angle) > 2:
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image
    except Exception as exc:
        logger.warning(f"Lỗi khi chỉnh nghiêng ảnh: {exc}")
        return image


@torch.no_grad()
def ocr_with_trocr_batch(images, max_batch_size=8):
    try:
        results = []
        for i in range(0, len(images), max_batch_size):
            batch = images[i : i + max_batch_size]
            pil_images = [Image.fromarray(img).convert("RGB") for img in batch]
            inputs = processor(pil_images, return_tensors="pt")
            pixel_values = inputs.pixel_values
            if CUDA_AVAILABLE:
                pixel_values = pixel_values.to(DEVICE_STR)
            generated_ids = trocr_model.generate(pixel_values, max_length=20, num_beams=3, early_stopping=True)
            batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend([text.strip().upper() for text in batch_texts])
        return results
    except Exception as exc:
        logger.error(f"Lỗi OCR hàng loạt: {exc}")
        return ["Không nhận dạng" for _ in images]


def warmup_models():
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = model(dummy, conf=0.35, device=0 if CUDA_AVAILABLE else "cpu", verbose=False)
    except Exception as exc:
        logger.warning(f"Warmup YOLO lỗi: {exc}")
    try:
        dummy2 = np.zeros((64, 256, 3), dtype=np.uint8)
        _ = ocr_with_trocr_batch([dummy2])
    except Exception as exc:
        logger.warning(f"Warmup TrOCR lỗi: {exc}")


def split_plate_two_lines(plate_img):
    height = plate_img.shape[0]
    mid = height // 2
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    projection = np.sum(gray, axis=1)
    for i in range(mid - 10, mid + 10):
        if 0 < i < height - 1 and projection[i] < 0.7 * max(projection):
            mid = i
            break
    return plate_img[:mid, :], plate_img[mid:, :]


def process_two_line_plate(plate_img, filename, x1, y1):
    line1_img, line2_img = split_plate_two_lines(plate_img)
    base_name = f"{filename}_{x1}_{y1}"
    line1_path = os.path.join(CHECK_FOLDER, f"{base_name}_line1.jpg")
    line2_path = os.path.join(CHECK_FOLDER, f"{base_name}_line2.jpg")
    cv2.imwrite(line1_path, line1_img)
    cv2.imwrite(line2_path, line2_img)
    texts = ocr_with_trocr_batch([line1_img, line2_img])
    return {
        "text": f"{texts[0]} {texts[1]}".strip(),
        "processed_images": [
            f"/static/check/processed/{os.path.basename(line1_path)}",
            f"/static/check/processed/{os.path.basename(line2_path)}",
        ],
    }


def process_single_line_plate(plate_img, filename, x1, y1):
    save_path = os.path.join(CHECK_FOLDER, f"{filename}_{x1}_{y1}.jpg")
    cv2.imwrite(save_path, plate_img)
    text = ocr_with_trocr_batch([plate_img])[0]
    return {"text": text, "processed_images": [f"/static/check/processed/{os.path.basename(save_path)}"]}


def process_single_plate(plate_img, aspect_ratio, filename, x1, y1, x2, y2, conf):
    try:
        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
            plate_img = deskew_image(plate_img)
        h, w = plate_img.shape[:2]
        is_two_line = (w / h) <= ONE_LINE_RATIO_THRESHOLD
        result = process_two_line_plate(plate_img, filename, x1, y1) if is_two_line else process_single_line_plate(
            plate_img, filename, x1, y1
        )
        if not result["text"] or len(result["text"]) < 4:
            result["text"] = "Không nhận dạng"
        return {
            "text": result["text"],
            "confidence": float(conf),
            "coordinates": [x1, y1, x2, y2],
            "processed_images": result["processed_images"],
        }
    except Exception as exc:
        logger.error(f"Lỗi khi xử lý biển số tại [{x1}, {y1}, {x2}, {y2}]: {exc}")
        return {"text": "Không nhận dạng", "confidence": float(conf), "coordinates": [x1, y1, x2, y2], "processed_images": []}


def write_to_csv(plates):
    if not plates:
        return
    try:
        with open(CSV_FILE, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for plate in plates:
                writer.writerow([timestamp, plate["text"], plate["confidence"], plate["coordinates"]])
    except Exception as exc:
        logger.error(f"Lỗi ghi CSV: {exc}")


def extract_plates_from_frame(frame):
    results = model(frame, conf=0.35, device=0 if CUDA_AVAILABLE else "cpu", verbose=False)
    plates_data = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < MIN_PLATE_WIDTH or box_height < MIN_PLATE_HEIGHT:
                continue
            aspect_ratio = box_width / box_height
            plate_img = frame[y1:y2, x1:x2]
            plates_data.append((plate_img, aspect_ratio, "camera", x1, y1, x2, y2, box.conf))
    return plates_data


def process_plates_batch(plates_data: List[Tuple[Any, ...]]):
    if not plates_data:
        return []
    with ThreadPoolExecutor(max_workers=min(len(plates_data), 4)) as executor:
        return list(
            executor.map(
                lambda x: process_single_plate(*x),
                [(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]) for data in plates_data],
            )
        )


def draw_results_on_frame(frame, plates):
    for plate in plates:
        x1, y1, x2, y2 = plate["coordinates"]
        text = plate["text"]
        conf = plate["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_size, _ = cv2.getTextSize(f"{text} ({conf:.2f})", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, f"{text} ({conf:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame
