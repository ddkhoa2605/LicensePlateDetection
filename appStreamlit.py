"""
Streamlit app for Vietnamese license plate detection.

Pipeline:
Video upload -> YOLO tracking -> padded crop -> OCR preprocessing -> OCR voting
-> validation -> result display.
"""

from collections import Counter, defaultdict
from pathlib import Path
import re
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

try:
    from ultralytics import YOLO
    from fast_plate_ocr import LicensePlateRecognizer
except ImportError as exc:
    st.error(f"Missing libraries: {exc}")
    st.info("Install: pip install streamlit ultralytics opencv-python fast-plate-ocr onnxruntime")
    st.stop()


APP_DIR = Path(__file__).resolve().parent
ROOT_MODEL_DIR = APP_DIR / "models"

DEFAULT_CONFIG = {
    "model_path": str(APP_DIR / "DetectLisence_YOLO11.pt"),
    "ocr_model_path": str(APP_DIR / "cct_s_v1_vn.onnx"),
    "ocr_config_path": str(APP_DIR / "cct_s_v1_vn_plate_config.yaml"),
    "min_confidence": 0.35,
    "skip_frames": 5,
    "min_track_reads": 2,
    "max_stored_images": 200,
    "show_video": True,
    "use_tracking": True,
}


def first_existing_path(*candidates):
    """Return the first existing candidate, otherwise the first candidate."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return str(Path(candidates[0])) if candidates else ""


def build_default_config():
    config = DEFAULT_CONFIG.copy()
    config["model_path"] = first_existing_path(
        APP_DIR / "DetectLisence_YOLO11.pt",
        ROOT_MODEL_DIR / "DetectLisence_YOLO11.pt",
        APP_DIR / "bin" / "Debug" / "models" / "yolov11_plate.pt",
    )
    config["ocr_model_path"] = first_existing_path(
        APP_DIR / "cct_s_v1_vn.onnx",
        ROOT_MODEL_DIR / "cct_s_v1_vn.onnx",
        APP_DIR / "bin" / "Debug" / "models" / "PlateDetectionCustomVN.onnx",
    )
    config["ocr_config_path"] = first_existing_path(
        APP_DIR / "cct_s_v1_vn_plate_config.yaml",
        ROOT_MODEL_DIR / "cct_s_v1_vn_plate_config.yaml",
        APP_DIR / "bin" / "Debug" / "models" / "VNCustomConfig.yaml",
    )
    return config


class SimplePlateDetector:
    """YOLO detector with OCR, tracking buffers, and plate voting."""

    PLATE_4W = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$")
    PLATE_2LINE = re.compile(r"^[0-9]{2}[A-Z][0-9]{1,2}[0-9]{4,5}$")
    MIN_OCR_TEXT_LENGTH = 3

    def __init__(
        self,
        model_path,
        ocr_model_path=None,
        ocr_config_path=None,
        min_track_reads=2,
        use_tracking=True,
    ):
        self.model = YOLO(model_path)
        self.min_track_reads = max(int(min_track_reads), 1)
        self.use_tracking = use_tracking
        self.track_buffer = defaultdict(list)
        self.track_best_image = {}
        self.synthetic_track_id = 0
        self.tracking_fallback_frames = 0

        if ocr_model_path and ocr_config_path:
            self.alpr = LicensePlateRecognizer(
                onnx_model_path=str(ocr_model_path),
                plate_config_path=str(ocr_config_path),
                device="cuda" if self._check_cuda() else "cpu",
            )
            self.ocr_enabled = True
        else:
            self.alpr = None
            self.ocr_enabled = False

    @staticmethod
    def _check_cuda():
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def clean_text(value):
        if not value:
            return ""
        value = value.upper().replace("_", "").replace(" ", "").replace("-", "").strip()
        return re.sub(r"[^A-Z0-9]", "", value)

    @staticmethod
    def _normalize_char_confs(text, confs):
        if confs is None:
            return [0.0] * len(text)

        arr = np.asarray(confs, dtype=float).flatten()
        if len(arr) == 0:
            return [0.0] * len(text)

        values = arr[: len(text)].tolist()
        if len(values) < len(text):
            values.extend([float(np.mean(arr))] * (len(text) - len(values)))
        return [float(max(0.0, min(1.0, value))) for value in values]

    def validate_plate(self, plate_text):
        clean = self.clean_text(plate_text)
        if len(clean) < 7:
            return False
        return bool(self.PLATE_4W.match(clean) or self.PLATE_2LINE.match(clean))

    def preprocess_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return None

        if len(plate_img.shape) == 2:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

        h, w = plate_img.shape[:2]
        if h < 20 or w < 60:
            return None

        if h < 50 or w < 100:
            scale = max(64 / h, 200 / w)
            plate_img = cv2.resize(
                plate_img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 15)

    def run_ocr(self, plate_img):
        if not self.ocr_enabled:
            return None, 0.0, []

        processed = self.preprocess_plate(plate_img)
        if processed is None:
            return None, 0.0, []

        try:
            texts, confs = self.alpr.run(processed, return_confidence=True)
            raw_plate = texts[0] if texts else ""
            plate = self.clean_text(raw_plate)
            if not plate:
                return None, 0.0, []

            first_conf = confs[0] if confs is not None and len(confs) > 0 else None
            char_confs = self._normalize_char_confs(plate, first_conf)
            avg_conf = float(np.mean(char_confs)) if char_confs else 0.0
            return plate, avg_conf, char_confs
        except Exception:
            return None, 0.0, []

    @staticmethod
    def crop_with_padding(frame, bbox, padding_ratio=0.10):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = max(int(box_w * padding_ratio), 4)
        pad_y = max(int(box_h * padding_ratio), 4)

        x1_c = max(0, x1 - pad_x)
        y1_c = max(0, y1 - pad_y)
        x2_c = min(w, x2 + pad_x)
        y2_c = min(h, y2 + pad_y)
        return frame[y1_c:y2_c, x1_c:x2_c]

    def vote_plate(self, track_id, min_reads=None):
        min_reads = min_reads or self.min_track_reads
        reads = self.track_buffer.get(track_id, [])
        if len(reads) < min_reads:
            return None, 0.0

        candidates = [
            (text, char_confs)
            for text, char_confs, _frame_num, _det_conf in reads
            if text and len(text) >= self.MIN_OCR_TEXT_LENGTH
        ]
        if not candidates:
            return None, 0.0

        target_len = Counter(len(text) for text, _ in candidates).most_common(1)[0][0]
        same_len = [(text, confs) for text, confs in candidates if len(text) == target_len]
        scores = [defaultdict(float) for _ in range(target_len)]

        for text, char_confs in same_len:
            normalized = self._normalize_char_confs(text, char_confs)
            for idx, (char, conf) in enumerate(zip(text, normalized)):
                scores[idx][char] += max(conf, 0.01)

        voted_chars = []
        position_conf = []
        for score in scores:
            if not score:
                continue
            total = sum(score.values())
            winner, winner_score = max(score.items(), key=lambda item: item[1])
            voted_chars.append(winner)
            position_conf.append(winner_score / total if total else 0.0)

        plate = "".join(voted_chars)
        avg_conf = float(np.mean(position_conf)) if position_conf else 0.0
        return plate, avg_conf

    def _best_read_for_track(self, track_id):
        reads = self.track_buffer.get(track_id, [])
        if not reads:
            return None
        return max(reads, key=lambda item: (float(np.mean(item[1])) if item[1] else 0.0, item[3]))

    def _build_final_detection(self, track_id, image_budget_remaining):
        voted_plate, voted_conf = self.vote_plate(track_id)

        best_read = self._best_read_for_track(track_id)
        if best_read is None:
            return None

        best_text, best_char_confs, frame_num, det_conf = best_read
        if not voted_plate:
            fallback_text = self.clean_text(best_text)
            if len(fallback_text) < self.MIN_OCR_TEXT_LENGTH:
                return None
            voted_plate = fallback_text
            voted_conf = float(np.mean(best_char_confs)) if best_char_confs else 0.0

        best_image = self.track_best_image.get(track_id)
        stored_image = best_image.copy() if best_image is not None and image_budget_remaining > 0 else None

        return {
            "track_id": track_id,
            "plate": voted_plate,
            "confidence": det_conf,
            "ocr_confidence": voted_conf,
            "is_valid": self.validate_plate(voted_plate),
            "frame": frame_num,
            "timestamp": None,
            "image": stored_image,
            "read_count": len(self.track_buffer.get(track_id, [])),
        }

    def finalize_lost_tracks(self, active_ids, image_budget_remaining):
        finalized = []
        active_set = set(active_ids)
        lost_ids = [track_id for track_id in list(self.track_buffer.keys()) if track_id not in active_set]

        for track_id in lost_ids:
            detection = self._build_final_detection(track_id, image_budget_remaining - len(finalized))
            if detection:
                finalized.append(detection)
            self.track_buffer.pop(track_id, None)
            self.track_best_image.pop(track_id, None)

        return finalized

    def finalize_all_tracks(self, image_budget_remaining):
        finalized = []
        for track_id in list(self.track_buffer.keys()):
            detection = self._build_final_detection(track_id, image_budget_remaining - len(finalized))
            if detection:
                finalized.append(detection)
            self.track_buffer.pop(track_id, None)
            self.track_best_image.pop(track_id, None)
        return finalized

    def _run_yolo(self, frame, min_confidence):
        if self.use_tracking:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    conf=min_confidence,
                )
                return results, True
            except Exception:
                self.tracking_fallback_frames += 1

        return self.model(frame, verbose=False, conf=min_confidence), False

    def process_frame(self, frame, frame_num, min_confidence=0.35, image_budget_remaining=0):
        results, tracking_active = self._run_yolo(frame, min_confidence)
        annotated_frame = frame.copy()
        active_ids = []
        immediate_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf < min_confidence:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = self.crop_with_padding(frame, (x1, y1, x2, y2))
                if cropped.size == 0:
                    continue

                if tracking_active and box.id is not None:
                    track_id = int(box.id[0])
                else:
                    self.synthetic_track_id += 1
                    track_id = -self.synthetic_track_id

                if tracking_active:
                    active_ids.append(track_id)
                plate_text, ocr_conf, char_confs = self.run_ocr(cropped)
                display_text = plate_text or "UNKNOWN"
                is_valid = self.validate_plate(display_text)

                if plate_text and tracking_active:
                    self.track_buffer[track_id].append((plate_text, char_confs, frame_num, conf))
                    current_best = self.track_best_image.get(track_id)
                    if current_best is None or ocr_conf >= self._best_image_conf(track_id):
                        self.track_best_image[track_id] = cropped.copy()

                voted_text, voted_conf = self.vote_plate(track_id) if tracking_active else (None, 0.0)
                if voted_text:
                    display_text = voted_text
                    ocr_conf = voted_conf
                    is_valid = self.validate_plate(voted_text)

                color = (0, 180, 0) if is_valid else (0, 0, 220)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track_id}: {display_text} ({ocr_conf:.2f})"
                self._draw_label(annotated_frame, label, x1, y1, color)

                should_emit_immediate = not tracking_active and (plate_text or not self.ocr_enabled)
                if should_emit_immediate:
                    immediate_detections.append(
                        {
                            "track_id": track_id,
                            "plate": plate_text or "UNKNOWN",
                            "confidence": conf,
                            "ocr_confidence": ocr_conf,
                            "is_valid": is_valid,
                            "frame": frame_num,
                            "timestamp": None,
                            "image": cropped.copy() if image_budget_remaining > len(immediate_detections) else None,
                            "read_count": 1,
                        }
                    )

        finalized = self.finalize_lost_tracks(active_ids, image_budget_remaining) if tracking_active else []
        return annotated_frame, immediate_detections + finalized

    def _best_image_conf(self, track_id):
        best_read = self._best_read_for_track(track_id)
        if best_read is None or not best_read[1]:
            return 0.0
        return float(np.mean(best_read[1]))

    @staticmethod
    def _draw_label(frame, label, x, y, color):
        label_y = max(y, 20)
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, label_y - label_h - 8), (x + label_w + 4, label_y), color, -1)
        cv2.putText(
            frame,
            label,
            (x + 2, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def add_timestamps(detections, fps):
    for detection in detections:
        detection["timestamp"] = detection["frame"] / fps if fps else 0.0
    return detections


def merge_by_best_confidence(detections):
    unique = {}
    for detection in detections:
        plate = detection["plate"]
        if plate not in unique or detection["ocr_confidence"] > unique[plate]["ocr_confidence"]:
            unique[plate] = detection
    return unique


def render_detection_image(detection):
    image = detection.get("image")
    if image is None:
        st.caption("Crop image was not stored because the image memory limit was reached.")
        return
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=detection["plate"], use_container_width=True)


def render_results(all_detections):
    st.header("Detection Results")
    valid_plates = [det for det in all_detections if det["is_valid"]]
    unique_valid_plates = merge_by_best_confidence(valid_plates)

    tab1, tab2, tab3 = st.tabs(["Summary", "All Detections", "Valid Plates Only"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Detections", len(all_detections))
        col2.metric("Valid Plates", len(valid_plates))
        col3.metric("Unique Plates", len(unique_valid_plates))
        accuracy = (len(valid_plates) / len(all_detections) * 100) if all_detections else 0.0
        col4.metric("Valid Rate", f"{accuracy:.1f}%")

        if unique_valid_plates:
            st.subheader("Plate Frequency")
            counts = Counter(det["plate"] for det in valid_plates)
            for plate, count in counts.most_common(10):
                st.write(f"**{plate}**: {count} time(s)")

    with tab2:
        st.subheader("All Detected Plates")
        for idx, detection in enumerate(all_detections):
            status = "Valid" if detection["is_valid"] else "Invalid"
            with st.expander(
                f"Detection #{idx + 1} - {detection['plate']} - {status} "
                f"(Track {detection['track_id']}, Frame {detection['frame']})"
            ):
                col1, col2 = st.columns([1, 2])
                with col1:
                    render_detection_image(detection)
                with col2:
                    st.write(f"**Plate:** {detection['plate']}")
                    st.write(f"**Status:** {status}")
                    st.write(f"**Track ID:** {detection['track_id']}")
                    st.write(f"**Frame:** {detection['frame']}")
                    st.write(f"**Time:** {detection['timestamp']:.2f}s")
                    st.write(f"**YOLO Confidence:** {detection['confidence']:.3f}")
                    st.write(f"**OCR/Vote Confidence:** {detection['ocr_confidence']:.3f}")
                    st.write(f"**OCR Reads:** {detection['read_count']}")

    with tab3:
        st.subheader("Valid Plates Only (Best Confidence)")
        if not unique_valid_plates:
            st.warning("No valid plates detected")
            return

        st.info(
            f"Found {len(unique_valid_plates)} unique valid plates "
            f"from {len(valid_plates)} valid detections."
        )

        for plate, detection in unique_valid_plates.items():
            with st.expander(f"{plate} (Best frame: {detection['frame']})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    render_detection_image(detection)
                with col2:
                    st.write(f"**Plate:** {detection['plate']}")
                    st.write(f"**Frame:** {detection['frame']}")
                    st.write(f"**Time:** {detection['timestamp']:.2f}s")
                    st.write(f"**YOLO Confidence:** {detection['confidence']:.3f}")
                    st.write(f"**OCR/Vote Confidence:** {detection['ocr_confidence']:.3f}")
                    st.write(f"**OCR Reads:** {detection['read_count']}")


def main():
    st.set_page_config(page_title="License Plate Detection", page_icon=":car:", layout="wide")
    st.title("License Plate Detection System")
    st.markdown("Upload a video to detect and recognize Vietnamese license plates.")

    config = build_default_config()

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file for plate detection.",
    )

    if uploaded_file is None:
        st.info("Please upload a video file to start detection.")
        return

    model_path = Path(config["model_path"])
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    ocr_model = config["ocr_model_path"] if config["ocr_model_path"] and Path(config["ocr_model_path"]).exists() else None
    ocr_config = config["ocr_config_path"] if config["ocr_config_path"] and Path(config["ocr_config_path"]).exists() else None

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()
    video_path = temp_video.name

    try:
        with st.spinner("Loading models..."):
            detector = SimplePlateDetector(
                model_path=str(model_path),
                ocr_model_path=ocr_model,
                ocr_config_path=ocr_config,
                min_track_reads=config["min_track_reads"],
                use_tracking=config["use_tracking"],
            )
            if detector.ocr_enabled:
                st.success("YOLO and OCR models loaded successfully.")
            else:
                st.warning("YOLO loaded. OCR disabled because model/config was not found.")

        if not st.button("Start Detection", type="primary"):
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Video Processing")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
        with col2:
            st.subheader("Detected Plates")
            live_text = st.empty()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

        all_detections = []
        stored_image_count = 0
        frame_count = 0
        processed_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % config["skip_frames"] == 0:
                    image_budget = max(config["max_stored_images"] - stored_image_count, 0)
                    annotated_frame, detections = detector.process_frame(
                        frame,
                        frame_num=frame_count,
                        min_confidence=config["min_confidence"],
                        image_budget_remaining=image_budget,
                    )
                    processed_count += 1
                    add_timestamps(detections, fps)
                    stored_image_count += sum(1 for detection in detections if detection.get("image") is not None)
                    all_detections.extend(detections)

                    if config["show_video"]:
                        video_placeholder.image(
                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True,
                        )

                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(
                    f"Frame: {frame_count}/{total_frames} | "
                    f"Processed: {processed_count} | "
                    f"Final detections: {len(all_detections)}"
                )

                recent_valid = [det["plate"] for det in all_detections[-5:] if det["is_valid"]]
                live_text.write("Recent valid plates: " + (", ".join(recent_valid) if recent_valid else "None yet"))

            remaining_budget = max(config["max_stored_images"] - stored_image_count, 0)
            final_detections = detector.finalize_all_tracks(remaining_budget)
            add_timestamps(final_detections, fps)
            stored_image_count += sum(1 for detection in final_detections if detection.get("image") is not None)
            all_detections.extend(final_detections)
            cap.release()

            elapsed_time = time.time() - start_time
            st.success(f"Processing complete in {elapsed_time:.1f}s.")
            st.info(
                f"Summary: {frame_count} frames read, {processed_count} frames processed, "
                f"{len(all_detections)} finalized plate detection(s)."
            )

            if all_detections:
                render_results(all_detections)
            else:
                st.warning("No plates detected in the video.")

        except Exception as exc:
            cap.release()
            st.error(f"Error during processing: {exc}")

    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>License Plate Detection System | YOLO + Fast Plate OCR</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
