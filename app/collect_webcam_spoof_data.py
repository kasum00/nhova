import sys
from pathlib import Path
from datetime import datetime

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
YOLOV5_DIR = ROOT / "yolov5"

sys.path.append(str(YOLOV5_DIR))

FACE_MODEL_PATH = ROOT / "models" / "face_detector" / "yolov5n_face.pt"

REAL_DIR = ROOT / "datasets" / "webcam_spoof_cls" / "train" / "real"
FAKE_DIR = ROOT / "datasets" / "webcam_spoof_cls" / "train" / "fake"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACE_CONF = 0.5
FACE_CROP_MARGIN = 60

def load_face_detector():
    model = torch.hub.load(
        str(YOLOV5_DIR),
        "custom",
        path=str(FACE_MODEL_PATH),
        source="local"
    )
    model.conf = FACE_CONF
    model.to(DEVICE)
    model.eval()
    return model


def crop_face(frame, bbox, margin=60):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return frame[y1:y2, x1:x2]


def save_face(face, label):
    if face is None or face.size == 0:
        print("Không có face để lưu.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if label == "real":
        save_path = REAL_DIR / f"real_{timestamp}.jpg"
    else:
        save_path = FAKE_DIR / f"fake_{timestamp}.jpg"

    cv2.imwrite(str(save_path), face)
    print("Saved:", save_path)


def main():
    print("Loading face detector...")
    face_detector = load_face_detector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    print("Thu dữ liệu webcam anti-spoofing")
    print("r = lưu ảnh REAL")
    print("f = lưu ảnh FAKE")
    print("q = thoát")

    last_face = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = face_detector(frame)
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) > 0:
            detections = sorted(detections, key=lambda x: x[4], reverse=True)
            det = detections[0]

            x1, y1, x2, y2, conf, cls = det

            face = crop_face(
                frame,
                [x1, y1, x2, y2],
                margin=FACE_CROP_MARGIN
            )

            last_face = face.copy()

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"face {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            "r: save REAL | f: save FAKE | q: quit",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Collect Webcam Spoof Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            save_face(last_face, "real")

        if key == ord("f"):
            save_face(last_face, "fake")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()