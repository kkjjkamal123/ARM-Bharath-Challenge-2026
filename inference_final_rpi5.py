import cv2
import numpy as np
import time
import os
import threading
import ai_edge_litert.interpreter as tflite
from picamera2 import Picamera2

# ================================================================
#  Bharat AI-SoC | Road Anomaly Detection — OBB + Obstacles
#  Model 1: YOLOv8n-obb INT8 TFLite  → potholes (rotated boxes)
#  Model 2: YOLOv8n    INT8 TFLite   → humans, dogs, obstacles
#  Target : RPi 5 ARM Cortex-A76 via XNNPACK
# ================================================================

# ── CONFIG ───────────────────────────────────────────────────────
OBB_MODEL_PATH  = "pothole_obb_int8.tflite"   # your trained OBB model
DET_MODEL_PATH  = "yolov8n_coco_int8.tflite"  # COCO obstacle model (see setup)
INPUT_SIZE      = 320
NUM_THREADS     = 4
CAMERA_W        = 640
CAMERA_H        = 480
DISPLAY_EVERY   = 2
LOG_FILE        = "road_anomalies_log.csv"
FILL_ALPHA      = 0.25

# Confidence thresholds — obstacles need higher thresh to avoid false positives
OBB_CONF        = 0.45
DET_CONF        = 0.5

# COCO classes we care about as road obstacles
# Full COCO list index: person=0, bicycle=1, car=2, dog=16, cat=15,
# motorcycle=3, bus=5, truck=7, traffic light=9, stop sign=11
OBSTACLE_CLASS_IDS = {
    0:  ("Person",         (0,   165, 255)),   # orange
    1:  ("Bicycle",        (255, 165,   0)),   # blue-ish
    2:  ("Car",            (255,   0, 255)),   # magenta
    3:  ("Motorcycle",     (255, 200,   0)),
    5:  ("Bus",            (200,   0, 255)),
    7:  ("Truck",          (180,   0, 255)),
    9:  ("Traffic Light",  (0,   255, 255)),   # yellow
    11: ("Stop Sign",      (0,     0, 255)),   # red
    15: ("Cat",            (255, 100, 100)),
    16: ("Dog",            (100, 100, 255)),   # pink-ish
}

# Pothole colour
POTHOLE_COLOUR = (0, 255, 0)   # green


# ── CAMERA ───────────────────────────────────────────────────────
class VideoStream:
    def __init__(self, width=640, height=480):
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        self.frame   = None
        self.stopped = False
        self.lock    = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            frame = self.cam.capture_array()
            with self.lock:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cam.stop()


# ── LOAD MODELS ──────────────────────────────────────────────────
def load_model(path, num_threads=4):
    interp = tflite.Interpreter(model_path=path, num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = sorted(interp.get_output_details(), key=lambda x: x["index"])
    return interp, inp, out

print("Loading OBB pothole model...")
obb_interp, obb_inp, obb_out = load_model(OBB_MODEL_PATH)
OBB_FLOAT = obb_inp["dtype"] == np.float32
print(f"  OBB input : {obb_inp['shape']}  float={OBB_FLOAT}")
print(f"  OBB output: {obb_out[0]['shape']}")

print("Loading COCO obstacle model...")
det_interp, det_inp, det_out = load_model(DET_MODEL_PATH)
DET_FLOAT = det_inp["dtype"] == np.float32
print(f"  DET input : {det_inp['shape']}  float={DET_FLOAT}")
print(f"  DET output: {det_out[0]['shape']}")


# ── CSV LOG ──────────────────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Type,Class,Confidence,Details\n")


# ── PREPROCESS ───────────────────────────────────────────────────
def preprocess(frame, input_size, is_float):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_float:
        return np.expand_dims(img.astype(np.float32) / 255.0, 0)
    return np.expand_dims(img.astype(np.uint8), 0)


# ── DECODE OBB (pothole model) ───────────────────────────────────
def decode_obb(raw, orig_w, orig_h, conf_thresh):
    """
    raw shape: (1, 6, 2100)
    channels : [xc, yc, w, h, angle_rad, conf]
    coords   : normalised to INPUT_SIZE
    """
    preds = raw[0].T    # → (2100, 6)

    rboxes = []
    scores = []

    sx = orig_w / INPUT_SIZE
    sy = orig_h / INPUT_SIZE

    for row in preds:
        conf = float(row[5])
        if conf < conf_thresh:
            continue

        xc  = float(row[0]) * sx
        yc  = float(row[1]) * sy
        w   = float(row[2]) * sx
        h   = float(row[3]) * sy
        ang = float(np.degrees(row[4]))

        rboxes.append((xc, yc, w, h, ang))
        scores.append(conf)

    if not rboxes:
        return [], []

    # NMS via axis-aligned proxy
    aabb = [[int(cx-w/2), int(cy-h/2), int(w), int(h)]
            for cx, cy, w, h, _ in rboxes]
    idx = cv2.dnn.NMSBoxes(aabb, scores, conf_thresh, 0.45)
    if len(idx) == 0:
        return [], []

    idx = idx.flatten()
    return [rboxes[i] for i in idx], [scores[i] for i in idx]


# ── DECODE DETECTION (COCO obstacle model) ───────────────────────
def decode_det(raw, orig_w, orig_h, conf_thresh):
    """
    raw shape: (1, 84, 2100)  — 80 COCO classes + 4 box coords
    channels : [xc, yc, w, h, cls0..cls79]
    coords   : normalised to INPUT_SIZE
    """
    preds = raw[0].T    # → (2100, 84)

    boxes     = []
    scores    = []
    class_ids = []

    sx = orig_w / INPUT_SIZE
    sy = orig_h / INPUT_SIZE

    for row in preds:
        cls_scores = row[4:]
        cls_id     = int(np.argmax(cls_scores))

        # Only care about our obstacle classes
        if cls_id not in OBSTACLE_CLASS_IDS:
            continue

        conf = float(cls_scores[cls_id])
        if conf < conf_thresh:
            continue

        xc = float(row[0]) * sx
        yc = float(row[1]) * sy
        w  = float(row[2]) * sx
        h  = float(row[3]) * sy

        x1 = max(0, int(xc - w / 2))
        y1 = max(0, int(yc - h / 2))
        bw = max(1, min(int(w), orig_w - x1))
        bh = max(1, min(int(h), orig_h - y1))

        boxes.append([x1, y1, bw, bh])
        scores.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return [], [], []

    idx = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.45)
    if len(idx) == 0:
        return [], [], []

    idx = idx.flatten()
    return ([boxes[i]     for i in idx],
            [scores[i]    for i in idx],
            [class_ids[i] for i in idx])


# ── DRAW OBB (rotated pothole boxes) ─────────────────────────────
def draw_obb(frame, overlay, rboxes, scores):
    for (cx, cy, w, h, ang), score in zip(rboxes, scores):
        rect = ((cx, cy), (w, h), ang)
        pts  = np.int32(cv2.boxPoints(rect))

        cv2.fillPoly(overlay, [pts], POTHOLE_COLOUR)
        cv2.polylines(frame, [pts], True, POTHOLE_COLOUR, 2)

        top = pts[np.argmin(pts[:, 1])]
        text = f"Pothole {score:.2f}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx, ly = max(0, int(top[0])), max(th+8, int(top[1]))
        cv2.rectangle(frame, (lx, ly-th-8), (lx+tw+4, ly),
                      POTHOLE_COLOUR, -1)
        cv2.putText(frame, text, (lx+2, ly-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)


# ── DRAW DET (regular obstacle boxes) ────────────────────────────
def draw_det(frame, boxes, scores, class_ids):
    for (x, y, w, h), score, cid in zip(boxes, scores, class_ids):
        name, colour = OBSTACLE_CLASS_IDS[cid]
        text = f"{name} {score:.2f}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)

        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame,
                      (x, max(y-th-8, 0)), (x+tw+4, max(y, th+8)),
                      colour, -1)
        cv2.putText(frame, text,
                    (x+2, max(y-4, th+4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)


# ── MAIN LOOP ────────────────────────────────────────────────────
vs = VideoStream(width=CAMERA_W, height=CAMERA_H).start()
time.sleep(2.0)

fps_count   = 0
frame_count = 0
start_time  = time.time()

print("\nSystem running — press Q to quit")
print("Detecting: Potholes (OBB) + Persons, Dogs, Vehicles (COCO)\n")

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]
        overlay = frame.copy()

        # ── PREPROCESS (shared image for both models) ────────────
        img_obb = preprocess(frame, INPUT_SIZE, OBB_FLOAT)
        img_det = preprocess(frame, INPUT_SIZE, DET_FLOAT)

        # ── INFERENCE — OBB (potholes) ───────────────────────────
        t0 = time.time()
        obb_interp.set_tensor(obb_inp["index"], img_obb)
        obb_interp.invoke()
        raw_obb   = obb_interp.get_tensor(obb_out[0]["index"])
        lat_obb   = (time.time() - t0) * 1000

        # ── INFERENCE — DET (obstacles) ──────────────────────────
        t1 = time.time()
        det_interp.set_tensor(det_inp["index"], img_det)
        det_interp.invoke()
        raw_det   = det_interp.get_tensor(det_out[0]["index"])
        lat_det   = (time.time() - t1) * 1000

        total_lat = lat_obb + lat_det

        # ── DECODE ───────────────────────────────────────────────
        rboxes, obb_scores = decode_obb(
            raw_obb, orig_w, orig_h, OBB_CONF)

        det_boxes, det_scores, det_cls = decode_det(
            raw_det, orig_w, orig_h, DET_CONF)

        # ── DRAW ─────────────────────────────────────────────────
        if rboxes:
            draw_obb(frame, overlay, rboxes, obb_scores)
        if det_boxes:
            draw_det(frame, det_boxes, det_scores, det_cls)

        # Blend OBB fill overlay
        if rboxes:
            cv2.addWeighted(overlay, FILL_ALPHA,
                            frame, 1-FILL_ALPHA, 0, frame)

        # ── LOG ──────────────────────────────────────────────────
        if rboxes or det_boxes:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(LOG_FILE, "a") as f:
                for (cx,cy,w,h,ang), sc in zip(rboxes, obb_scores):
                    f.write(f"{ts},Pothole,Pothole,{sc:.2f},"
                            f"cx={cx:.0f} cy={cy:.0f} angle={ang:.1f}\n")
                for (x,y,w,h), sc, cid in zip(
                        det_boxes, det_scores, det_cls):
                    name, _ = OBSTACLE_CLASS_IDS[cid]
                    f.write(f"{ts},Obstacle,{name},{sc:.2f},"
                            f"x={x} y={y} w={w} h={h}\n")

        # ── STATUS OVERLAY ───────────────────────────────────────
        fps_count   += 1
        frame_count += 1
        fps          = fps_count / (time.time() - start_time)

        if frame_count % DISPLAY_EVERY == 0:
            n_pot = len(rboxes)
            n_obs = len(det_boxes)

            # Anomaly indicator
            if n_pot > 0 or n_obs > 0:
                alert = f"ALERT: {n_pot} pothole(s)  {n_obs} obstacle(s)"
                cv2.putText(frame, alert, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Road Clear", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"OBB+DET INT8 | FPS:{fps:.1f} | "
                f"OBB:{lat_obb:.0f}ms DET:{lat_det:.0f}ms",
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.imshow("Bharat AI-SoC | Road Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    vs.stop()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"\nSession stats:")
    print(f"  Runtime : {elapsed:.0f}s")
    print(f"  Avg FPS : {fps_count/elapsed:.1f}")
    print(f"  Log     : {LOG_FILE}")
