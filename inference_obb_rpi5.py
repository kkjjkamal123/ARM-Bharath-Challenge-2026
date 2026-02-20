import cv2
import numpy as np
import time
import os
import threading
import ai_edge_litert.interpreter as tflite
from picamera2 import Picamera2

# ================================================================
#  Bharat AI-SoC | Road Anomaly Detection — OBB
#  Model  : YOLOv8n-obb INT8 TFLite
#  Target : RPi 5 ARM Cortex-A76 via XNNPACK
#  Output : (1, 6, num_anchors) — [xc, yc, w, h, angle, conf]
# ================================================================

# ── CONFIG ───────────────────────────────────────────────────────
MODEL_PATH     = "pothole_obb_int8.tflite"
LABELS_PATH    = "labels.txt"
CONF_THRESHOLD = 0.35
NMS_THRESHOLD  = 0.45
INPUT_SIZE     = 320
NUM_THREADS    = 4
CAMERA_W       = 640
CAMERA_H       = 480
DISPLAY_EVERY  = 2
LOG_FILE       = "road_anomalies_log.csv"
FILL_ALPHA     = 0.25    # transparency of filled rotated box


# ── LOAD LABELS ──────────────────────────────────────────────────
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH) as f:
        LABELS = [l.strip() for l in f if l.strip()]
else:
    LABELS = ["pothole"]
print("Classes:", LABELS)


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


# ── LOAD MODEL ───────────────────────────────────────────────────
print("Loading YOLOv8n-obb model...")
interp = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=NUM_THREADS
)
interp.allocate_tensors()

inp_det = interp.get_input_details()[0]
out_det = sorted(interp.get_output_details(), key=lambda x: x["index"])[0]

print(f"Input  : {inp_det['shape']}  {inp_det['dtype'].__name__}")
print(f"Output : {out_det['shape']}  {out_det['dtype'].__name__}")

IS_FLOAT = inp_det['dtype'] == np.float32
NC = len(LABELS)


# ── CSV LOG ──────────────────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Class,Confidence,CX,CY,W,H,AngleDeg\n")


# ── OBB DECODE ───────────────────────────────────────────────────
def decode_obb(raw_out, orig_w, orig_h, conf_thresh, nms_thresh):
    """
    raw_out shape: (1, 6+NC, num_anchors)
    channels: [xc, yc, w, h, angle_rad, conf, (extra cls scores)]
    coords: normalised to INPUT_SIZE (0 to INPUT_SIZE)
    angle : radians
    Returns lists of (cx, cy, w, h, angle_deg) in orig frame coords
    """
    preds = raw_out[0].T     # → (num_anchors, 6+NC)

    rboxes  = []   # (cx, cy, w, h, angle_deg) in orig coords
    scores  = []
    cls_ids = []

    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    for row in preds:
        xc    = float(row[0])
        yc    = float(row[1])
        w     = float(row[2])
        h     = float(row[3])
        angle = float(row[4])   # radians

        if NC == 1:
            conf   = float(row[5])
            cls_id = 0
        else:
            cls_scores = row[5:5+NC]
            cls_id     = int(np.argmax(cls_scores))
            conf       = float(cls_scores[cls_id])

        if conf < conf_thresh:
            continue

        # Scale to original frame
        cx_px = xc * scale_x
        cy_px = yc * scale_y
        w_px  = w  * scale_x
        h_px  = h  * scale_y
        angle_deg = np.degrees(angle)

        rboxes.append((cx_px, cy_px, w_px, h_px, angle_deg))
        scores.append(conf)
        cls_ids.append(cls_id)

    if not rboxes:
        return [], [], []

    # NMS using axis-aligned proxy boxes
    aabb = []
    for cx, cy, w, h, _ in rboxes:
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        aabb.append([x1, y1, int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(aabb, scores, conf_thresh, nms_thresh)
    if len(indices) == 0:
        return [], [], []

    idx = indices.flatten()
    return (
        [rboxes[i]  for i in idx],
        [scores[i]  for i in idx],
        [cls_ids[i] for i in idx],
    )


# ── DRAW OBB ─────────────────────────────────────────────────────
def draw_obb(frame, rboxes, scores, cls_ids):
    """Draw rotated bounding boxes with filled overlay and label."""
    overlay = frame.copy()

    for (cx, cy, w, h, angle_deg), score, cid in zip(rboxes, scores, cls_ids):
        label  = LABELS[cid] if cid < len(LABELS) else f"cls{cid}"
        colour = (0, 255, 0)

        # Get 4 corner points of rotated box
        rect   = ((cx, cy), (w, h), angle_deg)
        pts    = cv2.boxPoints(rect)
        pts    = np.int32(pts)

        # Fill rotated box on overlay
        cv2.fillPoly(overlay, [pts], colour)

        # Draw rotated box outline on frame
        cv2.polylines(frame, [pts], isClosed=True,
                      color=colour, thickness=2)

        # Label at top-left corner of rotated box
        top_pt = pts[np.argmin(pts[:, 1])]
        text   = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx = max(0, int(top_pt[0]))
        ly = max(th + 8, int(top_pt[1]))
        cv2.rectangle(frame,
                      (lx, ly - th - 8), (lx + tw + 4, ly),
                      colour, -1)
        cv2.putText(frame, text,
                    (lx + 2, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Blend fill overlay
    cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)
    return frame


# ── MAIN LOOP ────────────────────────────────────────────────────
vs = VideoStream(width=CAMERA_W, height=CAMERA_H).start()
time.sleep(2.0)

fps_count   = 0
frame_count = 0
start_time  = time.time()

print("\nSystem running — press Q to quit\n")

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]

        # ── PREPROCESS ──────────────────────────────────────────
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if IS_FLOAT:
            inp_data = np.expand_dims(
                img.astype(np.float32) / 255.0, axis=0)
        else:
            inp_data = np.expand_dims(img.astype(np.uint8), axis=0)

        # ── INFERENCE ───────────────────────────────────────────
        t0 = time.time()
        interp.set_tensor(inp_det['index'], inp_data)
        interp.invoke()
        raw = interp.get_tensor(out_det['index'])
        latency = (time.time() - t0) * 1000

        # ── DECODE ──────────────────────────────────────────────
        rboxes, scores, cls_ids = decode_obb(
            raw, orig_w, orig_h, CONF_THRESHOLD, NMS_THRESHOLD
        )

        # ── DRAW ────────────────────────────────────────────────
        if rboxes:
            frame = draw_obb(frame, rboxes, scores, cls_ids)

        # ── LOG ─────────────────────────────────────────────────
        if rboxes:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(LOG_FILE, "a") as f:
                for (cx, cy, w, h, ang), score, cid in zip(
                        rboxes, scores, cls_ids):
                    label = LABELS[cid] if cid < len(LABELS) else f"cls{cid}"
                    f.write(f"{ts},{label},{score:.2f},"
                            f"{cx:.1f},{cy:.1f},{w:.1f},{h:.1f},{ang:.1f}\n")

        # ── FPS + DISPLAY ────────────────────────────────────────
        fps_count   += 1
        frame_count += 1
        fps          = fps_count / (time.time() - start_time)

        if frame_count % DISPLAY_EVERY == 0:
            n     = len(rboxes)
            color = (0, 0, 255) if n > 0 else (0, 255, 0)

            cv2.putText(frame, f"Anomalies: {n}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(
                frame,
                f"YOLOv8n-OBB INT8 | FPS:{fps:.1f} | {latency:.0f}ms",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Bharat AI-SoC | Road Anomaly OBB Detection", frame)

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
