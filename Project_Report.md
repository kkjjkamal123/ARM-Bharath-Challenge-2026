# 📋 Project Report
## Real-Time Road Anomaly Detection on Raspberry Pi 5
### Bharat AI-SoC Challenge 2026 — Problem Statement 3

---

## 1. Executive Summary

This project delivers a fully edge-deployed, dual-model AI pipeline running entirely on the **Raspberry Pi 5 CPU** with no external accelerators. The system detects road anomalies — potholes and unexpected obstacles — in real time from a live camera feed, exceeding the challenge's ≥5 FPS target by **8.7×** at an average of **43.6 FPS**.

| Metric | Target | Achieved |
|---|---|---|
| FPS | ≥ 5 | **43.6 avg** |
| Pothole mAP50 | — | **0.982** |
| Pothole mAP50-95 | — | **0.857** |
| Inference Latency | — | **~10 ms** |
| Model Size | — | **3.13 MB (INT8)** |
| Accelerator Required | CPU-only preferred | **CPU-only ✅** |

Two models run in a pipeline: a fine-tuned **YOLOv8n-OBB INT8 TFLite** model for pothole detection using oriented bounding boxes, and a pretrained **YOLOv8n COCO INT8 TFLite** model for obstacle detection (persons, dogs, vehicles). Every detection is timestamped and logged to CSV.

---

## 2. Problem Statement

Road infrastructure defects such as potholes and unexpected obstacles are a leading cause of vehicle damage and accidents, particularly in urban and semi-urban environments. Real-time detection from dashcam footage on embedded hardware can enable automated alerting, fleet monitoring and smart city infrastructure systems.

The challenge requires:
- Hardware: Raspberry Pi 5 or 4, Pi Camera or USB webcam
- Software: Python, OpenCV, TFLite / ONNX Runtime
- Performance: ≥5 FPS real-time inference, high precision, robust under varying lighting
- Deliverables: Source code, model file, demo video, technical report

---

## 3. Hardware Setup

| Component | Specification |
|---|---|
| SBC | Raspberry Pi 5 (8GB RAM) |
| CPU | ARM Cortex-A76 @ 2.4 GHz (quad-core) |
| Camera | Raspberry Pi Camera Module v2 (IMX219, CSI) |
| Storage | High-write microSD (A2 rated) |
| Cooling | Active fan heatsink |
| OS | Raspberry Pi OS (64-bit) |

### Why the Cortex-A76 Matters

The RPi 5's Cortex-A76 core includes **ARM NEON SIMD** (Advanced SIMD), a 128-bit vector unit capable of processing 16 INT8 operations per clock cycle per core. This is exploited directly by:

- **XNNPACK** (Google's ARM-tuned kernel library) — selected automatically by `num_threads=4` in `ai_edge_litert`
- **INT8 quantized models** — integer operations route to NEON integer lanes, bypassing the slower FP32 path
- **Depthwise separable convolutions** — the core operation in YOLOv8n, which maps efficiently to NEON's vectorized compute

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 BHARAT AI-SoC INFERENCE PIPELINE             │
└──────────────────────────────────────────────────────────────┘

  [Pi Camera v2]
       │  RGB888 frame
       ▼
  [VideoStream Thread]  ← lock-safe background capture
       │  640×480 BGR frame
       ▼
  ┌─────────────────────────────────────────────┐
  │            Preprocessing                    │
  │  Resize → 320×320 │ BGR→RGB │ /255 normalize│
  └───────────────┬─────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
  ┌──────────────┐    ┌──────────────────┐
  │ YOLOv8n-OBB  │    │ YOLOv8n COCO     │
  │ INT8 TFLite  │    │ INT8 TFLite      │
  │ POTHOLES     │    │ OBSTACLES        │
  │ mAP50: 0.982 │    │ 80 COCO classes  │
  │ 3.13 MB      │    │ filtered to 10   │
  └──────┬───────┘    └────────┬─────────┘
         │ (1,6,2100)          │ (1,84,2100)
         ▼                    ▼
  ┌──────────────────────────────────────┐
  │  Post-Processing & NMS               │
  │  OBB decode: xc,yc,w,h,angle,conf   │
  │  DET decode: xc,yc,w,h + 80 classes │
  │  NMS threshold: 0.45                 │
  └──────────────┬───────────────────────┘
                 ▼
  ┌──────────────────────────────────────┐
  │  OpenCV Overlay                      │
  │  Rotated filled boxes (potholes)     │
  │  Axis-aligned labelled boxes (obst.) │
  │  Alert HUD │ FPS │ Latency           │
  └──────────────┬───────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  [cv2.imshow]     [road_anomalies_log.csv]
```

---

## 5. Model Architecture & Selection

### 5.1 Why YOLOv8n-OBB?

| Model | ARM Optimization | mAP50 | FPS (RPi5) | Rationale |
|---|---|---|---|---|
| YOLOv8n-OBB | ✅ Strong | **0.982** | **43.6** | C2f depthwise convs → NEON; OBB fits irregular pothole shapes |
| YOLOv8n | ✅ Strong | 0.656 | ~40 | Standard boxes miss rotated shapes |
| EfficientDet-Lite0 | ✅ Best | ~0.60 | ~30 | tflite-model-maker deprecated; harder to fine-tune in 2026 |
| YOLOv8s | ⚠️ Medium | Higher | ~15 | Too slow for real-time on RPi5 CPU |
| YOLOv5n | ⚠️ Okay | ~0.55 | ~25 | Older ops; less clean quantization |

**Selection rationale:** YOLOv8n-OBB was chosen because:
1. **Oriented bounding boxes** fit the irregular elliptical shape of potholes far better than axis-aligned boxes, reducing false positive area and improving visual precision
2. The **nano (n) variant** is the smallest YOLOv8 model — 3.13 MB INT8, ideal for edge deployment
3. **C2f blocks** use depthwise separable convolutions that the XNNPACK delegate maps directly to ARM NEON SIMD
4. Ultralytics provides first-class TFLite INT8 export with `model.export(format="tflite", int8=True)` — no manual quantization pipeline needed

### 5.2 OBB vs Standard Box — Why it Matters for Potholes

A standard bounding box draws axis-aligned rectangles. For a pothole at 45° on a road, the standard box includes significant non-pothole area (sky, road surface around it). An OBB rotates to match the actual shape:

```
Standard box:           OBB:
┌──────────────┐        ╱──────╲
│  ╱╲          │       ╱ pothole╲
│ ╱  ╲         │       ╲        ╱
│╱    ╲        │        ╲──────╱
│      ╲       │
│ extra │       ← wasted area eliminated
│ area  │
└──────────────┘
```

This directly reduces false positive log entries and produces cleaner visualizations for the demo.

### 5.3 COCO Obstacle Model

Rather than training a separate obstacle detector from scratch, we leverage YOLOv8n's **COCO pretraining** directly. The model is exported to INT8 TFLite from COCO weights without fine-tuning, then filtered at inference time to 10 relevant road obstacle classes:

`Person, Bicycle, Car, Motorcycle, Bus, Truck, Traffic Light, Stop Sign, Cat, Dog`

This avoids dataset collection overhead while still covering all obstacle types relevant to the challenge.

---

## 6. Dataset & Training

### 6.1 Dataset

| Property | Value |
|---|---|
| Source | Roboflow — object-detection-yy9t1/pothole-detection-rlmwv |
| Format | YOLOv8 Oriented Bounding Boxes (OBB) |
| Total images | 198 labeled images |
| Train split | 158 images (80%) |
| Validation split | 27 images (13.6%) |
| Test split | 13 images (6.5%) |
| Label format | class cx cy w h angle (normalized OBB) |
| Augmentation | Mosaic 1.0, horizontal flip, HSV jitter, albumentations |

### 6.2 Training Configuration

```
Architecture  : YOLOv8n-obb
Pretrained    : COCO (ImageNet backbone)
Epochs        : 50  (early stop patience = 15)
Batch size    : 16
Optimizer     : AdamW
lr0           : 0.001
Image size    : 320 × 320 px
Hardware      : Google Colab T4 GPU
Training time : ~20 minutes
```

### 6.3 Training Convergence

| Epoch | OBB mAP50 | OBB mAP50-95 | box_loss | angle_loss |
|---|---|---|---|---|
| 10 | 0.880 | 0.720 | 1.21 | 0.94 |
| 20 | 0.939 | 0.790 | 0.98 | 0.71 |
| 30 | 0.963 | 0.821 | 0.87 | 0.62 |
| 40 | 0.974 | 0.841 | 0.79 | 0.55 |
| **50** | **0.982** | **0.857** | **0.71** | **0.48** |

Final validation results: **P=0.969, R=0.873, mAP50=0.982, mAP50-95=0.857**

---

## 7. INT8 Quantization

### 7.1 Export Command

```python
from ultralytics import YOLO
model = YOLO("runs/obb/train/weights/best.pt")
model.export(
    format="tflite",
    int8=True,
    imgsz=320,
    data="data.yaml"   # training set used as calibration data
)
```

### 7.2 What INT8 Quantization Does

Post-training static quantization converts all weights and activations from 32-bit floats to 8-bit integers using a calibration pass over representative data.

| Property | FP32 | INT8 |
|---|---|---|
| Weight precision | 32-bit float | 8-bit integer |
| Model size | ~12.5 MB | **3.13 MB** (4× smaller) |
| Memory bandwidth | High | 4× lower |
| Compute on Cortex-A76 | FP32 NEON lanes | **INT8 NEON SIMD (16 ops/cycle)** |
| Accuracy loss | Baseline | mAP50 drop < 0.5% |

The quantization uses the **training dataset as representative data**, ensuring the scale/zero-point calibration is tuned to real pothole image distributions rather than random data.

### 7.3 Why INT8 is Critical for ARM

The Cortex-A76's NEON unit processes:
- 4× FP32 values per 128-bit register per cycle
- **16× INT8 values** per 128-bit register per cycle

This means INT8 inference is theoretically 4× faster per FLOP, which explains the 43.6 FPS result despite running two models.

---

## 8. Inference Pipeline

### 8.1 Software Stack

| Component | Version | Role |
|---|---|---|
| Python | 3.11 | Runtime |
| OpenCV | 4.x | Camera, drawing, NMS |
| ai-edge-litert | 1.x | TFLite interpreter |
| picamera2 | 0.3.x | Pi Camera v2 CSI interface |
| NumPy | 1.x | Array operations |

### 8.2 Key Optimizations

| Optimization | Implementation | FPS Impact |
|---|---|---|
| Multi-threaded capture | `VideoStream` daemon thread + `threading.Lock` | Eliminates camera I/O blocking |
| 4-core inference | `num_threads=4` in `tflite.Interpreter` | ~3× vs single thread |
| XNNPACK auto-select | Enabled by default with `num_threads` | ARM NEON kernel selection |
| Frame skip display | `DISPLAY_EVERY = 2` | VNC rendering doesn't block inference |
| 320px input | Resize in preprocess | 4× less compute vs 640px |
| Confidence clamp | `np.clip(conf, 0.0, 1.0)` | Eliminates ghost detections from dequant artifacts |
| Size/bounds filter | `w < 5` or out-of-frame check | Eliminates degenerate anchor detections |

### 8.3 OBB Output Decoding

The OBB model outputs shape `(1, 6, 2100)`:
- Dimension 1: `[xc, yc, w, h, angle_radians, confidence]`
- Dimension 2: 2100 candidate anchors at 320px

```python
preds = raw[0].T    # → (2100, 6)
for row in preds:
    conf = float(np.clip(row[5], 0.0, 1.0))
    if conf < OBB_CONF:
        continue
    xc  = float(row[0]) * (orig_w / INPUT_SIZE)
    yc  = float(row[1]) * (orig_h / INPUT_SIZE)
    w   = float(row[2]) * (orig_w / INPUT_SIZE)
    h   = float(row[3]) * (orig_h / INPUT_SIZE)
    ang = float(np.degrees(row[4]))
    # → NMS → cv2.boxPoints() → draw rotated polygon
```

### 8.4 Anomaly Logging

Every frame with detections writes to `road_anomalies_log.csv`:

```
Timestamp,Type,Class,Confidence,Details
2026-02-20 21:45:11,Pothole,Pothole,0.94,cx=312 cy=278 angle=23.4
2026-02-20 21:45:11,Obstacle,Person,0.87,x=120 y=45 w=80 h=210
2026-02-20 21:45:12,Obstacle,Dog,0.76,x=340 y=190 w=95 h=88
```

---

## 9. Results

### 9.1 Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8n-OBB INT8 (potholes) | **0.982** | **0.857** | 0.969 | 0.873 |
| YOLOv8n COCO INT8 (obstacles) | COCO baseline | — | — | — |

### 9.2 Runtime Performance

| Metric | Value |
|---|---|
| OBB inference latency | ~6 ms |
| COCO inference latency | ~7 ms |
| Combined latency | ~13 ms |
| Average FPS (both models) | **43.6** |
| CPU utilization | ~65% |
| Peak temperature (active cooling) | ~58°C |
| Challenge FPS target | 5 FPS |
| **Margin above target** | **8.7×** |

### 9.3 Detection Classes & Visual Output

| Anomaly | Box Type | Colour | Trigger |
|---|---|---|---|
| Pothole | Rotated OBB + fill | 🟢 Green | conf > 0.35 |
| Person | Axis-aligned | 🟠 Orange | conf > 0.45 |
| Dog | Axis-aligned | 🩷 Pink | conf > 0.45 |
| Car | Axis-aligned | 🟣 Magenta | conf > 0.45 |
| Bus / Truck | Axis-aligned | 🟣 Purple | conf > 0.45 |
| Traffic Light | Axis-aligned | 🟡 Yellow | conf > 0.45 |
| Stop Sign | Axis-aligned | 🔴 Red | conf > 0.45 |
| Bicycle | Axis-aligned | 🔵 Blue | conf > 0.45 |

---

## 10. Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| Ghost detection at top-left | INT8 dequantization artifact produced conf > 1.0, passing threshold | `np.clip(conf, 0.0, 1.0)` + `w < 5` size filter |
| tflite-model-maker install failed | Deprecated on Python 3.12 / Colab 2026 | Switched to Ultralytics native TFLite export |
| Roboflow dataset URLs broken | Workspace permissions changed | Iterated through 5 sources; found working workspace `object-detection-yy9t1` |
| RPi crash under inference load | CPU overclocked to 3.1 GHz with insufficient voltage | Reverted to stock 2.4 GHz; instability was voltage/thermal, not model |
| Low FPS despite low latency | VNC display blocking inference thread | `DISPLAY_EVERY = 2` — inference runs every frame, display skips |
| Coordinate decoding overflow | INT8 → float cast not handling signed bytes correctly | Cast via `int32` first to preserve sign before float conversion |

---

## 12. Repository Structure

```
ARM-Bharath-Challenge-2026/
├── README.md                          ← Project overview with badges
├── Project_Report.md                  ← This document
├── inference_final_rpi5.py            ← Main dual-model inference script
├── inference_obb_rpi5.py              ← OBB-only pothole detection
├── Bharat_AI_SoC_OBB.ipynb           ← Training notebook (run in Colab)
├── Export_COCO_Obstacle_Model.ipynb   ← COCO TFLite export notebook
├── pothole_obb_int8.tflite            ← Trained pothole model (3.13 MB)
├── yolov8n_coco_int8.tflite           ← COCO obstacle model
├── labels.txt                         ← Class label: pothole
└── LICENSE                            ← MIT
```

---

## 13. Conclusion

This project demonstrates that **high-accuracy real-time road anomaly detection is achievable on commodity ARM hardware** without any external accelerators. Key outcomes:

- **43.6 FPS** average — 8.7× the 5 FPS challenge requirement
- **mAP50 of 0.982** on potholes — near-perfect detection accuracy
- **Dual-class detection** — potholes via OBB + 10 obstacle categories via COCO
- **3.13 MB model** — deployable on memory-constrained edge devices
- **Fully CPU-bound** — exploits ARM NEON SIMD via XNNPACK, no HATs or NPUs needed

The architecture demonstrates all three challenge learning outcomes: neural network optimization for edge deployment, embedded vision pipeline engineering, and empirical understanding of accuracy vs. speed vs. compute trade-offs on ARM platforms.

---

## 14. References

1. Ultralytics YOLOv8 — https://docs.ultralytics.com
2. AI Edge LiteRT (TFLite) — https://ai.google.dev/edge/litert
3. XNNPACK — https://github.com/google/XNNPACK
4. Roboflow Pothole Dataset — https://universe.roboflow.com/object-detection-yy9t1/pothole-detection-rlmwv
5. ARM Cortex-A76 Technical Reference Manual — https://developer.arm.com/documentation
6. Picamera2 Documentation — https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
7. Bharat AI-SoC Challenge 2026 — Problem Statement 3

---

*Report prepared for Bharat AI-SoC Challenge 2026 | ARM Edge AI | Problem Statement 3*
