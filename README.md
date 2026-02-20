<div align="center">

<!-- ANIMATED HEADER -->
<img src="https://capsule-render.vercel.app/api?type=venom&height=300&text=Bharat%20AI-SoC&fontSize=70&color=0:00ff88,50:00ccff,100:ff0066&stroke=ffffff&strokeWidth=2&fontColor=ffffff&animation=fadeIn&desc=Road%20Anomaly%20Detection%20%7C%20ARM%20Edge%20AI&descSize=22&descAlignY=75" width="100%"/>

<!-- BADGES ROW 1 -->
<p>
  <img src="https://img.shields.io/badge/Platform-Raspberry%20Pi%205-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Architecture-ARM%20Cortex--A76-0091BD?style=for-the-badge&logo=arm&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-YOLOv8n--OBB-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"/>
</p>

<!-- BADGES ROW 2 -->
<p>
  <img src="https://img.shields.io/badge/Quantization-INT8-00C851?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Delegate-XNNPACK-764ABC?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FPS-43.6%20single%20%7C%2022%20dual-FF4444?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/mAP50-98.2%25-00C851?style=for-the-badge"/>
</p>

<!-- TYPING ANIMATION -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=18&pause=1000&color=00FF88&center=true&vCenter=true&width=700&lines=Real-Time+Road+Anomaly+Detection+%F0%9F%9B%A3%EF%B8%8F;43.6+FPS+on+ARM+Cortex-A76+%E2%9A%A1;YOLOv8n-OBB+%2B+INT8+Quantization+%F0%9F%8E%AF;mAP50%3A+0.982+%7C+mAP50-95%3A+0.857+%F0%9F%94%A5;Potholes+%2B+Obstacles+%2B+Persons+%2B+Dogs+%F0%9F%90%95" alt="Typing SVG" />
</a>

</div>

---

## 🧠 What This Does

A fully edge-deployed AI system running on **Raspberry Pi 5** that:
- Detects **potholes** using oriented bounding boxes (rotated to fit irregular shapes)
- Detects **unexpected obstacles** — people, dogs, vehicles — using a second COCO model
- Runs at **43.6 FPS average** with only **~10ms inference latency**
- Logs every detection with timestamp and GPS-ready coordinates to CSV
- Requires **zero internet connection** — fully on-device inference

---

## 📊 Performance

<div align="center">

| Metric | Value |
|--------|-------|
| 🎯 OBB mAP50 | **0.982** |
| 📐 OBB mAP50-95 | **0.857** |
| ⚡ Avg Inference Latency | **~10ms** |
| 🎬 Average FPS | **43.6** |
| 💾 Model Size (INT8) | **3.13 MB** |
| 🌡️ CPU Utilization | **~65%** |
| 🏃 Input Resolution | **320 × 320 px** |

</div>

> **Challenge Target:** ≥ 5 FPS → **Achieved 8.7× the target**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BHARAT AI-SoC INFERENCE PIPELINE                 │
├──────────────┬──────────────────────────────┬───────────────────────┤
│              │                              │                       │
│  Pi Camera   │   Preprocess (320×320)       │   CSV Anomaly Log     │
│  Module v2   │   BGR→RGB + Normalize        │   Timestamped         │
│  (CSI)       │                              │   Coordinates         │
│              │                              │                       │
└──────┬───────┴──────────────┬───────────────┴───────────┬───────────┘
       │                      │                           │
       ▼                      ▼                           ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐
│  VideoStream│    │  YOLOv8n-OBB     │    │  YOLOv8n COCO           │
│  Thread     │───▶│  INT8 TFLite     │    │  INT8 TFLite            │
│  (lock-safe)│    │  POTHOLES        │    │  OBSTACLES              │
└─────────────┘    │  mAP50: 0.982    │    │  80 Classes             │
                   └────────┬─────────┘    └───────────┬─────────────┘
                            │                          │
                            ▼                          ▼
                   ┌─────────────────────────────────────────┐
                   │   Post-Processing                        │
                   │   OBB Rotated Boxes + NMS                │
                   │   COCO Axis-Aligned Boxes + NMS          │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   OpenCV Overlay                         │
                   │   Filled rotated boxes (potholes)        │
                   │   Labelled rectangles (obstacles)        │
                   │   FPS + Latency HUD                      │
                   └─────────────────────────────────────────┘
```

**Entire pipeline runs on ARM Cortex-A76 CPU — no accelerators.**

---

## 🚀 Why ARM-Optimized?

| Optimization | Technical Detail |
|---|---|
| **YOLOv8n-OBB** | C2f blocks use depthwise separable convolutions that map directly to ARM NEON SIMD instructions |
| **INT8 Quantization** | Reduces model from FP32 → INT8 — uses integer ALU lanes on Cortex-A76, ~4× faster than FP32 |
| **XNNPACK Delegate** | Google's ARM-tuned kernel library — automatically selects NEON-optimized ops |
| **4-Thread Inference** | All 4 Cortex-A76 cores utilized via `num_threads=4` |
| **320px Input** | 4× less compute vs 640px — optimal latency/accuracy for edge deployment |
| **Lock-safe Camera Thread** | Decouples capture from inference — eliminates I/O blocking |

---

## 📦 Repository Structure

```
📦 Bharat-AI-SoC-Road-Anomaly-Detection/
├── 📄 README.md                          ← You are here
├── 📄 Project_Report.md                  ← Full technical report
├── 🐍 inference_final_rpi5.py            ← Main inference script (OBB + COCO)
├── 🐍 inference_obb_rpi5.py              ← OBB-only pothole detection
├── 📓 Bharat_AI_SoC_OBB.ipynb           ← Training notebook (Colab)
├── 📓 Export_COCO_Obstacle_Model.ipynb   ← COCO export notebook
├── 🤖 pothole_obb_int8.tflite            ← Trained OBB model (3.13 MB)
├── 🤖 yolov8n_coco_int8.tflite          ← COCO obstacle model
└── 📄 labels.txt                         ← Class labels
```

---

## ⚙️ Setup & Installation

### Hardware Required
- Raspberry Pi 5 (4GB or 8GB)
- Raspberry Pi Camera Module v2 (CSI)
- Active cooling (fan heatsink recommended)
- High-write microSD card (A2 rated)

### Software Setup

```bash
# 1. Clone repo
git clone https://github.com/kkjjkamal123/Bharat-AI-SoC-Road-Anomaly-Detection
cd Bharat-AI-SoC-Road-Anomaly-Detection

# 2. Install dependencies
sudo apt update
sudo apt install python3-opencv python3-picamera2 -y
pip install ai-edge-litert --break-system-packages

# 3. Run
python3 inference_final_rpi5.py
```

### Controls
| Key | Action |
|-----|--------|
| `Q` | Quit and show session stats |

---

## 🎯 Detection Classes

<div align="center">

| Class | Model | Box Type | Colour |
|-------|-------|----------|--------|
| 🕳️ Pothole | YOLOv8n-OBB | Rotated (fitted) | 🟢 Green |
| 🧍 Person | YOLOv8n COCO | Axis-aligned | 🟠 Orange |
| 🐕 Dog | YOLOv8n COCO | Axis-aligned | 🩷 Pink |
| 🚗 Car | YOLOv8n COCO | Axis-aligned | 🟣 Magenta |
| 🚌 Bus / Truck | YOLOv8n COCO | Axis-aligned | 🟣 Purple |
| 🚦 Traffic Light | YOLOv8n COCO | Axis-aligned | 🟡 Yellow |
| 🛑 Stop Sign | YOLOv8n COCO | Axis-aligned | 🔴 Red |
| 🚲 Bicycle | YOLOv8n COCO | Axis-aligned | 🔵 Blue |

</div>

---

## 📈 Training Details

```
Architecture  : YOLOv8n-OBB (Oriented Bounding Box — Nano)
Dataset       : Roboflow Pothole OBB Dataset (198 images)
Epochs        : 50  |  Batch: 16  |  Image: 320px
Optimizer     : AdamW  lr0=0.001
Augmentation  : Mosaic 1.0, flips, HSV, albumentations
Pretrained    : COCO (fine-tuned on pothole data)
Export        : TFLite INT8 post-training static quantization
Calibration   : Training set (representative dataset)
```

### Training Convergence

| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| 10 | 0.880 | 0.720 |
| 20 | 0.939 | 0.790 |
| 35 | 0.970 | 0.832 |
| **50** | **0.982** | **0.857** |

---

## 📝 Output Log Format

Every detection is saved to `road_anomalies_log.csv`:

```csv
Timestamp,Type,Class,Confidence,Details
2026-02-20 21:45:11,Pothole,Pothole,0.94,cx=312 cy=278 angle=23.4
2026-02-20 21:45:11,Obstacle,Person,0.87,x=120 y=45 w=80 h=210
2026-02-20 21:45:12,Obstacle,Dog,0.76,x=340 y=190 w=95 h=88
```

---

## 🔧 Configuration

Edit the top of `inference_final_rpi5.py` to tune behaviour:

```python
OBB_CONF    = 0.35    # raise to reduce false positives
DET_CONF    = 0.45    # obstacle confidence threshold
FILL_ALPHA  = 0.25    # OBB fill transparency (0=none, 1=solid)
DISPLAY_EVERY = 2     # render every Nth frame (raise to increase FPS)
NUM_THREADS = 4       # CPU cores for inference
```

---

## 🏆 Challenge Details

**Bharat AI-SoC Student Challenge | Problem Statement 3**
Real-Time Road Anomaly Detection from Dashcam Footage on Raspberry Pi

- **Organizers:** Arm Education, IIT Delhi, MeitY
- **Requirement:** ≥ 5 FPS real-time inference on CPU only
- **Achieved:** 43.6 FPS on single mode ( Potholes alone ) — **8.7× the requirement**
- **Achieved:** ~22 FPS on dual mode ( Potholes and Obstacles ) — **4.4× the requirement**
- **Approach:** Dual-model pipeline (OBB potholes + COCO obstacles), INT8 quantized, XNNPACK accelerated

---

## 👤 Author

**Made for Bharat AI-SoC Challenge 2026**

[![GitHub](https://img.shields.io/badge/GitHub-kkjjkamal123-181717?style=for-the-badge&logo=github)](https://github.com/kkjjkamal123)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00ff88,100:00ccff&height=100&section=footer"/>
</div>
