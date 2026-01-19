# DeepFake Engine v2

**DeepFake Engine v2** is a minimal end-to-end **deepfake detection pipeline** designed for research prototyping and hackathon use.
It supports **training a custom model** and running **inference on images and videos**, with explicit support for **multiple faces per frame**.

> ⚠️ This repository does **not include any training dataset**.
> Users are expected to supply their own real and fake images or videos.

---

## 🔍 Key Capabilities

- Dual-CNN (double convolutional) deepfake detection model
- Spatial + frequency-domain artifact analysis
- Multi-face detection and verification using **FaceNet (MTCNN)**
- Image inference and extensible video inference pipeline
- Designed for small datasets and rapid experimentation

---

## 🧠 System Overview

The pipeline follows a standard **forensic deepfake detection workflow**:

### 1️⃣ Input

- User-provided images or videos
- Supports both single-person and multi-person scenes

---

### 2️⃣ Frame Sampling (Videos)

- Videos are sampled at a fixed frame rate
- Frames are extracted uniformly across time to avoid bias

---

### 3️⃣ Face Detection & Preprocessing (Multi-Face)

- **FaceNet (MTCNN)** is used to detect **all faces** present in a frame
- For each detected face:
  - Face is cropped
  - Resized to a fixed resolution
  - Normalized before CNN processing

This enables:

- Independent verification of **multiple individuals**
- Per-face confidence scores
- Robust handling of crowded or real-world scenes

---

### 4️⃣ Deepfake Detection Model (Double CNN Architecture)

The core model uses **two parallel convolutional neural networks**:

#### 🔹 Spatial (RGB) CNN

- CNN backbone inspired by Xception / EfficientNet
- Learns visible manipulation artifacts such as:
  - Texture inconsistencies
  - Blending and boundary errors
  - Color and lighting mismatches

#### 🔹 Frequency CNN

- Operates on frequency-domain representations (FFT)
- Captures:
  - GAN upsampling artifacts
  - High-frequency noise patterns
  - Compression-related inconsistencies

#### 🔹 Feature Fusion & Classification

- Features from both CNN branches are concatenated
- A classifier outputs a **probability score (Real vs Fake)**

This **double-CNN design** improves generalization by combining:

- Human-visible spatial cues
- Machine-generated frequency artifacts

---

### 5️⃣ Inference

#### Image Inference

- Detects **all faces** in an image
- Produces:
  - Real/Fake prediction per face
  - Confidence score for each prediction

#### Video Inference

- Performs frame-level, face-level inference
- Predictions can be aggregated to obtain:
  - Video-level classification
  - (Optional) temporal consistency analysis

---

## 🏋️ Training

```bash
python train.py
```

- Trains the dual-CNN model on user-supplied data
- The best model checkpoint is saved automatically and used during inference

---

## 🔎 Inference Example

```bash
python inference.py /path/to/image.jpg
```

- Supports images containing multiple faces
- Outputs predictions independently for each detected face

---

## 📌 Notes

- No dataset is bundled with this repository
- Retraining updates inference behavior automatically
- Multi-face support is detection-based (not identity recognition)

---

## 📖 Acknowledgements

- Face detection powered by **FaceNet (MTCNN)**
- CNN-based forensic modeling inspired by deepfake detection research
- System design and implementation by **Ranit Laha**

---

## 🚀 Possible Extensions

- Timestamp-level manipulation localization in videos
- Temporal smoothing of frame predictions
- Identity-aware tracking using face embeddings
- Attention-based temporal modeling

---

**DeepFake Engine v2** focuses on combining a **double CNN architecture** with **FaceNet-based multi-face analysis**, making it well-suited for hackathons, demos, and forensic research prototypes.
