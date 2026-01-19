"""
Deepfake Detection with GUI

Usage:
  python 4_inference_gui.py

Features:
- GUI window with "Choose File" button
- Supports images and videos
- Shows results in the window
"""

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import threading


# ----------------------------
# Model (same as training)
# ----------------------------
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray.unsqueeze(1))
        magnitude = torch.abs(torch.fft.fftshift(fft))
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)

        x = magnitude.repeat(1, 3, 1, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.pool(x).flatten(1)


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg"
        )
        self.freq_branch = FrequencyBranch()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        rgb_feat = self.backbone(x)
        freq_feat = self.freq_branch(x)
        combined = torch.cat([rgb_feat, freq_feat], dim=1)
        return self.classifier(combined)


# ----------------------------
# Utils
# ----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> normalized tensor (1,3,224,224)"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def ema_smooth(scores, alpha=0.6):
    """Exponential moving average smoothing"""
    if not scores:
        return scores
    out = [scores[0]]
    for x in scores[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def scores_to_segments(scores, times, t_high=0.6, t_low=0.45, min_duration=0.5, merge_gap=0.35):
    segments = []
    in_seg = False
    start_t = None
    seg_scores = []

    for p, t in zip(scores, times):
        if not in_seg:
            if p >= t_high:
                in_seg = True
                start_t = t
                seg_scores = [p]
        else:
            seg_scores.append(p)
            if p <= t_low:
                end_t = t
                segments.append([start_t, end_t, float(np.mean(seg_scores))])
                in_seg = False
                start_t = None
                seg_scores = []

    if in_seg and start_t is not None:
        segments.append([start_t, times[-1], float(np.mean(seg_scores))])

    segments = [s for s in segments if (s[1] - s[0]) >= min_duration]

    if not segments:
        return []

    merged = [segments[0]]
    for s in segments[1:]:
        prev = merged[-1]
        gap = s[0] - prev[1]
        if gap <= merge_gap:
            prev_dur = max(prev[1] - prev[0], 1e-6)
            s_dur = max(s[1] - s[0], 1e-6)
            prev[1] = max(prev[1], s[1])
            prev[2] = float((prev[2] * prev_dur + s[2] * s_dur) / (prev_dur + s_dur))
        else:
            merged.append(s)

    return merged


def predict_image(model, image_path: str, device):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x = preprocess_bgr_frame(img).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, 1)
        fake_prob = probs[0, 1].item()

    return {
        "input_type": "image",
        "is_fake": fake_prob > 0.5,
        "confidence": round(fake_prob, 4),
    }


def predict_video(model, video_path: str, device, sample_fps=5, batch_size=16, smooth_alpha=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if not vid_fps or vid_fps <= 0:
        vid_fps = 30.0

    stride = max(int(round(vid_fps / sample_fps)), 1)

    frames = []
    times = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            t = frame_idx / vid_fps
            frames.append(frame)
            times.append(t)
        frame_idx += 1

    cap.release()

    if not frames:
        return {
            "input_type": "video",
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "manipulated_segments": [],
            "note": "No frames extracted from video."
        }

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            xs = torch.cat([preprocess_bgr_frame(f) for f in batch], dim=0).to(device)
            outputs = model(xs)
            probs = torch.softmax(outputs, 1)[:, 1].detach().cpu().numpy().tolist()
            scores.extend(probs)

    smooth_scores = ema_smooth(scores, alpha=smooth_alpha)

    k = max(1, int(0.1 * len(smooth_scores)))
    topk = sorted(smooth_scores, reverse=True)[:k]
    overall_conf = float(np.mean(topk))

    segments = scores_to_segments(smooth_scores, times, t_high=0.6, t_low=0.45, min_duration=0.5, merge_gap=0.35)

    manipulated_segments = [
        {
            "start_time": format_time(s[0]),
            "end_time": format_time(s[1]),
            "confidence": round(s[2], 4),
        }
        for s in segments
    ]

    return {
        "input_type": "video",
        "video_is_fake": overall_conf > 0.5,
        "overall_confidence": round(overall_conf, 4),
        "manipulated_segments": manipulated_segments,
    }


# ----------------------------
# GUI Application
# ----------------------------
class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detector")
        self.root.geometry("700x600")
        
        self.model = None
        self.device = None
        self.current_file = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Title
        title = tk.Label(
            self.root, 
            text="🔍 Deepfake Detection System", 
            font=("Arial", 20, "bold"),
            fg="#2c3e50"
        )
        title.pack(pady=20)
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, padx=20, fill="x")
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            font=("Arial", 10),
            fg="#7f8c8d",
            wraplength=500
        )
        self.file_label.pack(side="left", expand=True, fill="x")
        
        # Choose File Button
        self.choose_btn = tk.Button(
            file_frame,
            text="📁 Choose File",
            command=self.choose_file,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.choose_btn.pack(side="right", padx=10)
        
        # Analyze Button
        self.analyze_btn = tk.Button(
            self.root,
            text="🔎 Analyze",
            command=self.analyze_file,
            font=("Arial", 14, "bold"),
            bg="#2ecc71",
            fg="white",
            padx=40,
            pady=15,
            cursor="hand2",
            state="disabled"
        )
        self.analyze_btn.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=600
        )
        
        # Results frame
        results_label = tk.Label(
            self.root,
            text="Results:",
            font=("Arial", 12, "bold"),
            fg="#2c3e50"
        )
        results_label.pack(pady=(20, 5))
        
        self.results_text = scrolledtext.ScrolledText(
            self.root,
            height=15,
            width=80,
            font=("Courier", 10),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        self.results_text.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Loading model...",
            font=("Arial", 9),
            fg="#95a5a6",
            anchor="w"
        )
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=5)
    
    def load_model(self):
        def load():
            try:
                self.device = get_device()
                self.model = DeepfakeDetector().to(self.device)
                
                ckpt_path = Path("models/best_model.pth")
                if not ckpt_path.exists():
                    self.status_label.config(
                        text="❌ ERROR: models/best_model.pth not found!",
                        fg="red"
                    )
                    return
                
                checkpoint = torch.load(str(ckpt_path), map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                
                auc = checkpoint.get("auc", None)
                status = f"✅ Model loaded on {self.device}"
                if auc:
                    status += f" (AUC: {auc:.4f})"
                
                self.status_label.config(text=status, fg="green")
                self.choose_btn.config(state="normal")
                
            except Exception as e:
                self.status_label.config(
                    text=f"❌ Error loading model: {str(e)}",
                    fg="red"
                )
        
        threading.Thread(target=load, daemon=True).start()
    
    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv"),
                ("Images", "*.jpg *.jpeg *.png"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.current_file = file_path
            self.file_label.config(
                text=f"Selected: {Path(file_path).name}",
                fg="#2c3e50"
            )
            self.analyze_btn.config(state="normal")
            self.results_text.delete(1.0, tk.END)
    
    def analyze_file(self):
        if not self.current_file or not self.model:
            return
        
        def analyze():
            try:
                # Show progress
                self.progress.pack(pady=10)
                self.progress.start(10)
                self.analyze_btn.config(state="disabled")
                self.choose_btn.config(state="disabled")
                self.status_label.config(text="🔄 Analyzing...", fg="#f39c12")
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                
                # Process file
                suffix = Path(self.current_file).suffix.lower()
                
                if suffix in [".jpg", ".jpeg", ".png"]:
                    result = predict_image(self.model, self.current_file, self.device)
                else:
                    result = predict_video(self.model, self.current_file, self.device)
                
                # Display results
                output = json.dumps(result, indent=2)
                self.results_text.insert(1.0, output)
                
                # Color code result
                if result.get("is_fake") or result.get("video_is_fake"):
                    verdict = "⚠️ FAKE DETECTED"
                    color = "red"
                else:
                    verdict = "✅ APPEARS REAL"
                    color = "green"
                
                self.status_label.config(text=verdict, fg=color)
                
            except Exception as e:
                self.results_text.insert(1.0, f"ERROR: {str(e)}")
                self.status_label.config(text=f"❌ Error: {str(e)}", fg="red")
            
            finally:
                self.progress.stop()
                self.progress.pack_forget()
                self.analyze_btn.config(state="normal")
                self.choose_btn.config(state="normal")
        
        threading.Thread(target=analyze, daemon=True).start()


def main():
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()