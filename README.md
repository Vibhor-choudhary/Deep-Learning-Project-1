# AksharCoach — Deep Learning CA Project

AksharCoach is a confidence-aware, offline-first recognizer for isolated handwritten Devanagari characters and digits. It compares an MLP with a compact CNN under limited training data and controlled image distortions.

## Results from the recorded experiment

| Model | Held-out clean accuracy | Macro F1 | Distorted accuracy |
|---|---:|---:|---:|
| MLP | 75.14% | 74.90% | 66.94% |
| CNN | 91.80% | 91.80% | 87.07% |

The complete local application is in [`AksharCoach/`](AksharCoach/). It includes training/evaluation scripts, report, viva questions, saved checkpoints, and the local Gradio interface.

## Live Web App (Vercel)

The live web application is hosted at [https://deep-learning-project-1.vercel.app](https://deep-learning-project-1.vercel.app).

It features a **100% client-side in-browser neural engine** powered by **ONNX Runtime Web (WASM)**:
- **Interactive Stylus/Mouse Drawing Canvas**: Draw any Devanagari character or digit live with instant predictions.
- **Image Upload & Presets**: Drag and drop character images or 1-click test held-out samples.
- **Multi-Character Batch Analysis**: Upload a sheet of spaced characters for automatic bounding-box segmentation and batch recognition.
- **Model Comparison**: Switch between Compact CNN (91.8% accuracy) and MLP Baseline (75.1% accuracy).
- **Confidence-Aware Decisions**: Rejection thresholding (20% for CNN, 50% for MLP) flags out-of-distribution or ambiguous strokes.
- **Preprocessed Tensor Inspection**: Visualizes the normalized 32×32 input tensor fed to the network.

## Run locally

```bash
cd AksharCoach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --per-class 100 --epochs 20
python evaluate.py
python app.py
```

