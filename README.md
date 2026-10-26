# AksharCoach — Deep Learning CA Project

AksharCoach is a confidence-aware, offline-first recognizer for isolated handwritten Devanagari characters and digits. It compares an MLP with a compact CNN under limited training data and controlled image distortions.

## Results from the recorded experiment

| Model | Held-out clean accuracy | Macro F1 | Distorted accuracy |
|---|---:|---:|---:|
| MLP | 75.14% | 74.90% | 66.94% |
| CNN | 91.80% | 91.80% | 87.07% |

The complete local application is in [`AksharCoach/`](AksharCoach/). It includes training/evaluation scripts, report, viva questions, saved checkpoints, and the local Gradio interface.

## Run locally

```bash
cd AksharCoach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --per-class 100 --epochs 20
python evaluate.py
python3 app.py
```
Or run directly:
```bash
./app.py
```

The web deployment at [https://deep-learning-project-1.vercel.app](https://deep-learning-project-1.vercel.app) runs the identical interactive recognizer directly in the browser via ONNX Runtime WebAssembly, with full support for multi-character batch analysis, confidence-aware decision thresholding, and held-out evaluation evidence.
