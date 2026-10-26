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
python app.py
```

The Vercel deployment is a public project showcase. The interactive PyTorch/Gradio recognizer is intentionally kept local because it requires a large model runtime and the full image dataset; Vercel Functions are stateless and not a suitable host for a persistent ML inference server.
