# AksharCoach

An offline-first course project for isolated handwritten Devanagari characters/digits. It compares an MLP baseline to a compact CNN with limited training data, measures clean and distorted held-out test performance, and uses a validation-selected uncertainty threshold. It is **not full Hindi word or paragraph OCR**.

## Quick start (macOS/Linux)

```bash
cd /Users/vibhor/Documents/ChatGPT/DLP/AksharCoach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --per-class 100 --epochs 20
python evaluate.py
python -m pytest -q
python app.py
```

If port 7860 is occupied on your computer, use another local port: `AKSHARCOACH_PORT=8089 python app.py`.

The official UCI data was placed under `data/DevanagariHandwrittenCharacterDataset/` for this run. If absent, download and extract the URL in `data/README.md`; do not replace the official Test folder with augmented images.

## What the experiment does

Training samples are balanced per folder with seed 42. Fifteen percent of that subset is validation; the official Test folder is evaluated only after checkpoint selection. Training augmentation is small rotation/translation. The robustness test applies a controlled family of small affine motion, blur, and Gaussian pixel noise to held-out copies. Results and figures are generated in `outputs/`; they are never hand-written into the documentation.

## Two useful modes

- **Recognize:** one tightly cropped handwritten character.
- **Analyse a page of separate characters:** detects each separately spaced handwritten character on a plain, light background, then classifies every detected crop. Try the bundled 12-character demo first. It is intentionally not presented as printed-chart, word, or paragraph OCR.

`app.py` binds Gradio only to `127.0.0.1`, has no sharing link, and requires trained checkpoints plus evaluation outputs.
