# AksharCoach — Local Application

Offline AI recognizer for isolated handwritten Hindi (Devanagari) characters and digits.

---

## 1. Quick Start

Run the application locally:

```bash
cd AksharCoach
python3 app.py
```
*(Or directly: `./app.py`)*

Then open `http://127.0.0.1:7860` in your web browser.

---

## 2. What it is & How it Works

* **Single Character Mode:** Draw on the canvas or upload a character. The app resizes it to 32×32 and classifies it among 46 Devanagari classes.
* **Batch Mode:** Upload an image containing multiple spaced characters or an alphabet chart. The system automatically strips table/grid lines, detects each character box, and runs recognition across all symbols.
* **Safety Mechanism:** If a prediction has low confidence (<20%), it is flagged as **Uncertain** instead of guessing.

---

## 3. Results Summary

* **Compact CNN (Proposed):** **91.80% Accuracy** on official test data (13,800 images).
* **Basic MLP (Baseline):** **75.14% Accuracy**.
* **Distorted Test (Rotations/Blur):** CNN remains robust at **87.07%**, while MLP collapses to **66.94%**.

---

## 4. Scope for Viva

* **Supported:** 46 classes (36 consonants **क to ज्ञ**, 10 digits **० to ९**).
* **Not Supported:** Full words, connected text with shirorekha, or vowels (अ, आ, इ...).
* **Files:**
  * `app.py`: Local Gradio web UI.
  * `evaluate.py`: Generates test metrics and confusion matrix.
  * `train.py`: Model training pipeline.
  * `REPORT.md`: Full technical academic report.
  * `VIVA_QUESTIONS.md`: 38 common viva questions & answers.
