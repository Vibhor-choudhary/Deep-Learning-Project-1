# AksharCoach — Handwritten Hindi Character Recognition

An AI project for **CSM422 Deep Learning** that recognizes isolated handwritten Hindi (Devanagari) characters and numbers.

**Live Web Demo:** [https://deep-learning-project-1.vercel.app](https://deep-learning-project-1.vercel.app)

---

## 1. What is it?
AksharCoach is a machine learning tool that reads handwritten Hindi letters and digits. You draw or upload a character, and the system recognizes what letter it is and displays its confidence percentage.

## 2. Why did we build it?
To answer a simple question: **Can a Convolutional Neural Network (CNN) read messy handwriting better than a basic Neural Network (MLP)?**

When training data is small, standard neural networks struggle with handwriting because people write letters with different slants, sizes, and thicknesses. We tested both models on the official UCI Devanagari benchmark to compare their accuracy.

## 3. How does it work?
1. **Input:** The user uploads an image or draws a character on the interactive canvas.
2. **Preprocessing:** The image is cleaned, centered, inverted (white stroke on dark background), and resized to a standard 32×32 pixel square.
3. **CNN Recognition:** The neural network scans the image using small 3×3 filters that detect lines, curves, and loops.
4. **Decision:** The model outputs the top predicted character. If confidence is below 20%, it marks the result as **Uncertain** rather than making a wild guess.
5. **Batch Mode:** For pages with multiple spaced characters, an automatic segmenter strips table grid lines and classifies each character individually.

---

## 4. Key Results

Evaluated on the official test set (13,800 unseen handwritten images):

| Model | Clean Test Accuracy | Distorted (Rotated/Blurry) Accuracy | Why? |
| :--- | :---: | :---: | :--- |
| **Compact CNN** | **91.80%** | **87.07%** | **Wins:** Understands 2D stroke shapes and curves. |
| **Basic Model (MLP)** | **75.14%** | **66.94%** | **Loses:** Flattens image into scattered dots, losing stroke connections. |

---

## 5. Scope & Limitations (Viva Reference)

* **Supported:** 46 single characters:
  * 36 Consonants (**क to ज्ञ**)
  * 10 Digits (**० to ९**)
* **Not Supported:** Full sentences, words connected by a top line (shirorekha), or vowels (स्वर: अ, आ, इ...).
* **Confidence Safety:** Any stroke below 20% confidence is safely rejected as *Uncertain*.

---

## 6. How to Run Locally

```bash
cd AksharCoach
python3 app.py
```
*(Or simply `./app.py`)*

Then open your browser at `http://127.0.0.1:7860`.
