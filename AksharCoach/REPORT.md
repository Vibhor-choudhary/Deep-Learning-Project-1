# AksharCoach: Confidence-Aware Devanagari Handwriting Recognition Under Limited Training Data

**Student:** ____________________  
**Registration number:** ____________________  
**Course:** CSM422 Deep Learning  
**Faculty:** ____________________  
**Date:** ____________________

## Abstract

AksharCoach investigates isolated handwritten Devanagari character classification under a deliberately constrained training budget. A flattened-image multilayer perceptron (MLP) and a compact convolutional neural network (CNN) are trained on the same balanced subset of the UCI Devanagari Handwritten Character Dataset. They are evaluated on the publisher-provided held-out test partition, both clean and after controlled mild rotation/translation, blur, and noise. A validation-selected threshold marks low-score predictions as uncertain. Measured values are written by `evaluate.py` to `outputs/metrics.json`; no numerical outcome is asserted here before that script is run.

## Introduction, motivation, and problem statement

Devanagari supports Hindi, Marathi, Sanskrit, Nepali and other languages. Converting handwritten forms or educational exercises into digital signals is useful, but a small character image is ambiguous and variable. The project asks: **does a compact CNN provide more robust isolated-character classification than a flattened-image MLP when training data is limited and images are distorted?** This is an educational classifier, not a full handwritten-word OCR system.

## Objectives and related work

The implementation (1) compares two explainable models, (2) varies a documented limited data budget, (3) preserves UCI's held-out partition, (4) measures controlled robustness, and (5) exposes uncertainty in a local interface. `research_notes.md` compares relevant accessible repositories. Established work already solves this task; the contribution is the conservative experimental combination, not a claim of first recognition.

## Dataset and preprocessing

UCI's Devanagari Handwritten Character Dataset has 92,000 grayscale 32×32 PNGs: 46 classes × 2,000 examples, split 85% Train/15% Test. It is attributed under CC BY 4.0 ([UCI source](https://archive.ics.uci.edu/dataset/389/devanagari%2Bhandwritten%2Bcharacter%2Bdataset)). ImageFolder validates class folders and converts images to one grayscale channel. Pixels become tensors and are normalized from `[0,1]` to approximately `[-1,1]`. Only training data receives small affine augmentation. Uploads receive grayscale, autocontrast, optional inversion, crop, resize, and the same normalization.

## Course alignment

An artificial neuron receives inputs, weights and a bias; its activation makes a non-linear mapping possible. The models perform multiclass classification: 46 output logits become softmax scores. Weights/biases are learned **parameters**; learning rate, batch size, seed, epochs and data budget are chosen **hyperparameters**. A batch is vectorized matrix multiplication through feedforward layers. Cross-entropy compares logits to labels; backpropagation differentiates that loss and Adam updates parameters per mini-batch. Training changes parameters, while inference only executes the learned feedforward computation.

## Architecture and dimensions

The MLP flattens `1×32×32` to 1,024 inputs, then applies `Linear(1024,128)`, ReLU, dropout and `Linear(128,46)`. It loses explicit spatial adjacency. The CNN applies `Conv(1,32,3,pad=1)`, ReLU and 2×2 max-pooling: `1×32×32 → 32×16×16`; then `Conv(32,64,3,pad=1)`, ReLU and pool: `64×8×8`; flattening yields 4,096 features, followed by `4096→128→46`. Convolution filters preserve local stroke patterns; pooling reduces spatial size. Exact trainable counts are emitted by `train.py`.

## Experimental setup and reproducibility

Seed 42 selects the same balanced number per class. The validation partition derives only from the training directory; test data is untouched until evaluation. Cross-entropy, Adam, minibatches, best-validation checkpoints, CPU fallback, loss/accuracy curves and epoch durations are used. The command sequence is in `README.md`. The default quick run uses 25 examples/class and four epochs to meet a CPU deadline; changing `--per-class` or `--epochs` changes the claim and must be recorded.

## Evaluation and results

`evaluate.py` produces accuracy, macro precision/recall/F1, per-class CSV metrics, a CNN confusion matrix, training/validation curve and clean/distorted comparison. Insert the generated values only after running it:

| Model | Clean accuracy | Clean macro F1 | Distorted accuracy | Accepted % | Accuracy when accepted |
|---|---:|---:|---:|---:|---:|
| MLP | 75.14% | 74.90% | 66.94% | 67.64% | 89.78% |
| CNN | 91.80% | 91.80% | 87.07% | 99.91% | 91.88% |

This CPU run used 100 images/class before the training/validation split (3,910 train and 690 validation images), 20 epochs, and the full 13,800-image official test set. The CNN obtained **91.80%** clean accuracy and **91.80%** macro F1, versus the MLP's **75.14%** clean accuracy and **74.90%** macro F1. On controlled distorted copies the CNN obtained **87.07%** accuracy versus the MLP's **66.94%**. The validation-only CNN threshold was 0.20; it accepted 99.91% of clean examples with 91.88% accepted accuracy. A page mode can detect separately spaced handwritten characters and passes each crop through the same classifier, but this does not turn the system into printed-chart, word, or paragraph OCR. A softmax maximum is a model score, not a calibrated probability. Interpret the result as limited-budget evidence only.

## Limitations, responsible use, conclusion

The system handles one isolated character, not Hindi words, lines or paragraphs. A small CPU budget, writer diversity, 32×32 resolution, imperfect photo preprocessing, and uncalibrated scores limit generalization. It should not drive high-stakes decisions. Future work could test larger balanced budgets, calibration, more writers and segmentation before word recognition. The project demonstrates the Units I–III learning pipeline and the Unit IV/V benefit of spatial CNN features without introducing unnecessary advanced architectures.

## References

1. UCI Machine Learning Repository, “Devanagari Handwritten Character Dataset,” 2016, CC BY 4.0: https://archive.ics.uci.edu/dataset/389/devanagari%2Bhandwritten%2Bcharacter%2Bdataset
2. UCI page's linked original paper record: IEEE document 7400041.
3. LikithSh3tty, “devanagari-cnn,” GitHub repository, accessed 2026-10-26: https://github.com/LikithSh3tty/devanagari-cnn
4. tanveerbedi, “AksharAI-Devanagari-Handwritten-Recognition,” GitHub repository, accessed 2026-10-26: https://github.com/tanveerbedi/AksharAI-Devanagari-Handwritten-Recognition
