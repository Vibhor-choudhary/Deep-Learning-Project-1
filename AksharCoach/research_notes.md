# Research notes — AksharCoach

## Problem and scoped contribution

Handwritten Devanagari character recognition is an established optical-character-recognition task, not a novel claim of this course project. AksharCoach contributes a reproducible, deliberately limited-data comparison of a 1,024-input MLP and a compact CNN, plus fixed distortions and a validation-selected rejection rule. It classifies isolated characters only; it is not word or paragraph OCR.

## Dataset and attribution

The primary source is the [UCI Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/dataset/389/devanagari%2Bhandwritten%2Bcharacter%2Bdataset), donated 31 August 2016. Its page specifies 92,000 grayscale 32×32 PNG images, 46 classes with 2,000 examples each, and the publisher-supplied 85% training / 15% test partition. It lists CC BY 4.0. UCI also links the original IEEE record (document 7400041). The project preserves the official Test folder for final evaluation.

## Existing work reviewed

| Project | Stack and scope | Useful idea | Scope/limitation versus this project |
|---|---|---|---|
| [LikithSh3tty/devanagari-cnn](https://github.com/LikithSh3tty/devanagari-cnn) | PyTorch CNN, Gradio, limited-data 100/class | upload preprocessing and a CPU-focused CNN | reports a CNN application; it does not provide this project's MLP-vs-CNN controlled experiment or its exact robustness protocol. Its reported figures are its own, not ours. |
| [tanveerbedi/AksharAI-Devanagari-Handwritten-Recognition](https://github.com/tanveerbedi/AksharAI-Devanagari-Handwritten-Recognition) | TensorFlow/Keras CNN, Streamlit drawing canvas | compact 32×32 CNN UI | CNN-only, canvas-centred deployment; it points to a Kaggle copy rather than making the UCI source the primary pipeline. |
| [ayushllcode/devanagari-ocr-streamlit](https://github.com/ayushllcode/devanagari-ocr-streamlit) | Requested comparison target | — | Could not be fetched in the available network session; do not claim details, licence, results, or architecture without rechecking it. |

## Selected methods

PyTorch is used for transparent feedforward training and autograd; torchvision loads class folders and applies train-only affine augmentation. scikit-learn calculates macro metrics and the confusion matrix. Gradio provides an optional local-only interface. The MLP is `1024 → 128 → 46`; the CNN is `1×32×32 → 32×16×16 → 64×8×8 → 4096 → 128 → 46`.

## Licence and technical limitations

The UCI data needs CC BY 4.0 attribution. Repository licences must be verified directly before copying any implementation; this project contains independently written code. Softmax maxima are called **model scores**, not calibrated probabilities. Photograph preprocessing is heuristic and data budgets/epochs constrain conclusions. The stated CSM422 syllabus file was not present in the supplied workspace; alignment is therefore to the unit topics explicitly listed in the brief.
