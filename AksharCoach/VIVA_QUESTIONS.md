# Viva questions, answers, and a two-minute demo

## Likely questions

1. **What problem does AksharCoach solve?** It classifies one isolated handwritten Devanagari digit or character from an image.
2. **Who could use it?** Students or demonstrators exploring character recognition, not users needing document OCR.
3. **What is experimentally different?** It compares MLP/CNN under the same limited budget, tests distortions, and may reject low-score predictions.
4. **Main limitation?** It cannot segment or read full Hindi words, lines, or paragraphs.
5. **What is an artificial neuron?** A weighted sum of inputs plus bias passed through an activation.
6. **What are weights and biases?** Learned parameters controlling the transformation and offset.
7. **Parameter vs hyperparameter?** Parameters are learned; learning rate, batch size, epochs and architecture width are set by us.
8. **Why ReLU?** It adds nonlinearity and is simple to optimize.
9. **Why nonlinearity?** Stacked linear layers would collapse to one linear mapping.
10. **What is multiclass classification?** Selecting one of 46 possible classes.
11. **What is a logit?** An unnormalized score output before softmax.
12. **What is softmax?** A transformation of logits into positive scores that sum to one.
13. **Why is a softmax score not a calibrated probability?** Confidence can be overconfident unless calibration is separately tested.
14. **Why does the MLP use 1,024 inputs?** 32×32 grayscale pixels flatten to 1,024 values.
15. **What is the batch input shape?** `[batch_size, 1, 32, 32]`.
16. **What is feedforward computation?** Input passes layer by layer to logits without changing weights.
17. **What does convolution do?** Small filters scan local stroke patterns and create feature maps.
18. **What is a feature map?** A filter's spatial response over the image.
19. **What does pooling do?** It reduces feature-map width/height while retaining strong local responses.
20. **CNN dimensions after pool one?** `32×16×16`.
21. **After pool two and flatten?** `64×8×8`, then 4,096 values.
22. **Why 46 output neurons?** One logit per dataset class.
23. **What is cross-entropy loss?** It penalizes low score on the correct class.
24. **What is backpropagation?** Computing loss gradients through layers so parameters can be updated.
25. **What is gradient descent/Adam doing?** It changes weights in a loss-reducing direction; Adam adapts update scaling.
26. **What is learning rate?** The update step size.
27. **What is a mini-batch?** A small group of examples used for one update.
28. **What is an epoch?** One pass through the selected training examples.
29. **Training vs inference?** Training updates weights; inference uses fixed weights to predict.
30. **Why separate validation and test sets?** Validation selects checkpoints/thresholds; test estimates final generalization without tuning leakage.
31. **Why might CNN beat MLP?** It preserves local spatial stroke structure instead of flattening it immediately.
32. **What does macro F1 add?** It weights every class equally, exposing classes hidden by an overall average.
33. **What does a confusion matrix show?** Which true classes are commonly predicted as which other classes.
34. **What does augmentation do?** It exposes training to plausible small variations only.
35. **How do blur/noise/rotation affect it?** They alter edges and stroke positions, which can lower recognition accuracy.
36. **How is uncertainty handled?** A validation-selected threshold rejects a top score below the threshold and asks for a clearer image.
37. **Why report coverage and accepted accuracy?** A rejector must not look good merely by rejecting almost every case.
38. **Why use the official Test directory?** It is a publisher-provided held-out partition; no distorted training copies enter it.

## Two-minute demonstration

1. Open the README and say the system recognizes one character, not full words (10 sec).
2. Run the smoke test and point out both models output `[batch,46]` (15 sec).
3. Open the local app, upload a bundled held-out image, and show top three scores (25 sec).
4. Show the evaluation tab: actual clean/distorted MLP/CNN metrics, curve and confusion matrix (40 sec).
5. Upload a deliberately unclear/cropped image to show the rejection message (20 sec).
6. Close with why the CNN has `32×16×16 → 64×8×8` spatial features and state the isolated-character limitation (10 sec).

# Quick revision sheet

- **Input:** grayscale `1×32×32`; MLP flatten = 1,024 values.
- **MLP:** `1024 → 128 → 46`; simple baseline; loses 2-D locality.
- **CNN:** `1×32×32 → Conv32 → Pool → 32×16×16 → Conv64 → Pool → 64×8×8 → 4096 → 128 → 46`.
- **Loss:** `CrossEntropy(logits, class)`; softmax is used for display at inference.
- **Update:** `θ ← θ − learning_rate × gradient`; PyTorch autograd/backprop supplies gradients.
- **Split:** train subset → validation (checkpoint and threshold); official Test → final clean and distorted evaluation.
- **Metrics:** accuracy; macro precision/recall/F1; confusion matrix; coverage and accuracy after acceptance.
- **Important observation:** quote values from `outputs/metrics.json`, never invented values. CNN is expected to retain spatial cues but the recorded experiment decides.
- **Honesty:** 32×32 isolated-character model; no calibrated confidence; no word/paragraph OCR.
