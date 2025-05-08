Bird Species Classification with Light‑Weight MobileNetV2
### *Neelima Pulipati*

---

## Motivation
Identifying bird species from images is a classic fine‑grained recognition problem. I chose this topic because:
* It blends computer vision with biodiversity conservation—an area I’m passionate about.
* The dataset is large and diverse (220 categories), providing a realistic test‑bed for modern CNNs.
* Deploying a **compact** model such as MobileNetV2 encourages thinking about edge devices and latency‑constrained applications (e.g., in‑field bird‑watching apps).

## Multimodal Learning Context
Although this project focuses on the *visual* modality, multimodal learning research increasingly fuses images with **audio** (birdsong) and **text** (field guides). Recent work—from CLIP‑style cross‑modal contrastive learning to joint image‑audio transformers—shows that complementary signals boost accuracy and robustness. My project connects to this trajectory by laying a strong vision baseline that could later be fused with bird‑call spectrograms or textual metadata.

## What I Learned
* **Data pipelines**: Efficiently loading >10 k images with on‑the‑fly resizing in PyTorch.
* **Model surgery**: Trimming MobileNetV2 while preserving performance—understanding expansion ratios and depthwise convolutions.
* **Training tricks**: Early stopping via validation accuracy and adaptive learning rates.
* **Error analysis**: Visual inspection of mis‑classifications revealed that many errors occur between visually similar warblers—hinting at the value of additional modalities.

## Code Walkthrough & Demos
The next notebook cells demonstrate:
1. Loading and visualising samples
2. Preparing `BirdsDataset`/`DataLoader`
3. Implementing a tiny MobileNetV2 variant
4. Training for 100 epochs with live progress bars
5. Evaluating on the test set

Feel free to run each cell or jump to specific sections using the Jupyter outline.

## Reflections
* **Surprises**: A 1.3 M‑parameter model achieved >80 % validation accuracy—smaller than expected for 220 classes!
* **Improvements**:
  * *Data augmentation*: random crops, flips, and color jitter could push accuracy higher.
  * *Multimodal fusion*: integrate birdsong spectrogram CNN features.
  * *Knowledge distillation*: compress an even larger teacher network into the mobile‑friendly student.

---
