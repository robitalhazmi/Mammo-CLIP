# Mammo-CLIP & Mammo-FActOR: Experiment Reproduction Guide

This document serves as a comprehensive reference guide to reimplement the experiments presented in the paper *Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography*. The guide is structured specifically for generating LLM prompts to write PyTorch training, evaluation, and data processing scripts that replicate the paper's numerical results.

---

## 1. Model Architectures

### 1.1 Mammo-CLIP (Vision-Language Model)
The core model aligns image and text representations in a joint embedding space using Multi-View Supervision (MVS).

*   **Text Encoder ($f^T$):** BioClinicalBERT.
*   **Image Encoder ($f^I$):** EfficientNet-B2 (EN-B2) or EfficientNet-B5 (EN-B5).
    *   **Initialization:** ImageNet pre-trained weights.
*   **Projection:** Both encoders project outputs to the same dimensions, followed by $l_2$ normalization to yield representations $\mathcal{Z}^I$ and $\mathcal{Z}^T$.

### 1.2 Mammo-FActOR (Feature Attribution)
A lightweight module to map visual representations to textual attributes at the sentence level (without bounding boxes).

*   **Network:** Multi-Layer Perceptron (MLP).
*   **Architecture:** `Linear Layer` $ightarrow$ `ReLU` $ightarrow$ `Linear Layer`.
*   **Input:** Output of the frozen image encoder $f^I(x^I_i) \in \mathbb{R}^{C 	imes H 	imes W}$ and text representation $t^k_i \in \mathbb{R}^d$.
*   **Output:** Similarity weights mapped across channels $\in \mathbb{R}^C$.

---

## 2. Datasets & Pre-processing

### 2.1 Datasets
1.  **UPMC Dataset:** 
    *   13,829 patient-report pairs.
    *   25,355 screening mammograms (BI-RADS 0-2).
    *   Views: Includes CC and/or MLO views.
    *   Split: 80% train / 20% validation.
2.  **VinDr-Mammo (Public):**
    *   5,000 exams, 20,000 images (4 views per exam).
    *   Annotations: Breast-level BI-RADS (1-5), Density (A-D), bounding boxes for attributes (mass, calcifications, etc.).
    *   Split: Original paper's official train-test split.
3.  **RSNA-Mammo (Public):**
    *   11,913 patients, 486 cancer cases.
    *   Split: 70% train / 20% validation / 10% test.

### 2.2 Image Pre-processing
A crucial step for high-resolution semantic extraction. Apply these exactly:
1.  **Rule-based ROI extraction:** Isolate the breast Region of Interest (ROI).
2.  **Thresholding:** Set all pixel values `< 40` to `0`.
3.  **Background Cropping:** Eliminate consistently identical rows and columns (which denote background).
4.  **Resizing:** The resulting images (average aspect ratio 1:1.6 to 2) MUST be resized exactly to **$1520 	imes 912$**.

### 2.3 Data Augmentation
*   **Image Instance Augmentation:**
    *   Affine transformations: Rotation up to 20°, min translation of 0.1%, scaling factor `[0.8, 1.2]`, shearing by 20°.
    *   Elastic transformations: $lpha=10$, $\sigma=5$.
*   **Text Instance Augmentation:**
    *   Sentence swapping.
    *   Back-translation (Italian to English).
*   **Dataset Augmentation (Report Synthesis):**
    *   For external datasets without full reports, generate synthetic reports using prompts matching attributes (mass, calc), values (positive/negative), subtypes (suspicious, obscured), laterality (right/left), depth (anterior, mid, posterior), and position (upper, lower).

---

## 3. Training Configurations

### 3.1 Mammo-CLIP Pre-training
*   **Loss Function:** Multi-View Supervision (MVS) loss.
    *   Contrastive loss applies to original and augmented pairs.
    *   **Crucial detail:** The contrastive loss for the original and augmented *text* representation pair ($\mathcal{L}(\mathcal{Z}^T, 	ilde{\mathcal{Z}}^T)$) is down-weighted by **0.5**.
*   **Optimizer:** AdamW
*   **Initial Learning Rate:** `5e-5`
*   **Weight Decay:** `1e-4`
*   **Scheduler:** Cosine-annealing with warm-up for 1 epoch.
*   **Epochs:** 10
*   **Precision:** Mixed-precision training.

### 3.2 Mammo-FActOR Training
*   **Loss Function:** Contrastive loss between images with and without the targeted attributes (Eq 3).
*   **Optimizer:** standard (implied Adam).
*   **Learning Rate:** `1e-4` (0.0001)
*   **Temperature ($	au$):** 0.007
*   **Epochs:** 20

---

## 4. Downstream Evaluation Protocols

### 4.1 Classification (VinDr and RSNA)
*   **Metrics:** AUC (for Mass, Calcification, Cancer), Accuracy (for Density).
*   **Zero-shot (ZS):** Freeze both encoders. Use prompts: `{No (E), (E)}` where E is the condition (e.g., mass, calcification, cancer). 
*   **Linear Probe (LP):** Freeze both encoders. Train a linear classifier on 10%, 50%, or 100% of training data.
*   **Fine-tuning (FT):** Jointly fine-tune the image encoder and a linear classifier. Tested on 10%, 50%, and 100% of data.

### 4.2 Supervised Localization (VinDr)
*   **Architecture:** RetinaNet
*   **Hyperparameters:** Focal loss parameters $lpha=0.6$, $\gamma=2.0$.
*   **Thresholds:** Detector confidence threshold = `0.05`, IoU metric = `0.5`.
*   **Metric:** mAP.
*   **Method:** Fine-tune image encoder + RetinaNet using 10%, 50%, 100% data.

### 4.3 Weakly Supervised Localization (Mammo-FActOR)
*   **Metric:** mAP calculated by extracting heatmaps.
*   **Thresholding:** Extract isolated regions where pixel values > 95% quantile of the heatmap distribution.
*   **Evaluation:** Measure against ground-truth bounding boxes using $T(IoU) = 0.25$ and $T(IoU) = 0.50$.

---

## 5. Target Numerical Benchmarks (For Validation)

*To consider your replication successful, your evaluation scripts should approximate the following key results from the paper:*

**Classification - VinDr (Mammo-CLIP w/ EN-B5 trained on UPMC+VinDr):**
*   **Calcification (AUC):** ZS = 0.62 | LP (10%) = 0.93 | LP (100%) = 0.96 | FT (100%) = 0.98
*   **Mass (AUC):** ZS = 0.76 | LP (10%) = 0.78 | LP (100%) = 0.86 | FT (100%) = 0.88
*   **Density (Accuracy):** ZS = 0.15 | LP (10%) = 0.86 | LP (100%) = 0.88

**Classification - RSNA Cancer (Mammo-CLIP w/ EN-B5 trained on UPMC+VinDr):**
*   **Malignancy (AUC):** ZS = 0.60 | FT (10%) = 0.85 | FT (100%) = 0.91 | LP (100%) = 0.79

**Localization - VinDr (mAP at IoU=0.5):**
*   **Calcification:** FT (10%) = 0.10 | FT (100%) = 0.25
*   **Mass:** FT (10%) = 0.43 | FT (100%) = 0.58

---

## 6. Suggested LLM Implementation Prompts

You can use the following prompts iteratively with a coding LLM (like GPT-4 or Claude 3) to generate the replication codebase:

**Prompt 1: Data Pre-processing**
> "Write a PyTorch `Dataset` class for the VinDr and RSNA mammography datasets. Include a data pre-processing pipeline using OpenCV/PIL that applies a rule-based ROI extraction to isolate the breast, sets all pixel values < 40 to 0, crops out consistently identical rows/columns (background), and resizes the image to exactly 1520x912. Also, implement torchvision transforms for affine (rotation up to 20, min translation 0.1%, scaling 0.8-1.2, shear 20) and elastic transformations (alpha=10, sigma=5)."

**Prompt 2: Mammo-CLIP Architecture and MVS Loss**
> "Implement the Mammo-CLIP architecture in PyTorch. Use `BioClinicalBERT` from HuggingFace as the text encoder and `EfficientNet-B5` from `torchvision` (ImageNet pre-trained) as the image encoder. Add linear projection heads to both to map them to a joint embedding space. Then, implement the Multi-View Supervision (MVS) contrastive loss function handling original and augmented (image, text) pairs. Ensure the loss applied to the original and augmented text pair representations is down-weighted by 0.5."

**Prompt 3: Pre-training Loop**
> "Write the PyTorch training loop for Mammo-CLIP. Configure `AdamW` optimizer with an initial learning rate of 5e-5 and weight decay of 1e-4. Implement mixed-precision training using `torch.cuda.amp`. Add a cosine-annealing learning rate scheduler with a 1-epoch warm-up. The training should run for 10 epochs."

**Prompt 4: Mammo-FActOR Module**
> "Implement the Mammo-FActOR PyTorch module. It should take the frozen output of an EfficientNet-B5 image encoder and a sentence-level text embedding as input. The network is an MLP (Linear -> ReLU -> Linear). Implement the contrastive loss function designed to align sentences containing a specific attribute to the channels of the image encoder. Set the temperature tau to 0.007 and prepare an Adam optimizer with a learning rate of 0.0001."

**Prompt 5: Downstream Fine-Tuning (RetinaNet)**
> "Write a PyTorch evaluation script to fine-tune the image encoder of Mammo-CLIP for object detection using `RetinaNet`. Set focal loss parameters to alpha=0.6 and gamma=2.0. Write an evaluation function calculating mAP with an IoU threshold of 0.5 and detector confidence threshold of 0.05."
