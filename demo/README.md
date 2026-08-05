# Mammo-CoCoOp Web Demo

An interactive, web-based demonstration for **Mammo-CoCoOp**, a contrastive vision-language model tailored for mammogram analysis. This demo allows users to evaluate the model's capabilities in finding specific breast abnormalities (like Masses, Calcifications, or Malignancy) using Zero-Shot Prediction (via CoCoOp's Meta-Net) and Linear Probing. 

The application features an intuitive UI that generates **Grad-CAM heatmaps** to visualize exactly where the model is focusing, alongside ground-truth bounding boxes for accurate performance evaluation.

## Features

- **Mammo-CoCoOp (Zero-Shot)**: Leverage the power of Mammo-CoCoOp's dynamically learned, image-conditional text prompts to perform highly accurate zero-shot classification.
- **Linear Probe**: Evaluate task-specific performance using pre-trained linear classifiers fine-tuned on Mammo-CLIP's frozen image embeddings.
- **Grad-CAM Interpretability**: Automatically generates heatmaps overlaid on the original mammograms, showing the regions that most strongly influenced the model's prediction.
- **Ground Truth Validation**: Displays ground-truth bounding boxes perfectly aligned with the model's heatmaps, utilizing dynamic cropping and coordinate mapping from the VinDr metadata.
- **Dynamic Preprocessing**: Automatically isolates the Breast Region of Interest (ROI) and normalizes uploaded or sampled images before inference.

## Project Structure

```text
demo/
├── app.py                  # Main FastAPI backend server
├── core/                   # Backend logic
│   ├── config.py           # Configuration and Label mappings
│   ├── inference.py        # Prediction logic for Zero-shot and Linear Probe
│   ├── model_manager.py    # Model loading and caching utilities
│   ├── preprocessor.py     # Image ROI extraction and normalization
│   └── visualization.py    # Grad-CAM and Bounding Box rendering
├── static/
│   ├── css/style.css       # Custom UI styling
│   ├── js/app.js           # Frontend logic and API integration
│   └── sample_images/      # Pre-loaded VinDr mammogram samples
├── templates/
│   └── index.html          # Main application HTML interface
├── requirements.txt        # Python dependencies
└── setup_demo.sh           # Automated setup script (creates dirs & downloads checkpoints)
```

## Setup & Installation

**1. Create a Conda Environment (Optional but Recommended)**
```bash
conda create -n mammoclip python=3.12
conda activate mammoclip
```

**2. Install Dependencies**
```bash
cd demo
pip install -r requirements.txt
```

**3. Run Setup Script**
The demo requires certain directory structures, checkpoints, and datasets. You can run the setup script to initialize the environment:
```bash
chmod +x setup_demo.sh
./setup_demo.sh
```

## Running the Demo

To start the local web server, use `uvicorn` with the `--reload` flag for development:

```bash
cd demo
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Once the server starts, open your web browser and navigate to:
**http://localhost:8000**

## Usage Guide

1. **Select a Setting**: Choose between **Zero-shot** (Mammo-CoCoOp) or **Linear Probe**.
2. **Select Target Label**: Choose the clinical finding you want the model to look for (e.g., Mass, Calcification, Malignancy).
3. **Select an Image**: Pick one of the provided sample images from the dropdown menu, or upload your own mammogram.
4. **Run Analysis**: Click the "Run Analysis" button. The model will process the image, predict the likelihood of the selected finding, and generate an interpretable Grad-CAM heatmap highlighting the suspected region!

## Technical Notes
- The demo dynamically crops the breast ROI (removing black background) using intensity variance thresholds to ensure optimal model accuracy.
- Bounding boxes are drawn by querying `vindr_detection_v1_folds.csv` using the image's original file name. 
