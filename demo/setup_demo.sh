#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Mammo-CLIP Demo Setup"
echo "============================================"

# -----------------------------------------------
# 1. Create directory structure
# -----------------------------------------------
echo ""
echo "[1/5] Creating directory structure..."
mkdir -p core
mkdir -p static/{css,js,fonts,sample_images}
mkdir -p templates
mkdir -p checkpoints/{pretrained,downstream/classifier}
echo "  Done."

# -----------------------------------------------
# 2. Setup Kaggle credentials
# -----------------------------------------------
echo ""
echo "[2/5] Setting up Kaggle credentials..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "  WARNING: ~/.kaggle/kaggle.json not found."
    echo "  Please download your kaggle.json and place it in ~/.kaggle/ to download the VinDr dataset."
else
    chmod 600 ~/.kaggle/kaggle.json
    echo "  Kaggle credentials configured."
fi

# -----------------------------------------------
# 3. Download pretrained Mammo-CLIP checkpoints
# -----------------------------------------------
echo ""
echo "[3/5] Downloading pretrained Mammo-CLIP checkpoints from HuggingFace..."

# EN-B5 checkpoint
B5_CKPT="checkpoints/pretrained/b5-model-best-epoch-7.tar"
if [ ! -f "$B5_CKPT" ]; then
    echo "  Downloading EN-B5 checkpoint..."
    wget -q --show-progress -O "$B5_CKPT" \
        "https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Pre-trained-checkpoints/b5-model-best-epoch-7.tar"
    echo "  EN-B5 checkpoint downloaded."
else
    echo "  EN-B5 checkpoint already exists, skipping."
fi

# EN-B2 checkpoint
B2_CKPT="checkpoints/pretrained/b2-model-best-epoch-10.tar"
if [ ! -f "$B2_CKPT" ]; then
    echo "  Downloading EN-B2 checkpoint..."
    wget -q --show-progress -O "$B2_CKPT" \
        "https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar"
    echo "  EN-B2 checkpoint downloaded."
else
    echo "  EN-B2 checkpoint already exists, skipping."
fi

# -----------------------------------------------
# 4. Download downstream classifier checkpoints
# -----------------------------------------------
echo ""
echo "[4/5] Downloading downstream classifier checkpoints..."

DOWNSTREAM_DIR="checkpoints/downstream/classifier"

# The downstream checkpoints are in HuggingFace under:
# https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Downstream-checkpoints
# They include fold 0 checkpoints for LP and FT with EN-B5

#echo "  Downloading Downstream checkpoints zip (this may take a while)..."
ZIP_URL="https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Downstream-checkpoints/Downstream_evalualtion_b5_fold0.zip"
ZIP_PATH="$DOWNSTREAM_DIR/Downstream_evalualtion_b5_fold0.zip"

if [ ! -d "$DOWNSTREAM_DIR/mass" ] || [ -z "$(ls -A $DOWNSTREAM_DIR/mass 2>/dev/null)" ]; then
    echo "  Downloading Downstream checkpoints zip (this may take a while)..."
    wget -q --show-progress -O "$ZIP_PATH" "$ZIP_URL"
    echo "  Extracting..."
    unzip -q -o "$ZIP_PATH" -d "$DOWNSTREAM_DIR/"
    unzip -q -o "$DOWNSTREAM_DIR/classification/Models.zip" -d "$DOWNSTREAM_DIR/"
    
    mkdir -p "$DOWNSTREAM_DIR/mass"
    mkdir -p "$DOWNSTREAM_DIR/calcification"
    mkdir -p "$DOWNSTREAM_DIR/density"
    mkdir -p "$DOWNSTREAM_DIR/cancer"
    
    cp "$DOWNSTREAM_DIR/Models/Classifier/linear_probe/mass/"*.pth "$DOWNSTREAM_DIR/mass/" 2>/dev/null || true
    cp "$DOWNSTREAM_DIR/Models/Classifier/linear_probe/calcification/"*.pth "$DOWNSTREAM_DIR/calcification/" 2>/dev/null || true
    cp "$DOWNSTREAM_DIR/Models/Classifier/linear_probe/density/"*.pth "$DOWNSTREAM_DIR/density/" 2>/dev/null || true
    cp "$DOWNSTREAM_DIR/Models/Classifier/linear_probe/cancer/"*.pth "$DOWNSTREAM_DIR/cancer/" 2>/dev/null || true
    
    echo "  Cleaning up temporary zips..."
    rm -rf "$ZIP_PATH" "$DOWNSTREAM_DIR/classification" "$DOWNSTREAM_DIR/Models" "$DOWNSTREAM_DIR/detection" "$DOWNSTREAM_DIR/__MACOSX"
else
    echo "  Downstream checkpoints already exist, skipping."
fi

echo "  Downstream checkpoints done."

# -----------------------------------------------
# 5. Download sample VinDr images from Kaggle
# -----------------------------------------------
echo ""
echo "[5/5] Downloading sample VinDr mammogram images..."

SAMPLE_DIR="static/sample_images"

if [ -z "$(ls -A $SAMPLE_DIR 2>/dev/null | grep -v samples_metadata.json)" ]; then
    if command -v kaggle &> /dev/null; then
        echo "  Downloading VinDr PNG dataset from Kaggle..."
        TEMP_DIR=$(mktemp -d)
        
        # Download just a small portion. The Kaggle dataset is organized by study_id/image_id.png
        kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png \
            -p "$TEMP_DIR" --unzip 2>/dev/null || true
        
        # Copy first 10 images to sample_images
        find "$TEMP_DIR" -name "*.png" -type f | head -n 10 | while read img; do
            cp "$img" "$SAMPLE_DIR/$(basename "$img")"
        done
        
        rm -rf "$TEMP_DIR"
        echo "  Sample images downloaded."
    else
        echo "  Kaggle CLI not found. Install with: pip install kaggle"
        echo "  Then run this script again, or manually place PNG mammograms in $SAMPLE_DIR"
        echo "  You can also download from: https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png"
    fi
else
    echo "  Sample images already exist, skipping."
fi

# -----------------------------------------------
# Create samples metadata
# -----------------------------------------------
cat > "$SAMPLE_DIR/samples_metadata.json" << 'EOF'
{
    "note": "Place preprocessed VinDr mammogram PNG images in this directory. Images should be grayscale mammograms. Ground truth labels can be found in the VinDr annotations CSV files in src/codebase/data_csv/."
}
EOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Pretrained checkpoints:"
ls -lh checkpoints/pretrained/*.tar 2>/dev/null || echo "    (none found)"
echo ""
echo "  Downstream checkpoints:"
find checkpoints/downstream -name "*.pth" -exec echo "    {}" \; 2>/dev/null || echo "    (none found)"
echo ""
echo "  Sample images:"
ls "$SAMPLE_DIR"/*.png 2>/dev/null | wc -l | xargs -I{} echo "    {} images"
echo ""
echo "  To start the demo:"
echo "    cd demo"
echo "    pip install -r requirements.txt"
echo "    python app.py"
echo "============================================"
