import cv2
import numpy as np
import torch
from PIL import Image
import io
from .config import IMAGE_SIZE, MEAN, STD

def np_CountUpContinuingOnes(arr):
    return np.cumsum(arr) - np.maximum.accumulate(np.cumsum(arr) * (arr == 0))

def preprocess_image(image_bytes_or_path, is_path=False) -> tuple[torch.Tensor, np.ndarray]:
    if is_path:
        img_np = cv2.imread(image_bytes_or_path, cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            raise ValueError("Could not read image from path")
    else:
        if isinstance(image_bytes_or_path, bytes):
            img_np = np.frombuffer(image_bytes_or_path, np.uint8)
            img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Unsupported input format")

    # Breast ROI extraction
    mask = img_np > 40
    img_np = img_np * mask

    H, W = img_np.shape
    H_start, H_end = int(H * 0.1), int(H * 0.9)
    W_start, W_end = int(W * 0.1), int(W * 0.9)

    col_std = np.std(img_np[H_start:H_end, :], axis=0)
    col_mask = col_std > 10
    col_counts = np_CountUpContinuingOnes(col_mask)
    if np.max(col_counts) > 0:
        col_end = np.argmax(col_counts)
        col_start = col_end - np.max(col_counts) + 1
    else:
        col_start, col_end = 0, W

    row_std = np.std(img_np[:, W_start:W_end], axis=1)
    row_mask = row_std > 10
    row_counts = np_CountUpContinuingOnes(row_mask)
    if np.max(row_counts) > 0:
        row_end = np.argmax(row_counts)
        row_start = row_end - np.max(row_counts) + 1
    else:
        row_start, row_end = 0, H

    img_cropped = img_np[row_start:row_end, col_start:col_end]
    
    # Resize
    img_resized = cv2.resize(img_cropped, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    
    vis_img = img_resized.copy()
    
    # Normalize
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    
    # Stack to 3 channels
    img_stacked = np.stack([img_norm, img_norm, img_norm], axis=0)
    
    tensor = torch.from_numpy(img_stacked).unsqueeze(0)
    return tensor, vis_img

def load_sample_image(path) -> tuple[torch.Tensor, np.ndarray]:
    return preprocess_image(path, is_path=True)
