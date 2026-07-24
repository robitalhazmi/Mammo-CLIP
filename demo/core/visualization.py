import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
import base64
from io import BytesIO
from PIL import Image
import logging
import pandas as pd
import os
import ast
import io
from .config import IMAGE_SIZE

log = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for Mammo-CLIP models.
    
    Works with both BreastClip (CLIP) and BreastClipClassifier models
    by hooking into the image encoder's last conv layer.
    """
    def __init__(self, model, is_clip_model=True):
        self.model = model
        self.is_clip_model = is_clip_model
        self.gradients = None
        self.activations = None
        self._hooks = []
        
        # Find the target layer for hooks
        # Both model types have image_encoder which is a custom EfficientNet
        # The last conv layer is _conv_head
        target_layer = None
        if hasattr(model, 'image_encoder'):
            if hasattr(model.image_encoder, '_conv_head'):
                target_layer = model.image_encoder._conv_head
            elif hasattr(model.image_encoder, '_blocks'):
                target_layer = model.image_encoder._blocks[-1]
            elif hasattr(model.image_encoder, 'model'):
                # EfficientNet_Mammo wraps model inside .model
                if hasattr(model.image_encoder.model, 'conv_head'):
                    target_layer = model.image_encoder.model.conv_head
        
        self.target_layer = target_layer
        
        if self.target_layer is None:
            log.warning("Could not find target layer for Grad-CAM")
            return
            
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
    def generate(self, image_tensor, target_class, device):
        """Generate Grad-CAM heatmap.
        
        For CLIP models: uses encode_image + image_projection
        For Classifier models: uses forward() directly
        """
        self.model.zero_grad()
        # DO NOT require grad on image_tensor to save massive memory and avoid swapping
        image_tensor = image_tensor.to(device)
        
        # Temporarily enable requires_grad on the target layer to build the graph
        orig_requires_grad = {}
        if hasattr(self, 'target_layer') and self.target_layer is not None:
            for name, param in self.target_layer.named_parameters():
                orig_requires_grad[name] = param.requires_grad
                param.requires_grad = True

        try:
            if self.is_clip_model:
                # CLIP path: encode image and project
                img_features = self.model.encode_image(image_tensor)
                if hasattr(self.model, 'projection') and self.model.projection:
                    img_emb = self.model.image_projection(img_features)
                else:
                    img_emb = img_features
                # Use the target_class-th dimension as the score
                if img_emb.shape[1] > target_class:
                    score = img_emb[0, target_class]
                else:
                    score = img_emb.sum()
            else:
                # Classifier path
                output = self.model(image_tensor)
                if output.dim() > 1 and output.size(1) > 1:
                    score = output[0, target_class]
                else:
                    score = output.squeeze()
            
            score.backward(retain_graph=False)
        finally:
            # Restore original requires_grad state
            if hasattr(self, 'target_layer') and self.target_layer is not None:
                for name, param in self.target_layer.named_parameters():
                    param.requires_grad = orig_requires_grad.get(name, False)
        
        if self.gradients is None or self.activations is None:
            log.warning("Grad-CAM: No gradients/activations captured")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))
        
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Global average pool the gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Resize to input image dimensions
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        
        # Normalize
        if np.max(cam) > 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
        
        return cam


def generate_heatmap_overlay(original_image, heatmap, alpha=0.4):
    """Create a colored heatmap overlay on the original image."""
    # Apply colormap (inferno for medical imaging aesthetic)
    heatmap_colored = cm.inferno(heatmap)[:, :, :3] * 255.0
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    # Convert grayscale original to RGB if needed
    if len(original_image.shape) == 2:
        original_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_color = original_image
    
    # Resize heatmap to match original if needed
    if heatmap_colored.shape[:2] != original_color.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, 
                                      (original_color.shape[1], original_color.shape[0]))
    
    overlay = cv2.addWeighted(heatmap_colored, alpha, original_color, 1 - alpha, 0)
    return overlay


def image_to_base64(image_array):
    """Convert numpy array to base64-encoded PNG string."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image_array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def create_visualization(model, image_tensor, original_image, label, prediction_idx, device, is_clip_model=True, sample_image=None):
    """Generate complete visualization with Grad-CAM."""
    # Convert original image to displayable format
    orig_display = original_image.copy()
    if orig_display.dtype != np.uint8:
        orig_display = orig_display.astype(np.uint8)
        
    # Draw Ground Truth Bounding Box if sample_image is provided
    if sample_image:
        try:
            # The CSV path relative to this file
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "codebase", "data_csv", "vindr_detection_v1_folds.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Filter for the image
                matches = df[df['image_id'] == sample_image]
                if not matches.empty:
                    # Get target label in Title Case (e.g., 'mass' -> 'Mass')
                    target_category = label.title()
                    for _, row in matches.iterrows():
                        try:
                            categories = ast.literal_eval(row['finding_categories'])
                            if target_category in categories or (target_category == 'Calcification' and 'Suspicious Calcification' in categories):
                                # Found a matching bounding box
                                if pd.notna(row['xmin']) and pd.notna(row['ymin']) and pd.notna(row['xmax']) and pd.notna(row['ymax']):
                                    orig_h, orig_w = row['height'], row['width']
                                    if orig_h > 0 and orig_w > 0:
                                        # The CSV has resized coordinates scaled to IMAGE_SIZE (H=1520, W=912)
                                        target_h, target_w = IMAGE_SIZE[0], IMAGE_SIZE[1]
                                        
                                        # Normalize coordinates
                                        n_xmin = row['resized_xmin'] / target_w
                                        n_ymin = row['resized_ymin'] / target_h
                                        n_xmax = row['resized_xmax'] / target_w
                                        n_ymax = row['resized_ymax'] / target_h
                                        
                                        # Scale to orig_display dimensions
                                        disp_h, disp_w = orig_display.shape[:2]
                                        box_x1 = int(n_xmin * disp_w)
                                        box_y1 = int(n_ymin * disp_h)
                                        box_x2 = int(n_xmax * disp_w)
                                        box_y2 = int(n_ymax * disp_h)
                                        
                                        # Draw green box
                                        if len(orig_display.shape) == 2:
                                            orig_display = cv2.cvtColor(orig_display, cv2.COLOR_GRAY2RGB)
                                        cv2.rectangle(orig_display, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 3)
                        except Exception as e:
                            log.error(f"Error parsing bounding box: {e}")
        except Exception as e:
            log.error(f"Failed to load CSV for bounding box: {e}")
    
    
    try:
        gradcam = GradCAM(model, is_clip_model=is_clip_model)
        heatmap = gradcam.generate(image_tensor, prediction_idx, device)
        gradcam.cleanup()
        
        overlay = generate_heatmap_overlay(orig_display, heatmap)
        
        # Create standalone heatmap image
        heatmap_vis = cm.inferno(heatmap)[:, :, :3]
        heatmap_vis = (heatmap_vis * 255).astype(np.uint8)
        
        return {
            "original": image_to_base64(orig_display),
            "heatmap": image_to_base64(heatmap_vis),
            "overlay": image_to_base64(overlay)
        }
    except Exception as e:
        log.error(f"Failed to generate Grad-CAM: {e}", exc_info=True)
        return {
            "original": image_to_base64(orig_display),
            "heatmap": "",
            "overlay": ""
        }
