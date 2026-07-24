import torch
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from .config import DEFAULT_PROMPTS, LABEL_INFO


def zero_shot_predict(model, tokenizer, image_tensor, label, prompts, device) -> dict:
    """Zero-shot prediction using CLIP model.
    
    Encodes image and text prompts, computes cosine similarity,
    and returns class probabilities.
    """
    with torch.no_grad():
        # Encode image
        print("Encoding image..."); img_emb = model.encode_image(image_tensor.to(device))
        # Project if model has projection head
        if hasattr(model, 'projection') and model.projection:
            img_emb = model.image_projection(img_emb)
        img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        
        # Tokenize and encode text prompts
        tokens = tokenizer(
            prompts, padding='longest', truncation=True, 
            return_tensors='pt', max_length=256
        )
        # Move token tensors to device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        print("Encoding text..."); text_emb = model.encode_text(tokens)
        if hasattr(model, 'projection') and model.projection:
            text_emb = model.text_projection(text_emb)
        text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        
        # Compute cosine similarity and softmax
        print("Computing similarity..."); cos_sim = cosine_similarity(img_emb.cpu().numpy(), text_emb.cpu().numpy())
        probs = softmax(cos_sim, axis=1)[0]
        
    print("Done prediction!"); info = LABEL_INFO[label]
    pred_idx = int(np.argmax(probs))
    
    if info["type"] == "binary":
        confidence = float(probs[1])  # Probability of positive class
    else:
        confidence = float(probs[pred_idx])
    
    predicted_class = info["classes"][pred_idx]
        
    return {
        "probabilities": probs.tolist(),
        "predicted_class": predicted_class,
        "prediction_idx": pred_idx,
        "confidence": confidence,
        "classes": info["classes"]
    }


def classifier_predict(classifier_model, image_tensor, label, device) -> dict:
    """Prediction using downstream classifier (LP or FT).
    
    Passes image tensor through the classifier and returns
    probabilities for each class.
    """
    with torch.no_grad():
        logits = classifier_model(image_tensor.to(device))
        
    print("Done prediction!"); info = LABEL_INFO[label]
    
    if info["type"] == "binary":
        prob_pos = torch.sigmoid(logits).item()
        pred_idx = 1 if prob_pos >= 0.5 else 0
        predicted_class = info["classes"][pred_idx]
        confidence = prob_pos if pred_idx == 1 else 1.0 - prob_pos
        prob_list = [1.0 - prob_pos, prob_pos]
    else:
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_class = info["classes"][pred_idx]
        prob_list = probs.tolist()
        
    return {
        "probabilities": prob_list,
        "predicted_class": predicted_class,
        "prediction_idx": pred_idx,
        "confidence": confidence,
        "classes": info["classes"]
    }


def get_default_prompts(label) -> list[str]:
    """Return default zero-shot prompts for a given label."""
    return DEFAULT_PROMPTS.get(label, [])
