"""
MammoCoCoOp: Wraps a pre-trained Mammo-CLIP model with CoCoOp prompt learning.

Supports both Contrastive Training (InfoNCE over batch of unique texts) and
Zero-Shot Evaluation (Cosine similarity over fixed class templates).
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .prompt_learner import MammoPromptLearner

log = logging.getLogger(__name__)

# Add codebase to path so breastclip is importable
_CODEBASE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODEBASE_DIR not in sys.path:
    sys.path.insert(0, _CODEBASE_DIR)


class MammoCoCoOp(nn.Module):
    def __init__(
        self,
        clip_ckpt_path,
        n_ctx=4,
        ctx_init="",
        meta_net_reduction=16,
    ):
        super().__init__()

        # ---- Load checkpoint ----
        print(f"Loading Mammo-CLIP checkpoint from: {clip_ckpt_path}")
        ckpt = torch.load(clip_ckpt_path, map_location="cpu")
        model_config = ckpt["config"]["model"]

        # ---- Build BreastClip model architecture ----
        text_encoder_name = model_config["text_encoder"]["name"]
        cache_dir = model_config["text_encoder"].get("cache_dir", None)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, cache_dir=cache_dir)

        from breastclip.model import build_model
        clip_model = build_model(model_config, {}, tokenizer=tokenizer)

        # Load pre-trained weights
        clip_model.load_state_dict(ckpt["model"], strict=True)
        print("Loaded pre-trained Mammo-CLIP weights successfully")

        self.clip_model = clip_model
        self.text_pooling = clip_model.text_pooling
        self.has_projection = clip_model.projection

        # ---- Freeze ALL clip_model parameters ----
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("Froze all Mammo-CLIP parameters")

        img_dim = clip_model.image_encoder.out_dim
        text_dim = clip_model.text_encoder.out_dim

        # ---- Create learnable prompt learner ----
        bert_model = clip_model.text_encoder.text_encoder  # Underlying BERT model
        self.prompt_learner = MammoPromptLearner(
            bert_model=bert_model,
            tokenizer=tokenizer,
            img_dim=img_dim,
            text_dim=text_dim,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            meta_net_reduction=meta_net_reduction,
        )

        # ---- Report trainable parameters ----
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
        print(f"Trainable parameters ({len(trainable)}):")
        for name in trainable:
            print(f"  {name}")
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable: {total_trainable:,}")

    def encode_image(self, image):
        """Returns [B, img_dim] raw features and [B, proj_dim] normalized features."""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            if self.has_projection:
                image_proj = self.clip_model.image_projection(image_features)
            else:
                image_proj = image_features
            image_proj = F.normalize(image_proj, dim=-1)
        return image_features, image_proj

    def encode_text_with_prompts(self, inputs_embeds, attention_mask):
        """
        Encode soft prompts through frozen Bio_ClinicalBERT.
        Gradients flow through to the inputs_embeds.
        """
        bert_model = self.clip_model.text_encoder.text_encoder
        text_output = bert_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        text_features = text_output.last_hidden_state  # [N, seq_len, hidden]

        if self.text_pooling == "eos":
            eos_indices = attention_mask.sum(dim=-1) - 1
            text_pooled = text_features[
                torch.arange(text_features.shape[0], device=text_features.device),
                eos_indices,
            ]
        elif self.text_pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(text_features.size()).float()
            text_pooled = torch.sum(text_features * mask_expanded, dim=1) / torch.clamp(
                mask_expanded.sum(dim=1), min=1e-9
            )
        else:
            text_pooled = text_features[:, 0]  # bos

        if self.has_projection:
            text_proj = self.clip_model.text_projection(text_pooled)
        else:
            text_proj = text_pooled
        text_proj = F.normalize(text_proj, dim=-1)

        return text_proj

    def forward(self, image, text_batch, is_eval=False):
        """
        Args:
            image: [B, C, H, W]
            text_batch: list of strings. If is_eval=False, len(B). If True, len(C).
            is_eval: Boolean.

        Returns:
            If is_eval=False:
                image_proj: [B, proj_dim]
                text_proj: [B, proj_dim]
                logit_scale: scalar tensor
            If is_eval=True:
                logits: [B, C] class logits
        """
        B = image.shape[0]

        # 1. Encode image
        image_features_raw, image_proj = self.encode_image(image)

        # 2. Generate dynamic prompts via Meta-Net (gradients flow to prompt_learner)
        inputs_embeds, attention_mask = self.prompt_learner(
            image_features_raw.detach().float(),
            text_batch,
            is_eval=is_eval
        )

        # 3. Encode prompts
        text_proj = self.encode_text_with_prompts(inputs_embeds, attention_mask)

        # 4. Output logic
        logit_scale = self.clip_model.logit_scale.exp()

        if not is_eval:
            # Contrastive Training Mode
            # text_proj is [B, proj_dim]
            return image_proj, text_proj, logit_scale
        else:
            # Zero-Shot Eval Mode
            # text_proj is [B * C, proj_dim]
            C = len(text_batch)
            text_proj = text_proj.view(B, C, -1)  # [B, C, proj_dim]
            
            # [B, 1, proj_dim] * [B, C, proj_dim] -> [B, C]
            logits = logit_scale * (image_proj.unsqueeze(1) * text_proj).sum(dim=-1)
            return logits
