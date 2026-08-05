"""
MammoCoCoOp: Wraps a pre-trained Mammo-CLIP model with CoCoOp prompt learning.

Loads a frozen Mammo-CLIP checkpoint (image encoder + text encoder + projections)
and adds a learnable MammoPromptLearner on top. Only the prompt learner's
parameters (context vectors + Meta-Net) are trained.

During forward pass:
  1. Image features are extracted via frozen image encoder
  2. MammoPromptLearner generates image-conditional soft prompts
  3. Prompts are passed through frozen Bio_ClinicalBERT via inputs_embeds
  4. Text features are pooled and projected (frozen)
  5. Cosine similarity between image and text embeddings produces logits
  6. Cross-entropy loss is computed against class labels
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
    """
    Mammo-CLIP + CoCoOp model.

    Frozen components (from pre-trained checkpoint):
      - image_encoder (EfficientNet B5 or B2)
      - text_encoder (Bio_ClinicalBERT)
      - image_projection, text_projection
      - logit_scale

    Learnable components:
      - prompt_learner.ctx (context vectors)
      - prompt_learner.meta_net (image-conditional bias generator)
    """

    def __init__(
        self,
        clip_ckpt_path,
        classnames,
        n_ctx=4,
        ctx_init="",
        meta_net_reduction=16,
    ):
        """
        Args:
            clip_ckpt_path: Path to pre-trained Mammo-CLIP checkpoint (.tar)
            classnames: List of class name strings (e.g. ["no mass", "mass"])
            n_ctx: Number of learnable context tokens
            ctx_init: Initialization string for context vectors (empty = random)
            meta_net_reduction: Meta-Net bottleneck reduction factor
        """
        super().__init__()
        self.n_cls = len(classnames)

        # ---- Load checkpoint ----
        print(f"Loading Mammo-CLIP checkpoint from: {clip_ckpt_path}")
        ckpt = torch.load(clip_ckpt_path, map_location="cpu")
        model_config = ckpt["config"]["model"]

        # ---- Build BreastClip model architecture ----
        # Need tokenizer for model construction (vocab_size reference)
        text_encoder_name = model_config["text_encoder"]["name"]
        cache_dir = model_config["text_encoder"].get("cache_dir", None)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, cache_dir=cache_dir)

        from breastclip.model import build_model
        clip_model = build_model(model_config, {}, tokenizer=tokenizer)

        # Load pre-trained weights
        clip_model.load_state_dict(ckpt["model"], strict=True)
        print("Loaded pre-trained Mammo-CLIP weights successfully")

        # ---- Store components ----
        self.clip_model = clip_model
        self.text_pooling = clip_model.text_pooling
        self.has_projection = clip_model.projection

        # ---- Freeze ALL clip_model parameters ----
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("Froze all Mammo-CLIP parameters")

        # ---- Get dimensions ----
        img_dim = clip_model.image_encoder.out_dim
        text_dim = clip_model.text_encoder.out_dim
        print(f"Image encoder dim: {img_dim}, Text encoder dim: {text_dim}")

        # ---- Create learnable prompt learner ----
        bert_model = clip_model.text_encoder.text_encoder  # Underlying BERT model
        self.prompt_learner = MammoPromptLearner(
            classnames=classnames,
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
        frozen = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        print(f"Trainable parameters ({len(trainable)}):")
        for name in trainable:
            print(f"  {name}")
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Total trainable: {total_trainable:,} | Total frozen: {total_frozen:,}")

    def encode_image(self, image):
        """
        Encode image through frozen image encoder + projection.

        Returns:
            image_features: [B, img_dim] raw features (for Meta-Net input)
            image_proj: [B, proj_dim] projected + normalized (for cosine similarity)
        """
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
        Encode soft prompts through frozen Bio_ClinicalBERT + projection.

        IMPORTANT: No torch.no_grad() here! Gradients must flow through
        the text encoder's computation graph back to the prompt embeddings
        (which connect to the learnable ctx and meta_net).

        Args:
            inputs_embeds: [B*n_cls, seq_len, text_dim] soft prompt embeddings
            attention_mask: [B*n_cls, seq_len]

        Returns:
            text_proj: [B*n_cls, proj_dim] projected + normalized text features
        """
        # Pass through BERT via inputs_embeds (bypassing word embedding lookup)
        bert_model = self.clip_model.text_encoder.text_encoder
        text_output = bert_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        text_features = text_output.last_hidden_state  # [B*n_cls, seq_len, hidden]

        # Pool using same strategy as BreastClip
        if self.text_pooling == "eos":
            # EOS pooling: take last non-padded token
            eos_indices = attention_mask.sum(dim=-1) - 1
            text_pooled = text_features[
                torch.arange(text_features.shape[0], device=text_features.device),
                eos_indices,
            ]
        elif self.text_pooling == "bos":
            text_pooled = text_features[:, 0]
        elif self.text_pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(text_features.size()).float()
            text_pooled = torch.sum(text_features * mask_expanded, dim=1) / torch.clamp(
                mask_expanded.sum(dim=1), min=1e-9
            )
        else:
            raise NotImplementedError(f"Unsupported pooling: {self.text_pooling}")

        # Project (frozen but gradient flows through computation graph)
        if self.has_projection:
            text_proj = self.clip_model.text_projection(text_pooled)
        else:
            text_proj = text_pooled
        text_proj = F.normalize(text_proj, dim=-1)

        return text_proj

    def forward(self, image, labels=None):
        """
        Forward pass: image -> logits (or loss if labels provided).

        Args:
            image: [B, C, H, W] input mammogram images
            labels: [B] class indices (optional; if provided, returns loss)

        Returns:
            If labels is None: logits [B, n_cls]
            If labels provided: scalar cross-entropy loss
        """
        B = image.shape[0]

        # 1. Encode image (frozen, no_grad)
        image_features_raw, image_proj = self.encode_image(image)

        # 2. Generate dynamic prompts (LEARNABLE — gradients flow through meta_net + ctx)
        #    Use detached image features as Meta-Net input (no need to backprop through image encoder)
        inputs_embeds, attention_mask = self.prompt_learner(image_features_raw.detach().float())

        # 3. Encode prompts through frozen text encoder (gradients flow through to prompt_learner)
        text_proj = self.encode_text_with_prompts(inputs_embeds, attention_mask)

        # 4. Reshape: [B*n_cls, proj_dim] -> [B, n_cls, proj_dim]
        text_proj = text_proj.view(B, self.n_cls, -1)

        # 5. Cosine similarity scaled by logit_scale
        logit_scale = self.clip_model.logit_scale.exp()
        # image_proj: [B, proj_dim] -> [B, 1, proj_dim]
        logits = logit_scale * (image_proj.unsqueeze(1) * text_proj).sum(dim=-1)  # [B, n_cls]

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits
