"""
MammoPromptLearner: Learnable prompt generator for Mammo-CLIP + CoCoOp.

Adapts CoCoOp's conditional context optimization to work with Bio_ClinicalBERT
(HuggingFace BERT) instead of OpenAI CLIP's text transformer. Injects learnable
soft prompt tokens via BERT's `inputs_embeds` parameter.

Learnable components:
  - ctx: [n_ctx, text_dim] context vectors
  - meta_net: Linear -> ReLU -> Linear bottleneck that maps image features to a
              conditional bias token added to each context vector per image
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class MammoPromptLearner(nn.Module):
    """
    Generates image-conditional soft prompts for Bio_ClinicalBERT.

    For each input image, the Meta-Net produces a bias vector from the image
    encoder features. This bias is added to the shared learnable context vectors,
    creating instance-specific prompts. The prompts are then concatenated with
    class name token embeddings to form the full input for Bio_ClinicalBERT.

    Prompt structure: [CLS] ctx1' ctx2' ... ctxM' <class_tokens> [SEP] [PAD...]
    where ctx_m' = ctx_m + MetaNet(image_features)
    """

    def __init__(
        self,
        classnames,
        bert_model,
        tokenizer,
        img_dim,
        text_dim,
        n_ctx=4,
        ctx_init="",
        meta_net_reduction=16,
    ):
        """
        Args:
            classnames: List of class name strings (e.g. ["no mass", "mass"])
            bert_model: The underlying Bio_ClinicalBERT nn.Module
            tokenizer: HuggingFace tokenizer for Bio_ClinicalBERT
            img_dim: Image encoder output dimension (2048 for B5, 1408 for B2)
            text_dim: BERT hidden size (768)
            n_ctx: Number of learnable context tokens
            ctx_init: Initialization string; empty for random init
            meta_net_reduction: Bottleneck reduction factor for Meta-Net
        """
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.text_dim = text_dim

        # ---- Word embedding layer (frozen, for token lookup) ----
        word_embeddings = bert_model.embeddings.word_embeddings

        # ---- Context vector initialization ----
        if ctx_init and ctx_init.strip():
            ctx_init_clean = ctx_init.strip()
            token_ids = tokenizer.encode(ctx_init_clean, add_special_tokens=False)
            n_ctx = len(token_ids)
            self.n_ctx = n_ctx
            with torch.no_grad():
                ctx_vectors = word_embeddings(
                    torch.tensor(token_ids, dtype=torch.long)
                ).clone().float()
            print(f'Context initialized from: "{ctx_init_clean}" ({n_ctx} tokens)')
        else:
            ctx_vectors = torch.empty(n_ctx, text_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors, std=0.02)
            print(f"Context randomly initialized ({n_ctx} tokens)")

        self.ctx = nn.Parameter(ctx_vectors)  # [n_ctx, text_dim]

        # ---- Meta-Net: image features -> conditional bias token ----
        self.meta_net = nn.Sequential(
            OrderedDict([
                ("linear1", nn.Linear(img_dim, img_dim // meta_net_reduction)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(img_dim // meta_net_reduction, text_dim)),
            ])
        )

        # ---- Pre-compute frozen token embeddings ----
        cls_token_id = torch.tensor([tokenizer.cls_token_id], dtype=torch.long)
        sep_token_id = torch.tensor([tokenizer.sep_token_id], dtype=torch.long)

        with torch.no_grad():
            cls_embed = word_embeddings(cls_token_id).float()  # [1, text_dim]
            sep_embed = word_embeddings(sep_token_id).float()  # [1, text_dim]

        self.register_buffer("cls_embed", cls_embed)
        self.register_buffer("sep_embed", sep_embed)

        # ---- Pre-compute class name embeddings (frozen) ----
        class_token_lengths = []
        class_embedding_list = []

        for name in classnames:
            token_ids = tokenizer.encode(name, add_special_tokens=False)
            class_token_lengths.append(len(token_ids))
            with torch.no_grad():
                emb = word_embeddings(
                    torch.tensor(token_ids, dtype=torch.long)
                ).float()  # [n_tokens, text_dim]
            class_embedding_list.append(emb)

        self.class_token_lengths = class_token_lengths
        max_cls_len = max(class_token_lengths)
        self.max_cls_len = max_cls_len

        # Pad all class embeddings to max_cls_len and stack
        padded = []
        for emb in class_embedding_list:
            pad_len = max_cls_len - emb.shape[0]
            if pad_len > 0:
                emb = torch.cat(
                    [emb, torch.zeros(pad_len, text_dim, dtype=torch.float32)],
                    dim=0,
                )
            padded.append(emb)

        # [n_cls, max_cls_len, text_dim]
        self.register_buffer("class_embeddings", torch.stack(padded))

        # Max sequence length: [CLS] + ctx + class_tokens + [SEP]
        self.max_seq_len = 1 + self.n_ctx + max_cls_len + 1

        print(f"Prompt structure: [CLS] + {self.n_ctx} ctx + class_tokens(max {max_cls_len}) + [SEP]")
        print(f"Max prompt length: {self.max_seq_len}")
        print(f"Classes: {classnames}")

    def forward(self, image_features):
        """
        Generate image-conditional prompt embeddings for all classes.

        Args:
            image_features: [B, img_dim] raw image encoder output features

        Returns:
            inputs_embeds: [B * n_cls, max_seq_len, text_dim]
                Soft prompt embeddings to pass to BERT via inputs_embeds param.
                Ordered as [img0_cls0, img0_cls1, ..., img1_cls0, img1_cls1, ...]
            attention_mask: [B * n_cls, max_seq_len]
                Attention mask (1 = real token, 0 = padding)
        """
        B = image_features.shape[0]
        device = image_features.device

        # Meta-Net: generate image-conditional bias
        bias = self.meta_net(image_features)  # [B, text_dim]
        bias = bias.unsqueeze(1)              # [B, 1, text_dim]

        # Shift context vectors: broadcast add
        ctx = self.ctx.unsqueeze(0)           # [1, n_ctx, text_dim]
        ctx_shifted = ctx + bias              # [B, n_ctx, text_dim]

        # Expand fixed tokens for batching
        cls_expand = self.cls_embed.expand(B, -1, -1)  # [B, 1, text_dim]
        sep_expand = self.sep_embed.expand(B, -1, -1)  # [B, 1, text_dim]

        # Build prompts per class (vectorized over batch)
        all_embeds = []
        all_masks = []

        for cls_idx in range(self.n_cls):
            cls_len = self.class_token_lengths[cls_idx]
            # [cls_len, text_dim] -> [B, cls_len, text_dim]
            cls_emb = self.class_embeddings[cls_idx, :cls_len].unsqueeze(0).expand(B, -1, -1)

            # Concatenate: [CLS] + ctx_shifted + class_tokens + [SEP]
            prompt = torch.cat(
                [cls_expand, ctx_shifted, cls_emb, sep_expand], dim=1
            )  # [B, actual_len, text_dim]
            actual_len = prompt.shape[1]

            # Pad to max_seq_len if needed
            if actual_len < self.max_seq_len:
                pad = torch.zeros(
                    B,
                    self.max_seq_len - actual_len,
                    self.text_dim,
                    device=device,
                    dtype=prompt.dtype,
                )
                prompt = torch.cat([prompt, pad], dim=1)

            # Attention mask
            mask = torch.zeros(
                B, self.max_seq_len, device=device, dtype=torch.long
            )
            mask[:, :actual_len] = 1

            all_embeds.append(prompt)
            all_masks.append(mask)

        # Stack per-class prompts: [n_cls, B, max_seq_len, text_dim]
        # -> interleave to [B, n_cls, max_seq_len, text_dim]
        # -> reshape to [B*n_cls, max_seq_len, text_dim]
        inputs_embeds = torch.stack(all_embeds, dim=1)   # [B, n_cls, max_seq_len, text_dim]
        attention_mask = torch.stack(all_masks, dim=1)    # [B, n_cls, max_seq_len]

        inputs_embeds = inputs_embeds.view(B * self.n_cls, self.max_seq_len, self.text_dim)
        attention_mask = attention_mask.view(B * self.n_cls, self.max_seq_len)

        return inputs_embeds, attention_mask
