"""
MammoPromptLearner: Learnable prompt generator for Mammo-CLIP + CoCoOp.

Adapts CoCoOp's conditional context optimization to work with Bio_ClinicalBERT.
Supports dynamic text batches for Contrastive Learning, and fixed class names
for Zero-Shot evaluation.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class MammoPromptLearner(nn.Module):
    def __init__(
        self,
        bert_model,
        tokenizer,
        img_dim,
        text_dim,
        n_ctx=4,
        ctx_init="",
        meta_net_reduction=16,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.text_dim = text_dim
        self.tokenizer = tokenizer

        # ---- Word embedding layer (frozen, for token lookup) ----
        self.word_embeddings = bert_model.embeddings.word_embeddings

        # ---- Context vector initialization ----
        if ctx_init and ctx_init.strip():
            ctx_init_clean = ctx_init.strip()
            token_ids = tokenizer.encode(ctx_init_clean, add_special_tokens=False)
            n_ctx = len(token_ids)
            self.n_ctx = n_ctx
            with torch.no_grad():
                ctx_vectors = self.word_embeddings(
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

        # ---- Pre-compute frozen special token embeddings ----
        cls_token_id = torch.tensor([tokenizer.cls_token_id], dtype=torch.long)
        sep_token_id = torch.tensor([tokenizer.sep_token_id], dtype=torch.long)

        with torch.no_grad():
            cls_embed = self.word_embeddings(cls_token_id).float()  # [1, text_dim]
            sep_embed = self.word_embeddings(sep_token_id).float()  # [1, text_dim]

        self.register_buffer("cls_embed", cls_embed)
        self.register_buffer("sep_embed", sep_embed)

    def _get_text_embeddings(self, text_batch, device):
        """Tokenize and embed a batch of strings."""
        token_lengths = []
        embedding_list = []

        for text in text_batch:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_lengths.append(len(token_ids))
            with torch.no_grad():
                emb = self.word_embeddings(
                    torch.tensor(token_ids, dtype=torch.long, device=device)
                ).float()  # [n_tokens, text_dim]
            embedding_list.append(emb)

        max_len = max(token_lengths)
        
        # Pad all to max_len
        padded = []
        for emb in embedding_list:
            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, self.text_dim, device=device, dtype=torch.float32)
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)

        # [len(text_batch), max_len, text_dim]
        return torch.stack(padded), token_lengths, max_len

    def forward(self, image_features, text_batch, is_eval=False):
        """
        Args:
            image_features: [B, img_dim]
            text_batch: list of strings.
                If is_eval=False: len(text_batch) must equal B. Returns [B, max_seq_len, text_dim]
                If is_eval=True: len(text_batch) = C (classes). Returns [B * C, max_seq_len, text_dim]
            is_eval: Boolean indicating evaluation mode.
        """
        B = image_features.shape[0]
        device = image_features.device

        # Meta-Net: generate image-conditional bias
        bias = self.meta_net(image_features)  # [B, text_dim]
        bias = bias.unsqueeze(1)              # [B, 1, text_dim]

        # Shift context vectors: [1, n_ctx, text_dim] + [B, 1, text_dim]
        ctx_shifted = self.ctx.unsqueeze(0) + bias  # [B, n_ctx, text_dim]

        # Get text embeddings
        text_embeds, token_lengths, max_text_len = self._get_text_embeddings(text_batch, device)
        max_seq_len = 1 + self.n_ctx + max_text_len + 1

        all_embeds = []
        all_masks = []

        if not is_eval:
            # Contrastive Training Mode: len(text_batch) == B
            assert len(text_batch) == B, "In training mode, len(text_batch) must equal batch size B"
            
            cls_expand = self.cls_embed.expand(B, -1, -1)  # [B, 1, text_dim]
            sep_expand = self.sep_embed.expand(B, -1, -1)  # [B, 1, text_dim]
            
            for i in range(B):
                t_len = token_lengths[i]
                t_emb = text_embeds[i, :t_len].unsqueeze(0)  # [1, t_len, text_dim]
                
                # [CLS] + ctx_shifted[i] + text + [SEP]
                prompt = torch.cat(
                    [cls_expand[i:i+1], ctx_shifted[i:i+1], t_emb, sep_expand[i:i+1]], dim=1
                )
                actual_len = prompt.shape[1]
                
                if actual_len < max_seq_len:
                    pad = torch.zeros(1, max_seq_len - actual_len, self.text_dim, device=device, dtype=prompt.dtype)
                    prompt = torch.cat([prompt, pad], dim=1)
                    
                mask = torch.zeros(1, max_seq_len, device=device, dtype=torch.long)
                mask[:, :actual_len] = 1
                
                all_embeds.append(prompt)
                all_masks.append(mask)
                
            inputs_embeds = torch.cat(all_embeds, dim=0)  # [B, max_seq_len, text_dim]
            attention_mask = torch.cat(all_masks, dim=0)  # [B, max_seq_len]
            
        else:
            # Zero-Shot Eval Mode: text_batch has C classes, image_features has B images
            C = len(text_batch)
            
            cls_expand = self.cls_embed.expand(B, -1, -1)  # [B, 1, text_dim]
            sep_expand = self.sep_embed.expand(B, -1, -1)  # [B, 1, text_dim]
            
            for c_idx in range(C):
                t_len = token_lengths[c_idx]
                # [1, t_len, text_dim] -> [B, t_len, text_dim]
                t_emb = text_embeds[c_idx, :t_len].unsqueeze(0).expand(B, -1, -1)
                
                # [CLS] + ctx_shifted + text + [SEP]
                prompt = torch.cat(
                    [cls_expand, ctx_shifted, t_emb, sep_expand], dim=1
                )
                actual_len = prompt.shape[1]
                
                if actual_len < max_seq_len:
                    pad = torch.zeros(B, max_seq_len - actual_len, self.text_dim, device=device, dtype=prompt.dtype)
                    prompt = torch.cat([prompt, pad], dim=1)
                    
                mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
                mask[:, :actual_len] = 1
                
                all_embeds.append(prompt)
                all_masks.append(mask)
                
            # Stack: [C, B, max_seq_len, text_dim]
            inputs_embeds = torch.stack(all_embeds, dim=1)  # [B, C, max_seq_len, text_dim]
            attention_mask = torch.stack(all_masks, dim=1)  # [B, C, max_seq_len]
            
            # Reshape to [B * C, max_seq_len, text_dim]
            inputs_embeds = inputs_embeds.view(B * C, max_seq_len, self.text_dim)
            attention_mask = attention_mask.view(B * C, max_seq_len)

        return inputs_embeds, attention_mask
