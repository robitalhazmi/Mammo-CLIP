import os
import sys
import glob
import torch
import logging
from .config import MAMMO_CLIP_SRC_DIR, AVAILABLE_MODELS, CHECKPOINT_DIR

log = logging.getLogger(__name__)

# Add source codebase to path for importing breastclip modules
if MAMMO_CLIP_SRC_DIR not in sys.path:
    sys.path.insert(0, MAMMO_CLIP_SRC_DIR)


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_cache_key = None
        self._clip_cache_value = None
        self._classifier_cache_key = None  
        self._classifier_cache_value = None
        self._ckpt_cache = {}  # Cache loaded checkpoints for reuse
        log.info(f"ModelManager initialized on device: {self.device}")

    def get_device(self):
        return self.device

    def _load_checkpoint(self, model_key: str, label: str):
        """Load and cache the pretrained CLIP checkpoint."""
        cache_key = f"{model_key}_{label}"
        if cache_key in self._ckpt_cache:
            return self._ckpt_cache[cache_key]
        
        model_info = AVAILABLE_MODELS[model_key]
        
        # Checkpoint selection logic based on requested label
        if model_key == "en_b5":
            if label == "malignancy":
                ckpt_name = "b5-model-best-custom.tar"
            else:
                ckpt_name = "b5-model-best-epoch-7.tar"
        else:
            ckpt_name = model_info["checkpoint"]

        ckpt_path = os.path.join(CHECKPOINT_DIR, "pretrained", ckpt_name)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Run setup_demo.sh to download checkpoints."
            )
        
        log.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self._ckpt_cache[cache_key] = ckpt
        return ckpt

    def load_clip_model(self, model_key: str, label: str):
        """Load CLIP model and tokenizer."""
        cache_key = f"{model_key}_{label}"
        if self._clip_cache_key == cache_key and self._clip_cache_value is not None:
            return self._clip_cache_value

        from breastclip.model import build_model
        from transformers import AutoTokenizer

        ckpt = self._load_checkpoint(model_key, label)
        
        # The original checkpoints have cache_dir hardcoded to /ocean/... 
        # Overwrite it to point to local clone
        from core.config import BASE_DIR
        local_bert_path = os.path.join(BASE_DIR, "checkpoints", "pretrained", "Bio_ClinicalBERT")
        if 'text_encoder' in ckpt['config']['model']:
            ckpt['config']['model']['text_encoder']['name'] = local_bert_path
            ckpt['config']['model']['text_encoder']['cache_dir'] = ""
            
        tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
        
        model = build_model(
            model_config=ckpt['config']['model'],
            loss_config=ckpt['config']['loss'],
            tokenizer=tokenizer
        )
        model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(self.device)
        model.eval()

        self._clip_cache_key = cache_key
        self._clip_cache_value = (model, tokenizer)
        log.info(f"CLIP model '{model_key}' loaded successfully")
        return model, tokenizer

    def load_classifier(self, model_key: str, label: str, setting: str):
        """Load downstream classifier model.
        
        The BreastClipClassifier requires the pretrained CLIP checkpoint
        to extract the image encoder weights.
        """
        cache_key = f"{model_key}_{label}_{setting}"
        if self._classifier_cache_key == cache_key and self._classifier_cache_value is not None:
            return self._classifier_cache_value

        from Classifiers.models.breast_clip_classifier import BreastClipClassifier
        import argparse

        model_info = AVAILABLE_MODELS[model_key]
        
        # The BreastClipClassifier needs the pretrained CLIP checkpoint 
        # to initialize the image encoder
        ckpt = self._load_checkpoint(model_key, label)
        
        # Build arch string: e.g. 'breast_clip_det_b5_period_n_lp'
        backbone = model_key.split('_')[1]  # 'b2' or 'b5'
        arch = f"breast_clip_det_{backbone}_period_n_{setting}"
        
        n_class = 1

        args = argparse.Namespace(arch=arch)

        # BreastClipClassifier(args, ckpt, n_class)
        model = BreastClipClassifier(args, ckpt, n_class)

        # Find the downstream classifier checkpoint
        downstream_dir = os.path.join(CHECKPOINT_DIR, "downstream", "classifier")
        
        if n_class == 1:
            search_pattern = f"*{setting}*best_aucroc*.pth"
        else:
            search_pattern = f"*{setting}*best_acc*.pth"
        
        # Search across all subdirs
        all_possible_files = glob.glob(os.path.join(downstream_dir, "**", search_pattern), recursive=True)
        
        # Filter by label directory
        possible_files = [f for f in all_possible_files if f"/{label}/" in f or f"\\{label}\\" in f]
        
        if not possible_files:
            raise FileNotFoundError(
                f"Classifier checkpoint not found for {cache_key} in {downstream_dir} under label {label}. "
                f"Run setup_demo.sh to download downstream checkpoints."
            )

        classifier_ckpt_path = possible_files[0]
        log.info(f"Loading classifier checkpoint from {classifier_ckpt_path}")
        classifier_ckpt = torch.load(classifier_ckpt_path, map_location='cpu', weights_only=False)
        
        if isinstance(classifier_ckpt, dict) and 'state_dict' in classifier_ckpt:
            model.load_state_dict(classifier_ckpt['state_dict'], strict=False)
        elif isinstance(classifier_ckpt, dict) and 'model' in classifier_ckpt:
            model.load_state_dict(classifier_ckpt['model'], strict=False)
        else:
            model.load_state_dict(classifier_ckpt, strict=False)

        model = model.to(self.device)
        model.eval()

        self._classifier_cache_key = cache_key
        self._classifier_cache_value = model
        log.info(f"Classifier '{cache_key}' loaded successfully")
        return model
