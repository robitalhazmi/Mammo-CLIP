import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "static", "sample_images")
MAMMO_CLIP_SRC_DIR = os.path.join(os.path.dirname(BASE_DIR), "src", "codebase")

IMAGE_SIZE = (1520, 912)
MEAN = 0.3089279
STD = 0.25053555408335154
PROJECTION_DIM = 512

AVAILABLE_MODELS = {
    "en_b5": {
        "name": "Prompt-Guided MV VLM",
        "checkpoint": "b5-model-best-custom.tar",
        "encoder_name": "tf_efficientnet_b5_ns-detect",
        "out_dim": 2048,
        "supports_lp": True,
        "supports_ft": False,
    }
}

DEFAULT_PROMPTS = {
    "mass": ["no mass", "mass"],
    "calcification": ["no suspicious calcification", "suspicious calcification"],
    "malignancy": ["benign", "malignant"]
}

PAPER_METRICS = {
    "calcification": {
        "en_b2": {"zs_100": 0.68, "lp_10": 0.90, "lp_50": 0.92, "lp_100": 0.92, "ft_100": 0.98},
        "en_b5": {"zs_100": 0.62, "lp_10": 0.92, "lp_50": 0.94, "lp_100": 0.96, "ft_100": 0.98},
    },
    "mass": {
        "en_b2": {"zs_100": 0.58, "lp_10": 0.69, "lp_50": 0.72, "lp_100": 0.73, "ft_100": 0.85},
        "en_b5": {"zs_100": 0.76, "lp_10": 0.80, "lp_50": 0.84, "lp_100": 0.86, "ft_100": 0.88},
    },
    "density": {
        "en_b2": {"zs_100": 0.13, "lp_10": 0.80, "lp_50": 0.82, "lp_100": 0.84, "ft_100": 0.85},
        "en_b5": {"zs_100": 0.15, "lp_10": 0.83, "lp_50": 0.86, "lp_100": 0.86, "ft_100": 0.88},
    }
}

LABEL_INFO = {
    "mass": {"type": "binary", "metric": "AUC", "classes": ["No Mass", "Mass"]},
    "calcification": {"type": "binary", "metric": "AUC", "classes": ["No Calcification", "Calcification"]},
    "malignancy": {"type": "binary", "metric": "AUC", "classes": ["Benign", "Malignant"]}
}
