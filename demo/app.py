from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import json
import time
import os
import logging
import torch

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")

from core.config import (
    BASE_DIR, SAMPLE_IMAGES_DIR, AVAILABLE_MODELS, 
    PAPER_METRICS, LABEL_INFO, DEFAULT_PROMPTS
)
from core.model_manager import ModelManager
from core.preprocessor import preprocess_image, load_sample_image
from core.inference import zero_shot_predict, classifier_predict, get_default_prompts
from core.visualization import create_visualization, image_to_base64

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Mammo-CLIP Interactive Demo")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

model_manager = ModelManager()

# Map frontend setting values to backend short codes
SETTING_MAP = {
    "zero_shot": "zs",
    "linear_probe": "lp",
    "finetune": "ft",
    # Also accept short codes directly
    "zs": "zs",
    "lp": "lp",
    "ft": "ft",
}


@app.on_event("startup")
async def startup_event():
    device = model_manager.get_device()
    gpu_name = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    log.info(f"Mammo-CLIP Demo started | Device: {device} {gpu_name}")
    
    # Check for checkpoints
    from core.config import CHECKPOINT_DIR
    pretrained_dir = os.path.join(CHECKPOINT_DIR, "pretrained")
    if os.path.exists(pretrained_dir):
        ckpts = os.listdir(pretrained_dir)
        log.info(f"Available checkpoints: {ckpts}")
    else:
        log.warning(f"Checkpoint directory not found: {pretrained_dir}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health_check():
    device = model_manager.get_device()
    gpu_name = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "status": "healthy",
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_available": torch.cuda.is_available()
    }


@app.get("/api/sample-images")
async def get_sample_images():
    if not os.path.exists(SAMPLE_IMAGES_DIR):
        return []
    images = []
    for f in sorted(os.listdir(SAMPLE_IMAGES_DIR)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            images.append({
                "filename": f, 
                "path": f"/static/sample_images/{f}"
            })
    return images


@app.get("/api/paper-metrics")
async def get_paper_metrics():
    return PAPER_METRICS


@app.get("/api/models")
async def get_models():
    return AVAILABLE_MODELS


@app.get("/api/label-info")
async def get_label_info():
    return LABEL_INFO


@app.get("/api/default-prompts/{label}")
async def default_prompts(label: str):
    prompts = get_default_prompts(label)
    if not prompts:
        raise HTTPException(status_code=404, detail=f"Unknown label: {label}")
    return {"prompts": prompts}


@app.post("/api/predict")
async def predict(
    model: str = Form(...),
    label: str = Form(...),
    setting: str = Form(...),
    prompts: str = Form(None),
    image: UploadFile = File(None),
    sample_image: str = Form(None),
):
    start_time = time.time()
    
    try:
        # Validate inputs
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
        if label not in LABEL_INFO:
            raise HTTPException(status_code=400, detail=f"Invalid label: {label}")
        
        # Map setting name
        setting_code = SETTING_MAP.get(setting)
        if setting_code is None:
            raise HTTPException(status_code=400, detail=f"Invalid setting: {setting}")
        
        # Check model supports this setting
        model_info = AVAILABLE_MODELS[model]
        if setting_code == "lp" and not model_info.get("supports_lp", False):
            raise HTTPException(status_code=400, detail=f"Model {model} does not support linear probe")
        if setting_code == "ft" and not model_info.get("supports_ft", False):
            raise HTTPException(status_code=400, detail=f"Model {model} does not support finetune")
        
        # Preprocess image
        if image:
            content = await image.read()
            image_tensor, vis_img = preprocess_image(content)
        elif sample_image:
            image_path = os.path.join(SAMPLE_IMAGES_DIR, sample_image)
            if not os.path.exists(image_path):
                raise HTTPException(status_code=400, detail=f"Sample image not found: {sample_image}")
            image_tensor, vis_img = load_sample_image(image_path)
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        device = model_manager.get_device()

        if setting_code == "zs":
            # Zero-shot prediction
            prompt_list = json.loads(prompts) if prompts else get_default_prompts(label)
            clip_model, tokenizer = model_manager.load_clip_model(model, label)
            results = zero_shot_predict(clip_model, tokenizer, image_tensor, label, prompt_list, device)
            vis_data = create_visualization(
                clip_model, image_tensor, vis_img, label, 
                results["prediction_idx"], device, is_clip_model=True, sample_image=sample_image
            )
        else:
            # Classifier prediction (LP or FT)
            classifier = model_manager.load_classifier(model, label, setting_code)
            results = classifier_predict(classifier, image_tensor, label, device)
            vis_data = create_visualization(
                classifier, image_tensor, vis_img, label,
                results["prediction_idx"], device, is_clip_model=False, sample_image=sample_image
            )

        inference_time = round(time.time() - start_time, 3)

        return {
            "prediction": results,
            "visualization": vis_data,
            "metrics": PAPER_METRICS.get(label, {}).get(model, {}),
            "label_info": LABEL_INFO.get(label, {}),
            "inference_time": inference_time,
            "device": str(device),
            "model_name": AVAILABLE_MODELS[model]["name"],
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
