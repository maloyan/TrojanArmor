# models.py
import timm
import torch

def get_model(model_name, num_classes):
    if model_name.startswith("timm_"):
        return get_timm_model(model_name[5:], num_classes)
    elif model_name.startswith("hf_"):
        return get_huggingface_model(model_name[3:])
    else:
        # Add other model sources here
        raise ValueError(f"Unsupported model: {model_name}")

def get_timm_model(model_name, num_classes):
    if not timm.is_model(model_name):
        available_models = timm.list_models()
        print(f"Model '{model_name}' not found in the timm library. Available models:")
        for model in available_models:
            print(f"  - {model}")
        raise ValueError(f"Model '{model_name}' not found in the timm library.")
    
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def get_huggingface_model(model_name):
    try:
        model = timm.create_model(f"hf_hub:{model_name}", pretrained=True)
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}' from Hugging Face: {e}")
        raise
