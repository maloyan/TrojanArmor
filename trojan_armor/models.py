# models.py
import timm

def get_model(model_name, num_classes):
    if model_name.startswith("timm_"):
        return get_timm_model(model_name[5:], num_classes)
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
