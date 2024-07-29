from transformers import AutoModelForCausalLM
from .config import DEVICE, MODEL_ID

def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    return model