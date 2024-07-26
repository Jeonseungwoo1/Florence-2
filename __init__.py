import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_datasett
from PIL import Image
import requests