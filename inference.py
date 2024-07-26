import requests
from PIL import Image
from .config import DEVICE

def run_example(model, processor, task_prompt, image_url, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=prompt, images=image, return_tensor="pt").to(DEVICE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixels_values=inputs["pixels_valuse"],
        max_new_tokens = 1024,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tocken=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    print(parsed_answer)