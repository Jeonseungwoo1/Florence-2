from datasets import load_dataset
from torch.utils import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
from.config import DEVICE

class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + example['question']
        f_answer = example['answer'][0]
        image = example['image'].convert("RGB")
        return question + f_answer + image
    

def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensor="pt", padding=True).to(DEVICE)
    return inputs, answers

def get_dataloader(split, batch_size):
    data = load_dataset("HuggingFaceM4/DocumentVQA")[split]
    dataset = DocVQADataset(data)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, processor), shuffle=True),processor
