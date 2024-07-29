from .config import BATCH_SIZE
from .dataset import get_dataloader
from .model import load_model
from .train import train_model
from .inference import run_example


def main():
    train_loader, processor = get_dataloader('train', BATCH_SIZE)
    val_loader, _ = get_dataloader('validation', BATCH_SIZE)

    model = load_model()
    model = train_model(model, train_loader)

    example_image_url = "https://www.looper.com/img/gallery/the-ending-of-harry-potter-explained/intro.jpg?download=true"
    run_example(model, processor, "<CAPTION>", example_image_url)

if __name__ == "__main__":
    main()