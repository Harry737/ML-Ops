from torchvision import datasets
from PIL import Image
import os

os.makedirs("mnist_test_images", exist_ok=True)

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=False
)

for i in range(10):
    img, label = dataset[i]
    path = f"mnist_test_images/{i}_label_{label}.png"
    img.save(path)
    print("Saved:", path)
