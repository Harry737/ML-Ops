from torchvision import datasets, transforms
import torch

MODEL_PATH = "mnist_cnn.pt"
DEVICE = torch.device("cpu")
from main import Net

# Load model once at startup
model = Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

test_ds = datasets.MNIST(
    "data",
    train=False,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

img, label = test_ds[0]
print("True label:", label)

with torch.no_grad():
    pred = model(img.unsqueeze(0))
    print("Pred:", pred.argmax(1).item())
