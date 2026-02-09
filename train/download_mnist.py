from torchvision import datasets

datasets.MNIST(root="data", train=True, download=True)
datasets.MNIST(root="data", train=False, download=True)

# import torch

# print("CUDA device name:", torch.cuda.get_device_name(0))
