import argparse
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import boto3
import time
import os
import subprocess, textwrap
import sys

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    #use_accel = not args.no_accel and torch.accelerator.is_available()
    use_cuda = (not args.no_accel) and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)

    # if use_cuda:
    #     device = torch.accelerator.current_accelerator()
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        train_kwargs.update({
            'num_workers': 2,
            'pin_memory': True
        })
        test_kwargs.update({
            'num_workers': 2,
            'pin_memory': True
        })

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        # 1. Save model locally
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("artifacts/model-store", exist_ok=True)
        os.makedirs("artifacts/config", exist_ok=True)
        local_model_path = "artifacts/mnist_cnn.pt"

        torch.save(model.state_dict(), local_model_path)

        model_code = textwrap.dedent("""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)
        """)

        with open("artifacts/model.py", "w") as f:
            f.write(model_code)

        handler_code = textwrap.dedent("""
        from ts.torch_handler.base_handler import BaseHandler
        import torch
        from PIL import Image
        import io
        import base64
        from model import Net

        class MNISTHandler(BaseHandler):

            def initialize(self, ctx):
                import os
                self.device = "cpu"
                self.model = Net()

                model_dir = ctx.system_properties.get("model_dir")
                model_pt_path = os.path.join(model_dir, "mnist_cnn.pt")

                self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
                self.model.eval()

            def preprocess(self, data):
                if not data:
                    return torch.zeros(1,1,28,28)

                row = data[0]

                # TorchServe wraps request
                payload = row.get("data") or row.get("body")

                # ---- KServe V2 protocol ----
                if isinstance(payload, dict) and "inputs" in payload:
                    try:
                        base64_str = payload["inputs"][0]["data"][0]
                        image_bytes = base64.b64decode(base64_str)
                    except Exception as e:
                        raise ValueError(f"Invalid V2 payload format: {payload}") from e

                # ---- KFServing v1 ----
                elif isinstance(payload, dict) and "instances" in payload:
                    base64_str = payload["instances"][0]
                    image_bytes = base64.b64decode(base64_str)

                # ---- raw base64 ----
                elif isinstance(payload, str):
                    image_bytes = base64.b64decode(payload)

                # ---- torchserve test format ----
                elif isinstance(payload, (bytes, bytearray)):
                    image_bytes = payload

                else:
                    raise ValueError(f"Unsupported payload format: {type(payload)} -> {payload}")

                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                image = image.resize((28,28))

                tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
                tensor = tensor.view(1,1,28,28)/255.0
                return tensor

            def inference(self, inputs):
                with torch.no_grad():
                    outputs = self.model(inputs)
                return outputs

            def postprocess(self, outputs):
                preds = outputs.argmax(dim=1).tolist()
                return [{"predictions": preds}]
        """)

        handler_path = "artifacts/handler.py"
        with open(handler_path, "w") as f:
            f.write(handler_code)

        # -------------------------------
        # 3️⃣ Package TorchServe model (.mar)
        # -------------------------------
        result = subprocess.run([
            "torch-model-archiver",
            "--model-name", "mnist",
            "--version", "1.0",
            "--serialized-file", local_model_path,
            "--handler", handler_path,
            "--extra-files", "artifacts/model.py",
            "--export-path", "artifacts/model-store",
            "--force"
        ], capture_output=True, text=True)

        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("Model archiver failed")
        
        config_content = """inference_address=http://0.0.0.0:8085
                            management_address=http://0.0.0.0:8086
                            metrics_address=http://0.0.0.0:8087
                            grpc_inference_port=7070
                            model_store=/mnt/models/model-store
                            load_models=all

                            model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":1,"maxWorkers":1,"batchSize":1,"maxBatchDelay":100,"responseTimeout":120}}}}
                            """
        with open("artifacts/config/config.properties", "w") as f:
            f.write(config_content)

        print("✅ config.properties created")

        # -------------------------------
        # 4️⃣ Upload to S3
        # -------------------------------
        bucket = "mlops-harry"
        timestamp = int(time.time())
        s3_prefix = f"mnist/{timestamp}/"

        s3 = boto3.client("s3")

        # upload model-store/*
        for root, _, files in os.walk("artifacts/model-store"):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, "artifacts")
                s3.upload_file(local_path, bucket, s3_prefix + relative_path)

        # upload config/*
        for root, _, files in os.walk("artifacts/config"):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, "artifacts")
                s3.upload_file(local_path, bucket, s3_prefix + relative_path)

        print(f"✅ TorchServe model uploaded to s3://{bucket}/{s3_prefix}")

        # 2. Create timestamped S3 path
        # timestamp = int(time.time())
        # bucket_name = "mlops-harry"
        # s3_key = f"mnist/{timestamp}/mnist_cnn.pt"

        # # 3. Upload to S3
        # s3 = boto3.client("s3")
        # s3.upload_file(local_model_path, bucket_name, s3_key)

        # print(f"✅ Model uploaded to s3://{bucket_name}/{s3_key}")


if __name__ == '__main__':
    main()
