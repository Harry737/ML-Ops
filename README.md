# ML-Ops: PyTorch Model Training & Deployment Pipeline

A complete MLOps solution for training PyTorch models with automated CI/CD pipelines, GPU auto-scaling, and Kubernetes-based inference.

## Project Overview

This project demonstrates a production-grade MLOps workflow that includes:
- **Data Management**: DVC (Data Version Control) with S3 backend
- **Model Training**: PyTorch-based training with automated GPU provisioning
- **Infrastructure**: Amazon EKS with Karpenter for automatic GPU instance scaling
- **Model Registry**: Trained models stored in S3
- **Continuous Deployment**: ArgoCD-driven deployment with KServe and alternative inference methods

## Architecture

### Training Pipeline
1. Code changes pushed to repository
2. Training pipeline automatically triggers
3. Training Docker image is built
4. Karpenter provisions GPU instances in EKS cluster
5. Training job runs on GPU
6. Trained model uploaded to S3

### Deployment Pipeline
1. Model uploaded to S3 triggers EventBridge
2. AWS Lambda function initiates CD pipeline
3. KServe inference path is updated
4. ArgoCD automatically deploys the new model version

### Inference Options
- **KServe**: Production-grade model serving (recommended)
- **Alternative Deployment**: Custom inference deployment (see `inference/` folder)

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Training dependencies
├── requirements-api.txt               # API dependencies
├── main.py                            # Training script
├── api.py                             # Inference API server
├── train/                             # Training data and models
│   └── data/MNIST/                    # Training dataset (managed by DVC)
├── k8s/                               # Kubernetes configurations
│   └── argo/                          # ArgoCD application manifests
│       └── inference.yaml             # KServe inference deployment
├── inference/                         # Alternative inference deployment methods
└── docker/                            # Docker configurations
    └── Dockerfile.train               # Training image (built automatically)
```

## Getting Started

### Prerequisites
- Docker
- AWS CLI configured with appropriate credentials
- kubectl and Helm
- ArgoCD and Karpenter deployed on EKS cluster

### Local Development

Install dependencies:
```bash
pip install -r requirements.txt
python main.py
# To specify GPU: CUDA_VISIBLE_DEVICES=2 python main.py
```

### API Server

Install API dependencies and start the inference server:
```bash
pip install -r requirements-api.txt \
  --index-url https://download.pytorch.org/whl/cpu

uvicorn api:app --host 0.0.0.0 --port 8000
```

Example inference request:
```bash
base64 digit.png | tr -d '\n'
```

## Data Management with DVC

### Initialize DVC Remote (S3)
```bash
dvc add train/data/MNIST/
dvc remote add -d dvcremote s3://mlops-harry/data
dvc push
```

## CI/CD Pipeline Details

### Automatic Training Triggers
- Repository changes trigger the training pipeline
- Docker image built with training code
- Karpenter automatically provisions GPU instances
- Training job scheduled on available GPU nodes
- Model artifacts saved to S3

### Automatic Deployment Triggers
- S3 model upload → EventBridge event → Lambda function
- Lambda initiates KServe update or alternative deployment
- ArgoCD syncs and deploys updated configuration

## Configuration Files

- **k8s/argo/inference.yaml**: KServe-based model serving configuration
- **inference/**: Alternative deployment patterns without KServe

## Monitoring & Logging

- ArgoCD dashboard for deployment status
- CloudWatch logs for Lambda and training jobs
- Kubernetes pod logs for inference serving

## Contributing

When making changes:
1. Update code in the repository
2. Pipeline automatically triggers
3. Monitor training progress in logs
4. Once complete, model automatically deployed via ArgoCD
