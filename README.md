# Basic MNIST Example

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

pip install -r requirements-api.txt \
  --index-url https://download.pytorch.org/whl/cpu

uvicorn api:app --host 0.0.0.0 --port 8000

base64 digit.png | tr -d '\n'

dvc add train/data/MNIST/

dvc remote add -d dvcremote s3://mlops-harry/data

dvc push
