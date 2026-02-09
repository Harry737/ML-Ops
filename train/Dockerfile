FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AWS_DEFAULT_REGION=ap-south-1

RUN --mount=type=secret,id=aws_access_key \
    --mount=type=secret,id=aws_secret_key \
    --mount=type=secret,id=aws_region \
    export AWS_ACCESS_KEY_ID=$(cat /run/secrets/aws_access_key) && \
    export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/aws_secret_key) && \
    export AWS_DEFAULT_REGION=$(cat /run/secrets/aws_region) && \
    export AWS_REGION=$(cat /run/secrets/aws_region) && \
    echo ${AWS_DEFAULT_REGION} && \
    dvc init --no-scm && \
    dvc remote add -d dvcremote s3://mlops-harry/data && \
    dvc pull

CMD ["python", "main.py", "--save-model"]