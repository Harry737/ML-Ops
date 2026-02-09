FROM python:3.10-slim

# Prevent Python from writing pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# install aws cli v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

COPY . .

RUN --mount=type=secret,id=aws_access_key \
    --mount=type=secret,id=aws_secret_key \
    --mount=type=secret,id=aws_region \
    set -eux && \
    export AWS_ACCESS_KEY_ID="$(cat /run/secrets/aws_access_key)" && \
    export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/aws_secret_key)" && \
    export AWS_DEFAULT_REGION="$(cat /run/secrets/aws_region)" && \
    export AWS_REGION="$(cat /run/secrets/aws_region)" && \
    \
    LATEST_PT_FILE=$(aws s3api list-objects-v2 \
      --bucket mlops-harry \
      --prefix mnist/ \
      --query "reverse(sort_by(Contents[?ends_with(Key, '.pt')], &LastModified))[0].Key" \
      --output text) && \
    \
    echo "Downloading latest model: $LATEST_PT_FILE" && \
    aws s3 cp "s3://mlops-harry/$LATEST_PT_FILE" /app/mnist_cnn.pt

# Default command
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]



