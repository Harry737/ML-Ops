import json
import os
import urllib3

http = urllib3.PoolManager()

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]  # format: owner/repo

def lambda_handler(event, context):
    try:
        # EventBridge S3 event structure
        detail = event["detail"]

        bucket = detail["bucket"]["name"]
        key = detail["object"]["key"]

        print(f"Received upload: s3://{bucket}/{key}")

        # Only react to model file uploads
        if not key.endswith(".mar"):
            print("Not a model file. Ignoring.")
            return {"status": "ignored"}

        # Expected format:
        # mnist/<timestamp>/model/mnist_cnn.mar
        parts = key.split("/")

        if len(parts) < 4:
            raise Exception(f"Unexpected S3 key format: {key}")

        dataset = parts[0]          # mnist
        timestamp = parts[1]        # 1770651133

        model_path = f"s3://{bucket}/{dataset}/{timestamp}/"

        print(f"Resolved model path: {model_path}")

        payload = {
            "event_type": "model_uploaded",
            "client_payload": {
                "model_path": model_path
            }
        }

        url = f"https://api.github.com/repos/{REPO}/dispatches"

        response = http.request(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json"
            },
            body=json.dumps(payload)
        )

        print(f"GitHub response: {response.status}")

        if response.status not in [200, 201, 204]:
            raise Exception(f"GitHub API failed: {response.data}")

        return {"status": "triggered", "model_path": model_path}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise
