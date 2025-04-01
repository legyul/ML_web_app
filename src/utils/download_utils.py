import os
import io
import shutil
import boto3
from botocore.exceptions import NoCredentialsError
import zipfile
import pickle
from utils.logger_utils import logger
from huggingface_hub import snapshot_download

def download_llm_model_from_s3(S3_REGION, S3_BUCKET_NAME, s3_model_path, local_dir, required_files):
    """
    Download LLM model files (e.g. for RAG or LoRA) from S3 to local directory

    Parameters
    - s3_region: AWS S3 region
    - s3_bucket_name: S3 bucket name
    - s3_model_path: path to the model inside the bucket (e.g. "models/tinyllama_model/")
    - local_dir: local directory to save model files (e.g. "/tmp/tinyllama_model")
    - required_files: optional set of file name to download; if None, download all found files
    """
    from botocore.exceptions import ClientError
    logger.info(f"Checking if model exists at: {local_dir}")

    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        logger.info("Model already exists locally.")
        return
    
    logger.info(f"Model not found locally. Downloading from S3: {s3_model_path}")

    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_model_path)

        if "Contents" not in response:
            logger.error(f"No files found in S3 at {s3_model_path}")
            return

        for obj in response["Contents"]:
            key = obj["Key"]
            filename = os.path.basename(key)

            if not filename or filename.startswith(".") or filename.endswith("/"):
                continue

            for obj in response["Contents"]:
                key = obj["Key"]
                logger.info(f"[S3] Found key: {key}")

            if required_files is None or filename in required_files:
                dest_path = os.path.join(local_dir, filename)
                logger.info(f"files: {filename}")
                
                try:
                    logger.debug(f"Downloading: {filename}")
                    s3.download_file(S3_BUCKET_NAME, key, dest_path)
                    logger.info("Completed to download the model")
                except ClientError as e:
                    logger.error(f"Failed to download {filename} from S3: {e}")

            
        else:
            logger.error(f"No model files found in S3 path: {s3_model_path}")
    
    except NoCredentialsError:
        print("AWS credentials not found! Run 'aws configure' or check environment.")
    except Exception as e:
        print(f"Failed to download model from S3: {e}")


def load_model_from_s3(s3_key, model_filename="model.pkl"):
    """
    Download and load the trained model from S3

    Parameter
    - s3_key (string): The path of the model file on S3

    Return
    - Loaded model object
    """

    S3_REGION = "us-east-2"
    S3_BUCKET_NAME = "ml-platform-service"

    s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

    try:
        print(f"Downloading model from S3: {s3_key}")
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        zip_data = response['Body'].read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            if model_filename not in zf.namelist():
                raise FileNotFoundError(f"'{model_filename}' not found in zip archive")
            
            with zf.open(model_filename) as model_file:
                model = pickle.load(model_file)
                print("Complete to load model")
                return model
    
    except Exception as e:
        print(f"Fail to load model: {e}")
        return None


def download_model_from_huggingface():
    HF_MODEL_ID = "distilgpt2"
    HF_LOCAL_DIR = "/tmp/distilgpt2"

    REQUIRED_FILES = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors",
    ]

    if all(os.path.exists(os.path.join(HF_LOCAL_DIR, f)) for f in REQUIRED_FILES):
        print("[DEBUG] Model already fully exists. Skipping download.")
        return

    print("[DEBUG] Downloading model from Hugging Face...")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 안정성 증가

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=HF_LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.tflite", "*.ot", "*.mlmodel"],  # 용량 큰 필요 없는 파일 제거
    )
    print("[DEBUG] Model download complete.")