import logging
import io
import boto3
from datetime import datetime

S3_BUCKET_NAME = "ml-platform-service"
s3 = boto3.client("s3")

log_buffer = io.StringIO()

def setup_global_logger(log_level=logging.DEBUG, log_filename='default_log'):
    '''
    Setup a global logger with the specified log file name and level.
    If `s3` is provided, it will upload logs to S3.

    Parameters:
    - log_level (int): Logging level (e.g., DEBUG, INFO)
    - log_filename (str): The filename of the log (default is 'default_log.log')

    Returns:
    - logger: Configured logger
    - upload_log_to_s3: Function to upload log to S3
    '''
    logger = logging.getLogger('AppLogger')
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    class UTCFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            utc_dt = datetime.utcfromtimestamp(record.created)
            return utc_dt.strftime('%Y-%m-%d %H:%M:%S')  # UTC 시간 저장

    formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')

    # Memory Log Storage Handler
    stream_handler = logging.StreamHandler(log_buffer)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Log File Storage Handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Save log file name
    logger.log_filename = log_filename
    logger.log_buffer = log_buffer


    return logger

def upload_log_to_s3():
    if log_buffer.getvalue():
        log_content = log_buffer.getvalue()
        log_buffer.seek(0)

        s3_key = f"logs/{logger.log_filename}_log.log"
        s3.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=log_content)

        print(f"[DEBUG] Log uploaded to S3: {s3_key}")
    
    else:
        print("[ERROR] Log file is empty! Check log writing process!")
    
logger = setup_global_logger()