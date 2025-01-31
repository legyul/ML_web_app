import logging
from io import BytesIO
import os

def setup_global_logger(s3=None, bucket_name=None, log_level=logging.DEBUG, log_filename='default_log'):
    '''
    Setup a global logger with the specified log file name and level.
    If `s3` is provided, it will upload logs to S3.

    Parameters:
    - s3 (boto3.client): S3 클라이언트
    - log_level (int): Logging level (e.g., DEBUG, INFO)
    - log_filename (str): The filename of the log (default is 'default_log.log')

    Returns:
    - logger: Configured logger
    - upload_log_to_s3: Function to upload log to S3
    '''
    log_dir = './_logs'
    os.makedirs(log_dir, exist_ok=True)

    log_buffer = BytesIO()
    logger = logging.getLogger('global_logger')
    logger.setLevel(log_level)

    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    def upload_log_to_s3():
        if s3:
            log_buffer.seek(0)
            s3.upload_fileobj(log_buffer, bucket_name, f'result/{log_filename}_log.log')
            print(f"Log file {log_filename}_log uploaded to S3.")
        else:
            print(f"Log file {log_filename}_log is stored locally.")

    return logger, upload_log_to_s3