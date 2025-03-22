FROM public.ecr.aws/lambda/python:3.11

# Working Directory Settings
WORKDIR /var/task

ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_CACHE=/tmp/hf_cache

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py
COPY app.py .

# Specifying Lambda Handlers
CMD ["app.lambda_handler"]