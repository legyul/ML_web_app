FROM public.ecr.aws/lambda/python:3.11

# Working Directory Settings
WORKDIR /var/task

ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_CACHE=/tmp/hf_cache

RUN python3.11 -m pip install --upgrade pip

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y transformers && pip install transformers==4.38.2

# Copy app.py
COPY app.py .

# Specifying Lambda Handlers
CMD ["app.lambda_handler"]