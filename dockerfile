# Use Python 3.11 for base image
FROM python:3.11

# Install Java and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH
#ENV PYTHONPATH="/app/src:$PYTHONPATH"

# 2. Set /app for working directory in the container
WORKDIR /app

# 3. Copy all files in the current directory to /app in the container
COPY ./src /app/src
COPY ./templates /app/templates
COPY ./static /app/static
COPY ./data /app/data
COPY requirements.txt /app/

# 4. Install requirement files
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=src/app.py

# 5. Expose port 5000 for access from outside the container
EXPOSE 80

# 7. Run Flask server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "src.app:app"]
