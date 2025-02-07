# Use Python 3.11 for base image
FROM python:3.11

# Set /app for working directory in the container
WORKDIR /app

ENV PYTHONPATH=/app/src

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    gzip \
    openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Download and install OpenJDK 11 (Adoptium Temurin JDK)
# RUN mkdir -p /usr/lib/jvm && \
#     cd /usr/lib/jvm && \
#     curl -L -o openjdk11.tar.gz https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.20%2B8/OpenJDK11U-jdk_x64_linux_hotspot_11.0.20_8.tar.gz && \
#     tar -xvzf openjdk11.tar.gz && \
#     mv jdk-11.0.20+8 java-11-openjdk-amd64 && \
#     rm openjdk11.tar.gz

# Install Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz \
    && tar -xvzf spark-3.5.2-bin-hadoop3.tgz \
    && mv spark-3.5.2-bin-hadoop3 /usr/local/spark \
    && rm spark-3.5.2-bin-hadoop3.tgz

# Install requirement files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files in the current directory to /app in the container
COPY ./src /app/src
COPY ./templates /app/templates
COPY ./static /app/static
COPY requirements.txt /app/

#RUN export $(cat .env | xargs)



# Download necessary nltk resources
#RUN python -m nltk.downloader stopwords punkt

# Set Flask environment
#ENV FLASK_APP=src/app.py

# # 5. Expose port 5000 for access from outside the container
# EXPOSE 80

# 7. Run Flask server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "300", "--worker-class", "gthread", "--threads", "2", "src.app:app"]
