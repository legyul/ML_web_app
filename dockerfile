# Use Python 3.11 for base image
FROM python:3.11

# Set /app for working directory in the container
WORKDIR /app

ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg

RUN echo "deb http://deb.debian.org/debian bullseye main" > /etc/apt/sources.list.d/bullseye.list && \
    apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install Java and dependencies
#RUN apt-get update && apt-get install -y default-jdk curl && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz \
    && tar -xvzf spark-3.5.2-bin-hadoop3.tgz \
    && mv spark-3.5.2-bin-hadoop3 /usr/local/spark \
    && rm spark-3.5.2-bin-hadoop3.tgz

# Upgrade pip and install necessary packages
#RUN pip install --no-cache-dir --upgrade pip setuptools

# Install Spark
# ENV SPARK_VERSION=3.3.0
# RUN wget -qO - "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" | tar -xz -C /usr/local/ && \
#     mv /usr/local/spark-${SPARK_VERSION}-bin-hadoop3 /usr/local/spark


# Set Spark environment variables for PySpark
# Set Spark environment variables
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install requirement files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files in the current directory to /app in the container
COPY ./src /app/src
COPY ./templates /app/templates
COPY ./static /app/static
#COPY ./data /app/data
COPY requirements.txt /app/

RUN export $(cat .env | xargs)



# Download necessary nltk resources
#RUN python -m nltk.downloader stopwords punkt

# Set Flask environment
#ENV FLASK_APP=src/app.py

# # 5. Expose port 5000 for access from outside the container
# EXPOSE 80

# 7. Run Flask server
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "300", "--worker-class", "gthread", "--threads", "4", "src.app:app"]
