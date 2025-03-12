# Use Python 3.11 for base image
# FROM python:3.11
# Use Amazon Linux 2023 with Python 3.11
FROM amazonlinux:2023

# Set /app for working directory in the container
WORKDIR /app

ENV PYTHONPATH=/app/src

# Install dependencies
RUN yum update -y && yum install -y --allowerasing \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    tar \
    gzip \
    wget \
    curl \
    shadow-utils \
    java-11-amazon-corretto \
    && yum clean all

# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz -O spark.tgz && \
    tar -xzf spark.tgz && \
    mv spark-3.5.2-bin-hadoop3 /usr/local/spark && \
    rm spark.tgz

# Set Spark environment variables
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3


# Download and install OpenJDK 11 (Adoptium Temurin JDK)
# RUN mkdir -p /usr/lib/jvm && \
#     curl -fsSL https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz -o jdk11.tar.gz && \
#     tar -xzf jdk11.tar.gz -C /usr/lib/jvm && \
#     mv /usr/lib/jvm/jdk-11 /usr/lib/jvm/java-11-openjdk-amd64 && \
#     rm jdk11.tar.gz

# Install Spark
# RUN curl -O https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz \
#     && tar -xvzf spark-3.5.2-bin-hadoop3.tgz \
#     && mv spark-3.5.2-bin-hadoop3 /usr/local/spark \
#     && rm spark-3.5.2-bin-hadoop3.tgz

# Install Hadoop Native Library
RUN mkdir -p /usr/local/hadoop/lib/native && \
    wget -q https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz -O hadoop.tar.gz && \
    tar -xzf hadoop.tar.gz && \
    mv hadoop-3.3.6/lib/native/* /usr/local/hadoop/lib/native/ && \
    rm -rf hadoop-3.3.6 hadoop.tar.gz

# Set Hadoop environment variables
ENV HADOOP_OPTS="-Djava.library.path=/usr/local/hadoop/lib/native"
ENV LD_LIBRARY_PATH="/usr/local/hadoop/lib/native:$LD_LIBRARY_PATH"

# Install requirement files
# Upgrade pip
RUN python3.11 -m pip install --upgrade pip
# RUN pip install --upgrade pip
#RUN pip install --no-cache-dir pip setuptools wheel packaging Pillow pyparsing cycler

RUN yum install -y procps && yum clean all

# Install essential Python libraries (미리 설치)
RUN python3.11 -m pip install --no-cache-dir numpy pandas scikit-learn matplotlib seaborn && \
    python3.11 -m pip install --no-cache-dir pyspark nltk peft transformers datasets accelerate && \
    rm -rf /root/.cache/pip

RUN yum install -y procps && yum clean all

# Copy all files in the current directory to /app in the container
COPY ./src /app/src
COPY ./templates /app/templates
COPY ./static /app/static
COPY requirements.txt /app/

# Install remaining dependencies from requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# 7. Run Flask server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "300", "--worker-class", "gthread", "--threads", "1", "src.app:app"]
