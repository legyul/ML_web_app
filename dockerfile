# Use Python 3.11 for base image
FROM python:3.11

# Set /app for working directory in the container
WORKDIR /app

ENV PYTHONPATH=/app/src

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
    software-properties-common \
    gnupg \
    ca-certificates

# Download and install OpenJDK 11 (Adoptium Temurin JDK)
RUN mkdir -p /usr/lib/jvm && \
    curl -fsSL https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz -o jdk11.tar.gz && \
    tar -xzf jdk11.tar.gz -C /usr/lib/jvm && \
    mv /usr/lib/jvm/jdk-11 /usr/lib/jvm/java-11-openjdk-amd64 && \
    rm jdk11.tar.gz

# Install Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz \
    && tar -xvzf spark-3.5.2-bin-hadoop3.tgz \
    && mv spark-3.5.2-bin-hadoop3 /usr/local/spark \
    && rm spark-3.5.2-bin-hadoop3.tgz

# Install requirement files
RUN pip install --upgrade pip
#RUN pip install --no-cache-dir pip setuptools wheel packaging Pillow pyparsing cycler
RUN pip install --no-cache-dir numpy pandas scikit-learn matplotlib seaborn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade numpy pandas scikit-learn

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
