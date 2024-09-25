# Use Python 3.11 for base image
FROM python:3.11

# Install Java
RUN apt-get update && apt-get install -y openjdk-17-jdk

# Set JAVA_HOME environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# 2. Set /app for working directory in the container
WORKDIR /app

# 3. Copy all files in the current directory to /app in the container
COPY src/ .
COPY ./src /app/src
COPY ./templates /app/templates
COPY ./static /app/static
COPY ./data /app/data
COPY requirements.txt /app/

# 4. Install requirement files
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# PySpark 설치
#RUN pip install pyspark

# 5. Expose port 5000 for access from outside the container
EXPOSE 5000

# 6. Setting environment variables (setting the default executable in Flask)
ENV FLASK_APP=app.py

# 7. Run Flask server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]