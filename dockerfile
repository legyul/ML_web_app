# 1. 베이스 이미지로 Python 3.9 버전을 사용
FROM python:3.11

# Java 설치
RUN apt-get update && apt-get install -y openjdk-17-jdk

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# 2. 컨테이너 안에서 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 3. 현재 디렉토리의 모든 파일을 컨테이너의 /app으로 복사
COPY src/ .

# 4. 필요 패키지를 설치
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# PySpark 설치
RUN pip install pyspark

# 5. 컨테이너 외부에서 접근할 수 있도록 5000번 포트를 노출
EXPOSE 5000

# 6. 환경 변수 설정 (Flask의 기본 실행 파일 설정)
ENV FLASK_APP=app.py

# 7. Flask 서버 실행
CMD ["flask", "run", "--host=0.0.0.0", "-p", "5000"]