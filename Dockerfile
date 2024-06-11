# 이미지 기반으로는 Python을 사용
FROM python:3.9-slim AS base

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 시스템 종속성 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# 컨테이너 내에서 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일을 작업 디렉토리에 복사
COPY requirements.txt .

# requirements.txt에 명시된 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install pandas
RUN pip install sentence-transformers

# 애플리케이션 코드를 작업 디렉토리로 복사
COPY . .
COPY /app/last_book.csv .

# 외부로 포트 8000을 노출
EXPOSE 8000

# FastAPI 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
