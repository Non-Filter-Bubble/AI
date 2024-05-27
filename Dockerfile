# Build stage
FROM python:3.8-slim AS builder

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Production stage
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 AS production

COPY --from=builder /app /app

# Install Nginx
RUN apt-get update \
    && apt-get install -y nginx

# Copy Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Expose ports
EXPOSE 80

# Start Nginx and FastAPI application
CMD service nginx start && uvicorn app:app --host 0.0.0.0 --port 80
