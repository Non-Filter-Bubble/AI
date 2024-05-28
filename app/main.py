# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World"}

from fastapi import FastAPI
import requests

app = FastAPI()

SPRING_APP_URL = "http://43.203.38.124:8080"

@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

@app.get("/test")
async def send_request_to_spring():
    try:
        response = requests.get(f"{SPRING_APP_URL}/test")
        response.raise_for_status()  # 오류가 발생하면 예외를 발생시킴
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to connect to Spring application: {e}"}

