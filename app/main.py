# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World"}

from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

SPRING_APP_URL = "http://43.203.38.124:8080"

@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

@app.get("/test")
async def send_request_to_spring():
    try:
        # Spring 애플리케이션으로 GET 요청을 보냄
        response = requests.get(f"{SPRING_APP_URL}/test")
        response.raise_for_status()  # 오류가 발생하면 예외를 발생시킴
        
        # Spring으로부터 받은 응답을 반환
        return response.json()
    except requests.RequestException as e:
        # 요청이 실패한 경우 오류 메시지를 반환
        raise HTTPException(status_code=500, detail=f"Failed to connect to Spring application: {e}")
