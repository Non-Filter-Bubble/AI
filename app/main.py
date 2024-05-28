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
    async def echo_message(message: str):
        return message 

