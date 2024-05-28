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

@app.get("/test/{message}")
async def test_message(message: str):
    return {"test": message}