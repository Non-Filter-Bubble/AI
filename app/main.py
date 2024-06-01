# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World"}

from fastapi import FastAPI, HTTPException
import requests
import uvicorn
from pydantic import BaseModel
from typing import List


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPRING_APP_URL = "http://43.203.38.124:8080"
#ai server : http://43.200.64.238:8000/

class GenreRequest(BaseModel):
    user_id: str
    genres: List[str]

@app.post("/ai/books")
async def process_genres(request: GenreRequest):
    user_id = request.user_id
    genres = request.genres

    # AI 모델을 이용한 처리 로직 (예시)
    result = process_genres_with_ai(user_id, genres)
    # result={
    #     "user_id": user_id,
    #     "isbn": [[123,156,456],[123,156,456]]
    # }

    return result

def process_genres_with_ai(user_id: str, genres: List[str]):
    # AI 모델 로직 예시
    # 실제로는 여기서 AI 모델을 사용하여 추천 및 기타 작업을 수행
    return {
        "user_id": user_id,
        "recommended_movie": "Inception",
        "confidence_score": 0.95
    }

@app.get("/")
async def read_root():
    return {"message": "Hello, World with FASTAPI"}


@app.get("/test/{message}")
async def test_message(message: str):
    return {"test ": message}

