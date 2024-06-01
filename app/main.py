
# from fastapi import FastAPI, HTTPException
# import requests
# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World"}


from fastapi import FastAPI, HTTPException
import requests
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



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

    # result = process_genres_with_ai(user_id, genres)
    result={
        "user_id": user_id,
        "isbn":
            [[9788956604992, 9791158887544, 9788956057842, 9791185851204, 9791171251360, 9791158887568, 9791158887575, 9791158887582, 9791158887599, 9788983927668, 9788983927675, 9788983927644, 9788983927651, 9788990982704, 9791168341821, 9791193190036, 9791193190036, 9791193190036, 9788968970986, 9788954442145, 9788990982575, 9788934985051, 9788983927620, 9788932906744, 9791168340947],
                [9791164052455, 9788950992286, 9788990982704, 9788994343990, 9791165653330, 9788965749219, 9788960177765, 9791171251360, 9788952240569, 9788971848722, 9788947545419, 9788956604992, 9788990982612, 9791164280520, 9788965749257, 9788965749240, 9788965749295, 9788965749301, 9788925569154, 9788982738012, 9788932906744, 9788952227829, 9788965749226, 9788965749233, 9791188862290],
                [9788971992258, 9791171251360, 9788932919553, 9791188862290, 9791191248739, 9788956604992, 9791168340947, 9791168340947, 9791191043365, 9788975270062, 9791160408331, 9788934985051, 9791189015381, 9788976041470, 9788956609959, 9791187498186, 9788952240569, 9791165653330, 9788935630912, 9791190174756, 9791190174756, 9788990982704, 9791193024409, 9788930088824, 9788968332364],
                [9788932909998, 9791158887605, 9791189433550, 9788994343587, 9791196205591, 9788952234247, 9788983927767, 9791196443146, 9791187589419, 9788954676465, 9791127864606, 9791198084651, 9791197708572, 9788946422056, 9791167850249]]
    }
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
