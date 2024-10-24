

from book_recommend import *
from fastapi import FastAPI, HTTPException
import requests
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from GCN import *

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://s3-nonfilterbubble-react.s3-website.ap-northeast-2.amazonaws.com",  # 프론트엔드 주소
    "http://13.209.250.36:8080",  # 백 서버 주소
    "http://13.124.44.42",  # AI 서버 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

SPRING_APP_URL = "http://13.209.250.36:8080"

class GenreRequest(BaseModel):
    user_id: int
    genres: List[str]

@app.post("/ai/books")
async def process_genres(request: GenreRequest):
    user_id = request.user_id
    genres = request.genres

    # AI 모델을 이용한 처리 로직 (예시)

    result = process_genres_with_ai(user_id, genres)
    #print(result)
    return result

from typing import List, Union, Dict, Any
from itertools import chain


def convert_to_python_type(value: Union[np.int64, int, float, str, bool]) -> Union[int, float, str, bool]:
    if isinstance(value, np.generic):
        return value.item()
    return value

# 리스트 평탄화 및 numpy 타입 변환
def flatten_and_convert(sim_book: List[List[Any]]) -> List[Any]:
    flattened_list = list(chain.from_iterable(sim_book))
    converted_list = [convert_to_python_type(item) for item in flattened_list]
    return converted_list

def process_genres_with_ai(user_id: int, genres: List[str]):

    user_id=user_id

    set_genres = set(genres)   # set으로 변환
    list_genres = list(set_genres) # list로 변환

    gen=list_genres

    gcn_book_list,filter_sim_book,sim_book,favor_genre,df_book=run_recommendation_system(gen,user_id)

    # GCN 추천 시스템에 넣을 책 리스트 나중에 주석 풀고 다시 해주세요
    filter_book0, filter_book=run_GCN(str(user_id),gcn_book_list)
    filter_book = list(map(str, filter_book))

    gcn_filtered_list,gcn_non_filtered_list=gcn_list_filter_with_favor_genre(df_book, filter_book, favor_genre)

    books_for_new=gcn_non_filtered_list
    books_for_you = list(set(filter_sim_book + gcn_filtered_list))
    books_for_you = random.sample(books_for_you, 70)

    books_for_new = [isbn for isbn in books_for_new if len(isbn) == 13]
    books_for_you = [isbn for isbn in books_for_you if len(isbn) == 13]



    return {
        "user_id": user_id,
        "isbn_nonfilter": books_for_new,
        "isbn_filter": books_for_you,
    }

@app.get("/")
async def read_root():
    return {"message": "Hello, World with FASTAPI"}


@app.get("/test/{message}")
async def test_message(message: str):
    return {"test ": message}


