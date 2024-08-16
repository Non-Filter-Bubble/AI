
# from fastapi import FastAPI, HTTPException
# import requests
# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World"}


from book_recommend import *
from fastapi import FastAPI, HTTPException
import requests
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://43.203.38.124",  # 프론트엔드 주소
    "http://43.203.38.124:8080",  # 백 서버 주소
    "http://3.37.204.233",  # AI 서버 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

SPRING_APP_URL = "http://43.203.38.124:8080"
#ai server : http://43.200.64.238:8000/

class GenreRequest(BaseModel):
    user_id: int
    genres: List[str]

@app.post("/ai/books")
async def process_genres(request: GenreRequest):
    user_id = request.user_id
    genres = request.genres

    # AI 모델을 이용한 처리 로직 (예시)

    result = process_genres_with_ai(user_id, genres)
    print(result)

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

    set_genres = set(genres)   # set으로 변환
    list_genres = list(set_genres) # list로 변환

    #gen=['로맨스','일반소설','자전']
    gen=list_genres

    gcn_book_list,filter_sim_book,sim_book=run_recommendation_system(gen)

    # GCN 추천 시스템에 넣을 책 리스트 나중에 주석 풀고 다시 해주세요
    print(gcn_book_list)
    #nonfilter_book=GCN_book(user_id,gcn_book_list)
    #nonfilter_book_list=flatten_and_convert(nonfilter_book)


    # # nonfilter
    # selected_slices, selected_keywords = get_slices_and_keywords_by_genres(genres)
    # sim_book = find_similar_books(embedder, selected_slices, selected_keywords)
    #sim_book_list = list(reduce(lambda x, y: x+y, sim_book))
    sim_book_list = flatten_and_convert(sim_book)
    print(sim_book_list)

    # # filter : 같은 장르 내에서 유사 아이템
    # filter_selected_slices, filter_selected_keywords = get_filter_slices_and_keywords_by_genres(genres)
    # filter_sim_book = find_similar_books(embedder, filter_selected_slices, filter_selected_keywords)


    #filter_sim_book_list = list(reduce(lambda x, y: x+y, filter_
    filter_sim_book_list = flatten_and_convert(filter_sim_book)
    print(filter_sim_book_list)

    return {
        "user_id": user_id,
        "isbn_nonfilter": sim_book_list,
        "isbn_filter": filter_sim_book_list,
    }

@app.get("/")
async def read_root():
    return {"message": "Hello, World with FASTAPI"}


@app.get("/test/{message}")
async def test_message(message: str):
    return {"test ": message}
