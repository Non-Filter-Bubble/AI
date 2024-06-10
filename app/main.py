
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
#     result={
#         "user_id": user_id,
#         "isbn-nonfilter":
#             [9788932919553, 9788925569154, 9788965749257, 9791160408331, 9791158887568, 9788971848722, 9791193190036, 9791158887575, 9791193024409, 9791158887582, 9788950992286, 9791171251360, 9788954442145, 9791190174756, 9791191043365, 9788956609959, 9788975270062, 9791158887599, 9788990982704, 9788990982575, 9788965749295, 9788965749301, 9788994343990, 9788952240569, 9791168341821, 9788956604992, 9788935630912,
#  9788971992258, 9791185851204, 9788983927620, 9791164280520, 9791187498186, 9788947545419, 9788968332364, 9788932906744, 9791165653330, 9791168340947, 9788990982612, 9791188862290, 9791189015381, 9788930088824, 9788934985051, 9788983927644, 9788982738012, 9788983927651, 9788965749219, 9788960177765, 9791191248739, 9791164052455, 9788965749240, 9788968970986, 9788965749226, 9788965749233, 9788956057842, 9788983927668, 9788952227829, 9791158887544, 9788983927675, 9788976041470],
#         "isbn-filter" :
#                 [9788932909998, 9791158887605, 9791189433550, 9788994343587, 9791196205591, 9788952234247, 9788983927767, 9791196443146, 9791187589419, 9788954676465, 9791127864606, 9791198084651, 9791197708572, 9788946422056, 9791167850249]
#     }
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
    # AI 모델 로직 예시
    # 실제로는 여기서 AI 모델을 사용하여 추천 및 기타 작업을 수행
    # return {
    #     "user_id": user_id,
    #     "recommended_movie": "Inception",
    #     "confidence_score": 0.95
    # }
    # 리스트 중복 제거
    set_genres = set(genres)   # set으로 변환
    list_genres = list(set_genres) # list로 변환

    # nonfilter
    selected_slices, selected_keywords = get_slices_and_keywords_by_genres(genres)
    sim_book = find_similar_books(embedder, selected_slices, selected_keywords)

    #sim_book_list = list(reduce(lambda x, y: x+y, sim_book))
    sim_book_list = flatten_and_convert(sim_book)
    print(sim_book_list)

    # filter : 같은 장르 내에서 유사 아이템
    filter_selected_slices, filter_selected_keywords = get_filter_slices_and_keywords_by_genres(genres)
    filter_sim_book = find_similar_books(embedder, filter_selected_slices, filter_selected_keywords)


    #filter_sim_book_list = list(reduce(lambda x, y: x+y, filter_
    #sim_book
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
