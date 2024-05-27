from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List
import requests

app = FastAPI()

# CORS 설정
origins = [
    "http://43.203.38.124:8080",  # Spring 애플리케이션의 도메인 추가
    # 필요 시 추가 도메인 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 로드
df = pd.read_csv('last_book.csv')

# Sentence Transformer 모델 로드
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

class KeywordsRequest(BaseModel):
    keywords: List[str]

@app.post("/recommend")
async def recommend(request: KeywordsRequest):
    user_keywords = " ".join(request.keywords)
    user_embedding = embedder.encode(user_keywords, convert_to_tensor=True)

    # 모든 책에 대해 임베딩 계산
    df['embedding'] = df['KEYWORD_oneline'].apply(lambda x: embedder.encode(x, convert_to_tensor=True))
    # 코사인 유사도 계산
    df['similarity'] = df['embedding'].apply(lambda x: util.pytorch_cos_sim(user_embedding, x).item())

    # 유사도가 높은 상위 5권 추천
    recommendations = df.sort_values(by='similarity', ascending=False).head(5)

    # Spring 애플리케이션에 데이터 전송
    spring_app_url = "http://43.203.38.124:8080/recommend"
    response = requests.post(spring_app_url, json=recommendations[['ISBN_THIRTEEN_NO', 'GENRE_LV2', 'similarity']].to_dict(orient='records'))
    
    return response.json()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
