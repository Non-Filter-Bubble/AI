from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List

app = FastAPI()

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

    return recommendations[['ISBN_THIRTEEN_NO', 'GENRE_LV2', 'similarity']].to_dict(orient='records')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
