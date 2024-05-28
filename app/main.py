# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import requests

# app = FastAPI()

# # CORS 설정
# origins = [
#     "http://43.200.64.238",   # FastAPI 애플리케이션의 도메인
#     "http://43.203.38.124:8080",  # Spring 애플리케이션의 도메인
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "Hello from FastAPI!"}

# @app.get("/test")
# async def test():
#     # Spring 애플리케이션에 요청 보내기
#     spring_app_url = "http://43.203.38.124:8080/test"
#     response = requests.get(spring_app_url)
    
#     return {"response_from_spring": response.text}

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=80)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World"}
