

import pandas as pd
import numpy as np

import torch
from sklearn.metrics.pairwise import cosine_similarity

# Data loading
def load_data():
    #df_book = pd.read_pickle('book_embedding_yes.pkl')
    df_book = pd.read_pickle('book_embedding_yes_temp.pkl')
    df_recom=pd.read_pickle('for_recommend_df.pkl')
    #df_book = pd.read_pickle('book_embedding.pkl')
    df_genre = pd.read_pickle('genre_embedding.pkl')
    df_book = df_book.reset_index()
    return df_book,df_recom, df_genre

# Genre selection based on user preference
def get_favor_genre(gen, genre_match_dict, genre_dict):
    select = []
    for i in gen:
        select.append(genre_match_dict[i])
    favor_genre = [item for sublist in [genre_dict.get(i) for i in select] for item in sublist]
    return select, favor_genre


# Filter data based on preferred genre
def filter_by_favor_genre(df_book, df_recom, favor_genre):
    filter_df = pd.DataFrame()
    gcn_book=pd.DataFrame()
    for genre in favor_genre:
        tmp = df_book[df_book['GENRE_LV2'] == genre].sample(20)
        tmp_gcn = df_recom[df_recom['GENRE_LV2'] == genre].sample(3,replace=True)

        filter_df = pd.concat([filter_df, tmp])
        gcn_book = pd.concat([gcn_book, tmp_gcn])

    filter_list = filter_df['ISBN_THIRTEEN_NO'].tolist()
    gcn_book_list=gcn_book['ISBN_THIRTEEN_NO'].tolist()
    return filter_list,gcn_book_list

# Remove preferred genres from similar genres
def remove_favor_genre_from_similar(favor_genre, similar_genre):
    for value in favor_genre:
        while value in similar_genre:
            similar_genre.remove(value)
    return similar_genre

# Get similar genres
def get_similar_genre(select, genre_sim_dict, favor_genre):
    similar_genre = [item for sublist in [genre_sim_dict.get(i) for i in select] for item in sublist]
    similar_genre = remove_favor_genre_from_similar(favor_genre, similar_genre)
    return similar_genre

# Filter data based on similar genres
def filter_by_similar_genre(df_book, similar_genre):
    slice_df = pd.DataFrame()
    for genre in similar_genre:
        tmp = df_book[df_book['GENRE_LV2'] == genre]
        slice_df = pd.concat([slice_df, tmp])
    return slice_df

# Filter data based on similar genres
def gcn_list_filter_with_favor_genre(df_book, gcn_nonfilter_list,favor_genre):
    temp=df_book[df_book['ISBN_THIRTEEN_NO'].isin(gcn_nonfilter_list)]
    df_nonfiltered=temp[~temp['GENRE_LV2'].isin(favor_genre)]
    df_filtered = temp[temp['GENRE_LV2'].isin(favor_genre)]
    gcn_filtered_list=df_filtered['ISBN_THIRTEEN_NO'].tolist()
    gcn_nonfiltered_list = df_nonfiltered['ISBN_THIRTEEN_NO'].tolist()
    return gcn_filtered_list,gcn_nonfiltered_list

# Calculate cosine similarity and retrieve top-k results
def get_top_k_recommendations(select, genre_embedding_corpus, book_embedding_corpus, df_genre, df_book, top_k=100):
    results = []
    num = 0
    for i in select:
        cos_scores = cosine_similarity([genre_embedding_corpus[i]], book_embedding_corpus)[0]
        sorted_indices = np.argsort(-cos_scores)
        top_results = sorted_indices[:top_k]
        li_top = []
        for idx in top_results:
            li_top.append(df_book['ISBN_THIRTEEN_NO'][idx])

        results.append(li_top)
        num += 1
    return [item for sublist in results for item in sublist]

# Main function to run the recommendation system
def run_recommendation_system(gen,user_id):
    # Load data
    df_book,df_recom, df_genre = load_data()
    user_id=user_id
    # Genre mapping dictionaries
    genre_match_dict = {
        "로맨스": 1, "자전": 2, "일반소설": 3, "판타지": 4, "공포/스릴러": 5, "자연과학": 6,
        "생명과학": 7, "자기계발": 8, "인문학": 9, "역사": 10, "한국시": 11, "커뮤니케이션": 12,
        "시간관리": 13, "인간관계": 14, "경제/경영": 15
    }

    genre_dict = {
        1: ['로맨스', '사랑/연애'], 2: ['인물/자전적', '명상/치유', '일기/편지', '여행', '교양에세이'],
        3: ['일반소설'], 4: ['sf/과학', '판타지'], 5: ['공포/호러', '미스터리/스릴러'], 6: ['물리학', '천문', '지구과학', '수학'],
        7: ['생물학', '화학'], 8: ['성공/처세', '자기능력계발'], 9: ['심리학', '철학', '인문교양', '사회'], 10: ['역사'],
        11: ['한국시', '일반시'], 12: ['화술/협상'], 13: ['시간관리'], 14: ['인간관계'], 15: ['경제', '경영']
    }

    genre_sim_dict = {
        1: ['일반소설', '한국시', '일반시', '인간관계', '공포/호러', '미스터리/스릴러', '시간관리'],
        2: ['화술/협상', '역사', '한국시', '일반시', '공포/호러', '미스터리/스릴러', '인간관계'],
        3: ['한국시', '일반시', '로맨스', '사랑/연애', '공포/호러', '미스터리/스릴러', '인간관계', 'sf/과학', '판타지'],
        4: ['한국시', '일반시', '생물학', '화학', '역사', '물리학', '천문', '지구과학', '수학', '공포/호러', '미스터리/스릴러'],
        5: ['한국시', '일반시', '역사', '인물/자전적', '명상/치유', '일기/편지', '여행', '교양에세이', 'sf/과학', '판타지', '일반소설'],
        6: ['생물학', '화학', '화술/협상', '공포/호러', '미스터리/스릴러', '심리학', '철학', '인문교양', '사회'],
        7: ['물리학', '천문', '지구과학', '수학', 'sf/과학', '판타지', '한국시', '일반시', '역사', '성공/처세', '자기능력계발'],
        8: ['시간관리', '생물학', '화학', '인물/자전적', '명상/치유', '일기/편지', '여행', '교양에세이', '화술/협상', '심리학', '철학', '인문교양', '사회'],
        9: ['인간관계', '시간관리', '화술/협상', '물리학', '천문', '지구과학', '수학', '생물학', '화학'],
        10: ['생물학', '화학', '인물/자전적', '명상/치유', '일기/편지', '여행', '교양에세이', 'sf/과학', '판타지', '한국시', '일반시', '공포/호러', '미스터리/스릴러'],
        11: ['sf/과학', '판타지', '공포/호러', '미스터리/스릴러', '생물학', '화학', '역사', '일반소설'],
        12: ['인물/자전적', '명상/치유', '일기/편지', '여행', '교양에세이', '인간관계', '시간관리', '심리학', '철학', '인문교양', '사회', '경제', '경영'],
        13: ['심리학', '철학', '인문교양', '사회', '경제', '경영', '화술/협상', '성공/처세', '자기능력계발'],
        14: ['심리학', '철학', '인문교양', '사회', '화술/협상', '시간관리', '경제', '경영', '인물/자전적', '명상/치유', '일기/편지', '여행', '교양에세이'],
        15: ['시간관리', '인간관계', '화술/협상', '역사', 'sf/과학', '판타지']
    }
    #선호 장르 뽑기
    select, favor_genre = get_favor_genre(gen, genre_match_dict, genre_dict)

    #선호 장르 기반 도서 추천 리스트 추출 및 GCN 책 리스트 추출
    filter_isbn_list,gcn_book_list= filter_by_favor_genre(df_book, df_recom, favor_genre)

    #Nonfilter with embedding
    similar_genre=get_similar_genre(select,genre_sim_dict,favor_genre)
    slice_df=filter_by_similar_genre(df_book,similar_genre)

    # book_embedding = slice_df['embedding2'].tolist()
    # book_embedding_corpus = torch.tensor(book_embedding)


    genre_embedding = df_genre['embedding2'].tolist()
    genre_embedding_corpus = torch.tensor(genre_embedding)

    #nonfilter_isbn_list=get_top_k_recommendations(select,genre_embedding_corpus, book_embedding_corpus,df_genre, df_book, top_k=100)
    nonfilter_isbn_list=[]

    # print("GCN : " ,gcn_book_list)
    #
    # print("FILTER : " ,filter_isbn_list)
    # print("NONFILTER : " ,nonfilter_isbn_list)

    return gcn_book_list,filter_isbn_list,nonfilter_isbn_list,favor_genre,df_book



