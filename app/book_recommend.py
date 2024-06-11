import random
import re
import pandas as pd
import numpy as np
from functools import reduce

# import pickle
# with open('embedder.pickle', 'rb') as f:
#     embedder = pickle.load(f)

df=pd.read_csv('last_book.csv',index_col=False)


def remove_special_characters_and_make_list(input_string):
    # 특수문자 제거
    cleaned_string = re.sub(r'[^\w\s,]', '', input_string)

    # 쉼표(,)로 구분된 리스트로 변환
    result_list = cleaned_string.split(',')

    # 공백 제거 및 빈 문자열 제거
    result_list = [element.strip() for element in result_list if element.strip()]

    return result_list

# 특수문자 제거 및 리스트 생성
df["KEYWORD_list"]=list(map(remove_special_characters_and_make_list,df["KEYWORD"]))


def list_to_string(lst):
    return ' '.join(lst)

df['KEYWORD_oneline'] = list(map(lambda x: list_to_string(x), df['KEYWORD_list']))

df_horror=df[df['GENRE_LV2'] == '공포/호러']
df_romance=df[df['GENRE_LV2'] == '로맨스']
df_mystery=df[df['GENRE_LV2'] == '미스터리/스릴러']
df_sf=df[df['GENRE_LV2'] == 'SF/과학']
df_fantasy=df[df['GENRE_LV2'] == '판타지']
df_history=df[df['GENRE_LV2'] == '역사']
df_physics=df[df['GENRE_LV2'] == '물리학']
df_biology=df[df['GENRE_LV2'] == '생명과학']
df_math=df[df['GENRE_LV2'] == '수학']
df_star=df[df['GENRE_LV2'] == '천문학']
df_earth=df[df['GENRE_LV2'] == '지구과학']
df_chemi=df[df['GENRE_LV2'] == '화학']
df_phsyco=df[df['GENRE_LV2'] == '심리학']
df_iron=df[df['GENRE_LV2'] == '철학']
df_inmun=df[df['GENRE_LV2'] == '인문교양']
df_korPoem=df[df['GENRE_LV2'] == '한국 시']
df_trip=df[df['GENRE_LV2'] == '여행']
df_love=df[df['GENRE_LV2'] == '사랑/연애']
df_human=df[df['GENRE_LV2'] == '인물/자전적']
df_diary=df[df['GENRE_LV2'] == '일기/편지']
df_medit=df[df['GENRE_LV2'] == '명상/치유']
df_speak=df[df['GENRE_LV2'] == '화술/협상']
df_success=df[df['GENRE_LV2'] == '성공/처세']
df_time=df[df['GENRE_LV2'] == '시간관리']
df_abil=df[df['GENRE_LV2'] == '자기능력계발']
df_relation=df[df['GENRE_LV2'] == '인간관계']
df_economy=df[df['GENRE_LV2'] == '경제']
df_manage=df[df['GENRE_LV2'] == '경영']

genre_df_list=[df_horror,df_romance,df_mystery,df_sf,df_fantasy,df_history,
               df_physics,df_biology,df_math,df_star,df_earth,df_chemi,df_phsyco,
               df_iron,df_inmun,df_korPoem,df_trip,df_love,df_human,df_diary,
               df_medit,df_speak,df_success,df_time,df_abil,df_relation,df_economy,df_manage]

combined_genre=['로맨스/사랑/연애',
 '인물/자전적+명상/치유+일기/편지+여행',
 'sf/과학+판타지',
 '공포/호러+미스터리/스릴러',
 '물리학+천문+지구과학+수학',
 '생물학+화학',
 '성공/처세+자기능력계발',
 '경제+경영',
 '심리학+철학+인문교양',
 '역사',
 '한국시',
 '화술/협상',
 '시간관리',
 '인간관계']

genre_keyword=[]
for s in genre_df_list:
  tmp=[]
  for i in s:
    if i=="KEYWORD_list":
      for k in s[i]:
        for j in k:
          tmp.append(j)
  genre_keyword.append(tmp)

genre=['공포/호러','로맨스','미스터리/스릴러','SF/과학',
       '판타지','역사','물리학','생물학','수학','천문','지구과학',
       '화학','심리학','철학','인문교양','한국 시','여행','사랑/연애','인물/자전적',
       '일기/편지','명상/치유','화술/협상','성공/처세','시간관리','자기능력계발','인간관계','경제','경영']
df_genre_keyword=pd.DataFrame(data=[genre_keyword],columns=genre)


love_keywords = [
    '사랑', '연애', '로맨스', '감정', '만남', '이별', '애정', '짝사랑', '데이트',
    '관계', '두근거림', '로맨틱', '키스', '결혼', '설렘', '운명', '행복', '이별',
    '기다림', '포옹', '미스터리', '로맨틱코미디', '우정', '미스테리', '우정',
    '진심', '관심', '소원', '희망', '추억'
]
humanism_keywords=['자전적', '인물', '회고록', '이야기', '성장', '삶', '경험','명상', '치유', '힐링', '마음', '몸', '스트레스', '정신건강',
    '의지', '리프레시', '평온', '이완', '일기', '편지', '기록', '다이어리', '감정', '고백', '일상',
    '추억', '소통', '고민', '행복', '슬픔', '생각', '기쁨', '소망','여행', '경험', '발견', '모험', '문화', '풍경', '해외',
    '국내', '방황', '발걸음', '즐거움', '새로운', '도전', '열정', '탐험']
sf_keywords=[
    '과학', '판타지', '우주', '로봇', '미래', '우주선', '로봇',
    '마법', '마법사', '요정', '왕국', '비밀', '모험', '환상', '물체'
]
horror_mystery_keywords = [
    '공포', '호러', '미스터리', '스릴러', '긴장감', '비밀', '사건',
    '귀신', '불안', '살인', '수수께끼', '악마', '추리', '마비',
    '무서움', '살아있는', '죽음', '긴박', '무서운', '사이코패스',
    '공포감', '전율', '고어', '혼란', '고통', '이방인', '암시',
    '연금술', '암흑', '악마의', '고통스러운', '괴물', '역사', '혐오'
]
science_math_keywords = [
    '물리학', '천문학', '지구과학', '수학', '물리', '천체', '우주',
    '지구', '운동', '역학', '물리적', '광학', '전자기학', '기체',
    '열역학', '원자', '핵', '반응', '우주선', '우주비행', '해석학',
    '미적분', '수학적', '기하학', '확률', '통계', '미분', '초월함수',
    '근거리', '원격', '전파', '우주관측', '산업', '공학', '산업혁명',
    '기술', '발전', '화학', '생물학', '생태학', '지구과학적', '지질학',
    '단위', '공간', '차원', '시간', '속력', '가속도', '중력', '질량',
    '에너지', '전력', '전압', '저항', '전류', '전자', '플라즈마', '자기',
    '자성', '유전체', '원자력', '핵분열', '핵융합', '우주이론', '상대성이론',
    '양자역학', '중력파', '머신러닝', '인공지능', '로봇공학'
]
biology_chemistry_keywords = [
    '생물학', '화학', '생명', '세포', '유전', '진화', '유기', '무기',
    '분자', '원자', '원소', '화학물질', '화합물', '반응', '화학반응',
    '분자생물학', '세포생물학', '유전학', '생리학', '해부학', '생화학',
    '분자생물학', '생태학', '미생물학', '생명체', '유전체', '유전자',
    '단백질', '유전적', '환경', '생물체', '생태계', '생물적', '화학자',
    '생물학자', '분자구조', '분자생물학자', '화학물질', '유기화학', '무기화학',
    '생화학자', '화학반응', '생물학적', '화학적', '유전적', '발암물질',
    '독성', '약물', '합성', '분해', '분석', '실험', '실험실', '연구',
    '학문', '과학', '이론', '실험체', '실험장비', '모델', '반응식'
]
success_skills_keywords = [
    '성공', '처세', '자기능력계발', '자기계발', '성공법', '성공신화', '성공스토리',
    '성공사례', '성공전략', '목표', '계획', '자기관리', '시간관리', '자기개발',
    '자아실현', '인생', '성장', '도전', '관리', '목표설정', '의지력', '실천',
    '노력', '자기파괴', '변화', '목적', '자신감', '자기통제', '역량', '자기주도',
    '협상', '리더십', '인간관계', '소통', '인간관계능력', '소통능력', '성공적',
    '목표달성', '성취', '개인성장', '실패', '도전', '몰입', '결정', '자기관리',
    '행동', '성취감', '자기도전', '관심', '자아성찰', '자아개발', '실천', '자아실현',
    '지혜', '자기계발서', '성공작', '노하우', '자기계발서', '성공전략', '감정관리',
    '관심', '독서', '지식', '자기계발도서', '실천', '시도', '결정', '자기결정', '목적지',
    '인내', '자제', '책임감', '포기', '자기반성', '성장', '도전', '자신감', '자기반성',
    '성공작', '성공비결', '자기계발법', '자기계발서', '성공전략', '목표설정', '자기주도',
    '자기몰입', '목표', '취업', '자기평가', '강의', '스피치', '멘토링', '멘토', '코칭',
    '자기발전', '자기관리', '자기계발서', '스터디', '교육', '훈련', '실무', '팀워크',
    '효율', '성공코드', '성공행동', '성공습관', '성공가이드'
]
economics_management_keywords = [
    '경제', '경영', '기업', '금융', '회계', '경제학', '시장', '마케팅',
    '전략', '경영전략', '경제정책', '경영학', '재무', '투자', '경영자',
    '경영실무', '경영전략', '산업', '경제분석', '경제이론', '금융시장',
    '자본', '이익', '비즈니스', '소비자', '공급', '수요', '유통',
    '인플레이션', '디플레이션', '경기', '수익', '시장분석', '경쟁',
    '세무', '부채', '자산', '수익률', '이자', '금리', '환율', '주가',
    '주식', '부동산', '투자자', '투자금', '창업', '창조경제', '경제성장',
    '경제발전', '경제지표', '수요예측', '공급망', '매출', '경영전략',
    '경영혁신', '경영관리', '경영자', '경영진', '경영컨설팅', '경영실무',
    '경영코칭', '경영교육', '경제학자', '경영학자', '효율적', '효과적',
    '생산성'
]
psychology_philosophy_humanities_keywords = [
    '심리학', '철학', '인문학', '인문교양', '정신', '의식', '의지', '자아',
    '의미', '가치', '사상', '윤리', '미학', '사유', '인식', '감정', '성격',
    '정서', '무의식', '의식', '사고', '인간', '존재', '자유', '존엄성', '진리',
    '행복', '미래', '역사', '역사학', '시대', '사회', '문화', '인간관계', '사회학',
    '심리학자', '철학자', '인문학자', '인문교양서', '철학서', '심리학서', '사회학서',
    '문화학', '역사학자', '역사서', '역사비평', '사상사', '인간사', '신화', '전통',
    '교육', '문학', '문학이론', '비평', '문학사', '문화비평', '문화사', '문화이론',
    '사회이론', '역사이론', '인식론', '인식과학', '심리학이론', '심리학개론', '철학이론',
    '철학사상', '철학과학', '철학적사유', '철학비평', '인문학적사유', '인문학과학',
    '인문학비평', '인문학이론', '심리학적사유', '심리학과학', '심리학비평', '심리학이론'
]
history_keywords = [
    '역사', '역사학', '역사비평', '역사서', '역사이론', '역사자료', '역사사회',
    '역사문화', '역사사상', '역사관', '역사인물', '역사사건', '역사시대',
    '역사국가', '역사문명', '역사지리', '역사과학', '역사이야기', '역사고전',
    '역사서술', '역사기록', '역사전쟁', '역사사회', '역사문학', '역사지식'
]
korean_poetry_keywords = [
    '한국시', '한시', '시조', '한국시인', '한국시집', '한국시선', '한국시문학',
    '한국시풍경', '한국시풍속', '한국시장', '한국시술', '한국시란', '한국시교',
    '한국시문', '한국시문화'
]
negotiation_keywords = [
    '화술', '협상', '협상전략', '협상실무', '협상이론', '협상투자', '협상가',
    '협상자', '협상기술', '협상자세', '협상실력', '협상능력', '협상전략',
    '협상능력향상', '화술교육', '협상실습', '협상스킬', '협상학습', '협상훈련',
    '협상자격', '협상자격증', '협상가이드', '협상자료', '협상술', '협상가치'
]
time_management_keywords = [
    '시간관리', '시간분배', '일정관리', '일정표', '스케줄', '일정조정',
    '시간계획', '일정계획', '시간절약', '효율적시간활용', '시간투자',
    '시간낭비', '시간절약법', '시간적극활용', '시간효율', '시간운용',
    '시간투자법', '시간운용법', '시간활용법', '시간배분', '시간절감'
]
interpersonal_relationships_keywords = [
    '인간관계', '대인관계', '사회관계', '대인관계능력', '사회관계능력',
    '인간관계기술', '대인관계기술', '사회관계기술', '인간관계능력향상',
    '대인관계능력향상', '사회관계능력향상', '인간관계능력향상법',
    '대인관계능력향상법', '사회관계능력향상법', '인간관계능력향상방법'
]


all=[]
for i in df_genre_keyword:
  for j in df_genre_keyword[i]:
    all.append(j)

romance_list=" ".join(all[1]+all[17]+love_keywords)
humanism_list=" ".join(all[18]+all[20]+all[19]+all[16]+humanism_keywords)
sf_list=" ".join(all[3]+all[4]+sf_keywords)
horror_list=" ".join(all[0]+all[2]+horror_mystery_keywords)
math_physics_list=" ".join(all[6]+all[8]+all[9]+all[10]+science_math_keywords)
chemistry_list=" ".join(all[11]+all[7]+biology_chemistry_keywords)
success_list=" ".join(all[22]+all[24]+success_skills_keywords)
economy_list=" ".join(all[26]+all[17]+economics_management_keywords)
psychology_list=" ".join(all[12]+all[13]+all[14]+psychology_philosophy_humanities_keywords)
history_list=" ".join(all[5]+history_keywords)
korean_poem_list=" ".join(all[15]+korean_poetry_keywords)
nego_list=" ".join(all[21]+negotiation_keywords)
time_list=" ".join(all[23]+time_management_keywords)
relation_list=" ".join(all[25]+interpersonal_relationships_keywords)


romance_sim=pd.concat([df_horror,df_mystery,df_sf,df_fantasy,df_diary,df_trip,df_human,df_medit])
human_sim=pd.concat([df_romance,df_love,df_success,df_abil,df_phsyco,df_iron,df_inmun])
sf_sim=pd.concat([df_horror,df_mystery,df_romance,df_love,df_history])
horror_sim=pd.concat([df_romance,df_love,df_diary,df_trip,df_human,df_medit,df_history])
physics_sim=pd.concat([df_biology,df_chemi,df_phsyco,df_iron,df_inmun,df_economy,df_manage])
bio_sim=pd.concat([df_physics,df_math,df_star,df_earth,df_speak,df_phsyco,df_iron,df_inmun])
success_sim=pd.concat([df_phsyco,df_iron,df_inmun,df_relation,df_time])
economy_sim=pd.concat([df_phsyco,df_iron,df_inmun,df_success,df_abil,df_speak])
phsyco_sim=pd.concat([df_relation,df_speak,df_economy,df_manage])
history_sim=pd.concat([df_horror,df_mystery,df_human,df_diary,df_medit,df_trip,df_romance,df_love])
poem_sim=pd.concat([df_romance,df_love,df_human,df_diary,df_medit,df_trip,df_success,df_abil])
speak_sim=pd.concat([df_relation,df_time,df_success,df_abil])
time_sim=pd.concat([df_speak,df_economy,df_manage,df_success,df_abil])
relation_sim=pd.concat([df_speak,df_success,df_abil,df_phsyco,df_iron,df_inmun])

filter_romance_df = pd.concat([df_romance, df_love])
filter_humanism_df = pd.concat([df_human, df_medit, df_diary, df_trip])
filter_sf_df = pd.concat([df_sf, df_fantasy])
filter_horror_df = pd.concat([df_horror, df_mystery])
filter_math_physics_df = pd.concat([df_physics, df_star, df_earth, df_math])
filter_chemistry_df = pd.concat([df_biology, df_chemi])
filter_success_df = pd.concat([df_success, df_abil])
filter_economy_df = pd.concat([df_economy, df_manage])
filter_psychology_df = pd.concat([df_phsyco, df_iron, df_inmun])
filter_history_df = df_history
filter_korean_poem_df = df_korPoem
filter_nego_df = df_speak
filter_time_df = df_time
filter_relation_df = df_relation


romance_sim_slice=romance_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
human_sim_slice=human_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
sf_sim_slice=sf_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
horror_sim_slice=horror_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
physics_sim_slice=physics_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
bio_sim_slice=bio_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
success_sim_slice=success_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
economy_sim_slice=economy_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
phsyco_sim_slice=phsyco_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
history_sim_slice=history_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
poem_sim_slice=poem_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
speak_sim_slice=speak_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
time_sim_slice=time_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
relation_sim_slice=relation_sim[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]

filter_romance_sim_slice = filter_romance_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_human_sim_slice = filter_humanism_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_sf_sim_slice = filter_sf_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_horror_sim_slice = filter_horror_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_physics_sim_slice = filter_math_physics_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_bio_sim_slice = filter_chemistry_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_success_sim_slice = filter_success_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_economy_sim_slice = filter_economy_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_phsyco_sim_slice = filter_psychology_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_history_sim_slice = filter_history_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_poem_sim_slice = filter_korean_poem_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_speak_sim_slice = filter_nego_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_time_sim_slice = filter_time_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]
filter_relation_sim_slice = filter_relation_df[['ISBN_THIRTEEN_NO','GENRE_LV2','KEYWORD_oneline']]


from sentence_transformers import SentenceTransformer, models ,util
import pickle

pickle_file_path = "embedder3.pickle"

with open(pickle_file_path, 'rb') as file:
    embedder = pickle.load(file)

slice_df_cluster = {
    "romance": romance_sim_slice,
    "human": human_sim_slice,
    "sf": sf_sim_slice,
    "horror": horror_sim_slice,
    "physics": physics_sim_slice,
    "bio": bio_sim_slice,
    "success": success_sim_slice,
    "economy": economy_sim_slice,
    "psycho": phsyco_sim_slice,
    "history": history_sim_slice,
    "poem": poem_sim_slice,
    "speak": speak_sim_slice,
    "time": time_sim_slice,
    "relation": relation_sim_slice
}

filter_slice_df_cluster = {
    "romance": filter_romance_sim_slice,
    "human": filter_human_sim_slice,
    "sf": filter_sf_sim_slice,
    "horror": filter_horror_sim_slice,
    "physics": filter_physics_sim_slice,
    "bio": filter_bio_sim_slice,
    "success": filter_success_sim_slice,
    "economy": filter_economy_sim_slice,
    "psycho": filter_phsyco_sim_slice,
    "history": filter_history_sim_slice,
    "poem": filter_poem_sim_slice,
    "speak": filter_speak_sim_slice,
    "time": filter_time_sim_slice,
    "relation": filter_relation_sim_slice
}

all_keywords_list = {
    "romance": romance_list,
    "human": humanism_list,
    "sf": sf_list,
    "horror": horror_list,
    "physics": math_physics_list,
    "bio": chemistry_list,
    "success": success_list,
    "economy": economy_list,
    "psycho": psychology_list,
    "history": history_list,
    "poem": korean_poem_list,
    "speak": nego_list,
    "time": time_list,
    "relation": relation_list
}

def book_recommend(genre_list):
    recommend_df=pd.DataFrame()
    for genre in genre_list:
        if genre in slice_df_cluster:
            recommend_df=pd.concat([recommend_df,slice_df_cluster[genre]])
        else:
            print(f"Warning: Genre '{genre}' not recognized.")
    recommend_isbn=recommend_df['ISBN_THIRTEEN_NO'].to_list()
    recommend_isbn_slice=random.sample(recommend_isbn,15)


    return recommend_isbn_slice


def load_model(model_name='jhgan/ko-sroberta-multitask'):
    """
    SentenceTransformer 모델을 로드합니다.
    """
    return SentenceTransformer(model_name)


def get_slices_and_keywords_by_genres(genres):
    """
    입력된 장르 리스트를 기반으로 해당하는 슬라이스와 키워드 리스트를 반환합니다.
    """
    print("get_slices_and_keywords_by_genres")
    selected_slices = []
    selected_keywords = []
    for genre in genres:
        if genre in slice_df_cluster:
            selected_slices.append(slice_df_cluster[genre])
            selected_keywords.append(all_keywords_list[genre])
        else:
            print(f"Warning: Genre '{genre}' not recognized.")
    return selected_slices, selected_keywords


def get_filter_slices_and_keywords_by_genres(genres):
    """
    입력된 장르 리스트를 기반으로 해당하는 슬라이스와 키워드 리스트를 반환합니다.
    """
    selected_slices = []
    selected_keywords = []
    for genre in genres:
        if genre in filter_slice_df_cluster:
            selected_slices.append(filter_slice_df_cluster[genre])
            selected_keywords.append(all_keywords_list[genre])
        else:
            print(f"Warning: Genre '{genre}' not recognized.")
    return selected_slices, selected_keywords


def find_similar_books(embedder, selected_slices, selected_keywords, top_k=25):
    """
    각 슬라이스에 대해 유사한 책을 찾고 ISBN 리스트를 반환합니다.
    """
    sim_book = []
    for i in range(len(selected_slices)):
        tmp = selected_slices[i].reset_index()
        corpus = tmp

        # 코퍼스 임베딩 생성
        corpus_embeddings = embedder.encode(corpus["KEYWORD_oneline"], convert_to_tensor=True)

        # 쿼리 문장 (키워드 리스트)
        query = selected_keywords[i]

        # 쿼리 임베딩 생성
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        li_top = []

        # 상위 top_k 결과를 찾기 위해 np.argpartition 사용
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        for idx in top_results[0:top_k]:
            li_top.append(corpus.loc[idx.item()]['ISBN_THIRTEEN_NO'])

        sim_book.append(li_top)
    return sim_book


def main(genre_list):
    # 사용 예제
    genre_list=genre_list
    input_genres = genre_list  # 사용자로부터 입력받은 장르 리스트

    selected_slices, selected_keywords = get_slices_and_keywords_by_genres(input_genres)

    #non-filter book
    sim_book = find_similar_books(embedder, selected_slices, selected_keywords)

    #just_genre_book
    recommend_list_isbn=book_recommend(genre_list)




if __name__ == "__main__":
    main()
