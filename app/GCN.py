'''
  학습 설정
'''



from box import Box
import warnings
warnings.filterwarnings(action = 'ignore')

import pandas as pd
import numpy as np

import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sparse
import pickle





def get_metrics(W_u, W_i, n_users, n_items, train_data, test_data, K, testing=0):
    test_user_ids = torch.LongTensor(test_data['user_id_idx'].unique())
    relevance_score = torch.matmul(W_u, torch.transpose(W_i, 0, 1))

    # 사용자와 아이템 행렬 초기화
    R = sparse.dok_matrix((n_users, n_items), dtype=np.float32)
    R[train_data['user_id_idx'], train_data['item_id_idx']] = 1.0

    # 희소 텐서(행렬)
    R_tensor = convert_to_sparse_tensor(R)  # 희소 행렬 반환. R텐서 안에 sparse 텐서를 담음
    R_tensor_dense = R_tensor.to_dense()  # 모델 안에 넣어서 처리하기 위해 dense 처리
    R_tensor_dense = R_tensor_dense * (-np.inf)  # -무한대로 초기화
    R_tensor_dense = torch.nan_to_num(R_tensor_dense, nan=0.0)

    relevance_score = relevance_score + R_tensor_dense

    # 추천할 때 top-k 인덱스
    topk_idx = torch.topk(relevance_score, K).indices

    topk_idx_df = pd.DataFrame(topk_idx.numpy(), columns=['top_idx' + str(x + 1) for x in range(K)])
    topk_idx_df['user_id'] = topk_idx_df.index
    topk_idx_df['topk_item'] = topk_idx_df[['top_idx' + str(x + 1) for x in range(K)]].values.tolist()
    topk_idx_df = topk_idx_df[['user_id', 'topk_item']]

    # 테스트 데이터에서 유저가 실제로 상호작용했던 아이템 추출 후 데이터프레임 형태로
    test_items = test_data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()

    # 유저 별로 실제로 구매한 아이템들의 리스트와 추천 결과로 나왔던 top-k개의 아이템 리스트를 하나의 데이터프레임 안에 저장
    metrics_df = pd.merge(test_items, topk_idx_df, how='left', left_on='user_id_idx', right_on='user_id')

    # 유저가 실제로 구매한 아이템들과 추천 결과로 나온 아이템들의 교집합을 'interact_items'에 추가
    metrics_df['interact_items'] = [list(set(a).intersection(b)) for a, b in
                                    zip(metrics_df.item_id_idx, metrics_df.topk_item)]

    # recall 계산 : 상호작용한 아이템 중 추천된 상품 비율
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['interact_items']) / len(x['item_id_idx']), axis=1)

    if (testing == 1):
        return topk_idx_df

    # hit_list : 추천 모델이 맞춘 아이템 리스트
    def get_dcg_idcg(item_id_idx, hit_list):
        dcg = sum([hit * np.reciprocal(np.log1p(idx + 1)) for idx, hit in enumerate(hit_list)])
        idcg = sum([np.reciprocal(np.log1p(idx + 1)) for idx in range(min(len(item_id_idx), len(hit_list)))])
        return dcg / idcg

    # 맞추면 1, 못 맞췄으면 0
    metrics_df['hit_list'] = metrics_df.apply(
        lambda x: [1 if i in set(x['item_id_idx']) else 0 for i in x['topk_item']], axis=1)
    metrics_df['ndcg'] = metrics_df.apply(lambda x: get_dcg_idcg(x['item_id_idx'], x['hit_list']), axis=1)

    return metrics_df['recall'].mean(), metrics_df['ndcg'].mean()

'''
  positive_scores가 negative_scores보다 크도록 하는 관계를 학습시키는 함수
'''
def bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0,reg):
    pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim = 1) # user_embedding과 positive_embedding의 내적을 더한 결과
    neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim = 1) # user_embedding과 negative_embedding의 내적을 더한 결과
    reg=reg
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    # 정규화 항이 존재한다면, L2-norm을 최종 손실에 추가
    if reg > 0 :
        l2_norm = (user_emb_0.norm().pow(2) + pos_emb_0.norm().pow(2) + neg_emb_0.norm().pow(2)) / float(len(users))
        reg_loss = reg * l2_norm
        loss += reg_loss

    return loss


# 희소 행렬 반환
def convert_to_sparse_tensor(dok_mtrx):
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data  # 희소 행렬 값 추출

    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))  # 희소 행렬의 행과 열을 추출해서 배열로 만들기

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return dok_mtrx_sparse_tensor

def prepare_dataset():
    uData_path = "data_2.csv"

    df = pd.read_csv(uData_path)
    data = df[['item_id', 'user_id', 'rating']]
    train, test = train_test_split(data.values, test_size=0.2, random_state=42)
    train = pd.DataFrame(train, columns=data.columns)
    test = pd.DataFrame(test, columns=data.columns)

    user_label_encoder = LabelEncoder()  # 유저를 위한 인코더
    item_label_encoder = LabelEncoder()  # 아이템을 위한 인코더

    train['user_id_idx'] = user_label_encoder.fit_transform(train['user_id'].values)  # 인코더를 사용해서 유저 아이디를 정수형 인덱스로 변환
    train['item_id_idx'] = item_label_encoder.fit_transform(train['item_id'].values)  # 인코더를 사용해서 아이템 아이디를 정수형 인덱스로 변환

    train_user_ids = train['user_id'].unique()  # 고유한 유저 아이디 추출
    train_item_ids = train['item_id'].unique()  # 고유한 아이템 아이디 추출

    # (테스트 데이터의 조건) 테스트 데이터의 유저 아이디와 아이템 아이디는 학습 데이터 내에 존재해야 한다
    test = test[(test['user_id'].isin(train_user_ids) & (test['item_id'].isin(train_item_ids)))]

    # 인코더는 학습 데이터에 사용된 인코더를 사용!
    test['user_id_idx'] = user_label_encoder.transform(test['user_id'].values)
    test['item_id_idx'] = item_label_encoder.transform(test['item_id'].values)

    n_users = train['user_id_idx'].nunique()
    n_items = train['item_id_idx'].nunique()

    return train,test,user_label_encoder,item_label_encoder,n_users,n_items





def new_adj_mat(user_label_encoder,item_label_encoder,n_users,n_items,user_id, book_list, adj_mat,train):
    '''
      신규 유저 등록 후 인접행렬 수정 및 인코딩

      타 함수로부터 리턴받는 신규 유저 값은 new_user_id 변수로 받고, 선호 장르 도서 리스트는 new_user_item으로 받아야 함
      new_user_df : 인코딩 전 user_id, item_id와 인코딩 후 user_id_idx, item_id_idx로 이루어진 데이터프레임
    '''

    # 신규 유저 추천의 경우 (원래는 다른 함수로부터 user_id와 book_list를 받아 옴
    train=train
    new_user_id=user_id
    new_user_item=book_list


    # new_user_id = 'subin'
    # new_user_item = ['9788982737145', '9788925520339', '9788991449848',  # 미스터리/스릴러
    #                  '9788961886758', '9788991396920', '9788925505442',  # 로맨스
    #                  '9788930085489', '9788955965032', '9788996146308']  # 자기능력계발

    classes = item_label_encoder.classes_
    # item_label_encoder.classes_에 존재하는 값만 남기기

    filtered_list = [item for item in new_user_item if item in classes]
    filtered_list = list(set(filtered_list))

    # user_label_encoder는 상단에서 사용했던(기존 인접행렬 데이터 인코딩에 사용했던) 것과 동일한 것이어야 함. 꼭!!
    user_label_encoder.classes_ = np.append(user_label_encoder.classes_, [new_user_id])

    new_user_df = pd.DataFrame()

    # 꼭 item_id 열부터 정의하고 user_id 열을 정의해야 함!!
    new_user_df['item_id'] = filtered_list
    new_user_df['user_id'] = new_user_id
    new_user_df = new_user_df[['user_id', 'item_id']]
    # new_user_df['item_id'] = pd.to_numeric(new_user_df['item_id'])
    new_user_df['user_id_idx'] = user_label_encoder.transform(
        new_user_df['user_id'].values)  # 인코더를 사용해서 유저 아이디를 정수형 인덱스로 변환

    # 리스트 안의 값이 데이터프레임의 'item_id' 열에 있는지 확인하고 해당 행의 'value' 열 출력
    new_item_idx_list = []
    for item in new_user_df['item_id'].values:
        # 데이터프레임에서 'item_id'가 현재 리스트 항목과 일치하는 행을 필터링
        matching_rows = train[train['item_id'] == item]
        if not matching_rows.empty:
            new_item_idx_list.append(matching_rows.iloc[0]['item_id_idx'])
            continue

    new_user_df['item_id_idx'] = new_item_idx_list

    n_new_user = new_user_df['user_id_idx'].nunique()  # 신규유저 명수인데, 그냥 1임.

    # n_items : 12612개. 상단 코드(기존 유저-아이템 행렬 R 만들 때)에 선언되어 있음
    new_user_R = sparse.dok_matrix((n_new_user, n_items), dtype=np.float32)
    new_user_R[0, new_user_df['item_id_idx']] = 1.0  # 인접행렬에 추가할 것이기 때문에, 희소 리스트 생성

    # 피클로 저장해두었던 adj_mat 인접행렬을 array로 변경!
    adj_mat_arr = adj_mat.toarray()

    new_user_R_arr = [0.0 for _ in range(n_users)]  # 인접행렬에 삽입할 행. 앞부분은 0으로 채워야 함.
    new_user_R_row = new_user_R_arr + new_user_R.toarray()[0].tolist()

    new_user_R_col = list(new_user_R_row)  # 인접행렬에 삽입할 열.
    new_user_R_col.insert(0, 0.0)  # 인접행렬에 행을 추가한 뒤에 열을 추가할 것이기 때문에 0을 추가함

    # 인코딩된 신규 유저 아이디. 인덱스값으로 쓸 거임
    new_user_id_idx = new_user_df['user_id_idx'][0]

    # 인접행렬에 위에서 만든 행(new_user_R_row)과 열(new_user_R_col) 삽입. 대략 3.4초 소요됨
    adj_mat_arr = np.insert(adj_mat_arr, new_user_id_idx, new_user_R_row, axis=0)
    adj_mat_arr = np.insert(adj_mat_arr, new_user_id_idx, new_user_R_col, axis=1)

    # 기존 모델 코드에 사용될 수 있도록 dok 형식으로 변환. 7~8초 소요됨. 이 결과가 리턴값으로 나와야 함
    adj_mat_after = sparse.dok_matrix(adj_mat_arr)

    return adj_mat_after






# 학습을 시킬 때, 특정 배치 사이즈로 계속해서 이 모델 안에 데이터를 흘려보내주는 함수
def data_loader(data, batch_size, n_users, n_items) :
    # 유저별로 상호작용한 아이템의 목록 생성
    # 유저 기준 그룹핑 -> 각 유저가 상호작용한 아이템을 리스트 형태로 만듦
    interacted_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()

    # 유저와 상호작용 하지 않은 데이터(negative sample)를 뽑는 함수
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_items - 1) # 아이템 중 하나 랜덤으로 추출
            if neg_id not in x : # neg_id가 x 안에 없는 경우, 즉 상호작용 하지 않은 아이템인 경우
                return neg_id

    indices = [x for x in range(n_users)] # 모든 유저에 대한 인덱스

    # 배치 사이즈가 전체 유저의 수보다 큰 경우 -> 중복 샘플링 / 그렇지 않다면 중복 없이 샘플링
    if n_users < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()

    # 선택된 유저에 대한 데이터프레임 생성
    users_df = pd.DataFrame(users, columns = ['users'])
    # 선택된 유저와 상호작용한 아이템의 목록을 병합
    interacted_items_df = pd.merge(interacted_items_df, users_df, how = 'right', left_on = 'user_id_idx', right_on = 'users')

    # 상호작용한 아이템(pos_items)와 상호작용하지 않은 아이템(neg_items) 샘플링
    pos_items = interacted_items_df['item_id_idx'].apply(lambda x : random.choice(x)).values
    neg_items = interacted_items_df['item_id_idx'].apply(lambda x : sample_neg(x)).values

    return list(users), list(pos_items), list(neg_items)

def train_model(lightGCN,train, test, n_users, n_items,epochs, batch_size, lr,top_k,reg):
    '''
      학습 함수를 실행하기 위한 트레이너
      config batch_size: 2^14
      top_k: 100
    '''
    lightGCN=lightGCN
    epochs=epochs
    batch_size=batch_size
    lr=lr
    top_k=top_k
    reg=reg
    topk_epoch_10 = pd.DataFrame()
    topk_epoch_20 = pd.DataFrame()

    # loss를 담기 위한 리스트
    loss_list_epoch = []

    # optimizer 정의 : Adam 활용
    optimizer = torch.optim.Adam(lightGCN.parameters(), lr=lr)

    # 에폭을 돌면서 학습 진행
    # tqdm 사용 : 학습의 진행 상황을 게이지로 추적 관찰 가능
    for epoch in tqdm(range(epochs)):
        n_batch = int(len(train) / batch_size)
        final_loss_list = []  # 각 epoch마다 loss를 계산해 줄 수 있는 리스트

        lightGCN.train()  # lightGCN 모델을 학습 모드로 변경

        for batch_idx in range(n_batch):
            optimizer.zero_grad()  # 옵티마이저는 항상 초기화시켜줘야 함(그레디언트 초기화)
            users, pos_items, neg_items = data_loader(train, batch_size, n_users, n_items)  # 배치별로 데이터 로딩
            users_emb, pos_emb, neg_emb, users_emb_0, pos_emb_0, neg_emb_0 = lightGCN.forward(users, pos_items,
                                                                                              neg_items)  # 모델을 forward 시킴

            # BPR loss를 계산해서 final_loss에 저장
            final_loss = bpr_loss(users, users_emb, pos_emb, neg_emb, users_emb_0, pos_emb_0, neg_emb_0,reg)
            final_loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
            final_loss_list.append(final_loss.item())

        lightGCN.eval()  # lightGCN 모델을 평가 모드로 변경, 즉 이제 가중치가 업데이트되지 않음
        with torch.no_grad():
            final_user_emb, final_item_emb, initial_user_emb, initial_item_emb = lightGCN.propagate_through_layers()
            topk_df = get_metrics(final_user_emb, final_item_emb, n_users+1, n_items, train, test, top_k, 1)

        # epoch별로 진행을 했을 때, 각각의 매트릭들을 저장
        loss_list_epoch.append(round(np.mean(final_loss_list), 4))

        print(f"[EPOCH : {epoch}, Train Loss : {round(np.mean(final_loss_list), 4)}]")

        if (epoch == 0):
            topk_epoch_10 = topk_df
        if (epoch == 1):
            topk_epoch_20 = topk_df

    return topk_epoch_10, topk_epoch_20



def decoding(user_label_encoder,item_label_encoder,user_id, topk_item) :
    topk_predict = {}

    topk_predict['user_id'] = user_label_encoder.inverse_transform([user_id])
    topk_predict['item_id'] = [item_label_encoder.inverse_transform([item]) for item in topk_item]

    return topk_predict



def run_GCN(user_id,book_list):
    config = {
        'latent_dim': 64,
        'lr': 0.001,
        'batch_size': 2 ** 14,
        'top_k': 100,
        'n_layers': 3,
        'reg': 1e-4,
        'epochs': 2
    }
    config = Box(config)


    user_id=user_id
    book_list=book_list

    train,test,user_label_encoder,item_label_encoder,n_users,n_items=prepare_dataset()
    #adj_mat = pickle.load(open("adj_mat.pkl", 'rb'))
    adj_mat = pd.read_pickle("adj_mat.pkl")
    adj_mat_after=new_adj_mat(user_label_encoder, item_label_encoder, n_users, n_items, user_id, book_list, adj_mat,train)
    adj_mat=adj_mat_after

    '''
      모델링
    '''

    class LightGCN(nn.Module):

        def __init__(self, data, n_users, n_items, n_layers, latent_dim):
            super(LightGCN, self).__init__()
            self.data = data
            self.n_users = n_users
            self.n_items = n_items
            self.n_layers = n_layers
            self.latent_dim = latent_dim
            self.init_embedding()
            self.norm_adj_mat_sparse_tensor = self.get_norm_adj()  # 정규화된 인접행렬 생성

        def init_embedding(self):
            self.E0 = nn.Embedding(self.n_users + self.n_items, self.latent_dim)
            nn.init.xavier_uniform_(self.E0.weight)
            self.E0.weight = nn.Parameter(self.E0.weight)

        def get_norm_adj(self):
            print("=============== get_norm_adj() 접근 ===============")

            # 각 행마다 0이 아닌 값을 더함
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum + 1e-9, -0.5).flatten()  # 1e-9를 더해주는 이유는, 0이 되는 것을 방지하기 위함
            d_mat_inv = sparse.diags(d_inv)  # 대각행렬

            # D^(-1/2) * A * D^(-1/2) 논문에서의 수식임
            norm_adj_mat = d_mat_inv.dot(adj_mat).dot(d_mat_inv)  # 정규화한 매트릭스

            # 희소행렬로 변경
            norm_adj_mat_coo = norm_adj_mat.tocoo().astype(np.float32)
            values = norm_adj_mat_coo.data  # 희소행렬 값

            indices = np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col))  # 희소행렬의 인덱스
            i = torch.LongTensor(indices)  # 인덱스를 파이토치의 롱플로로 변경
            v = torch.FloatTensor(values)

            return torch.sparse.FloatTensor(i, v, torch.Size(norm_adj_mat_coo.shape))

        # 임베딩들을 전파하면서 갱신
        def propagate_through_layers(self):
            all_layers_embedding = [self.E0.weight]
            E_lyr = self.E0.weight  # 현재 레이어를 초기 레이어로

            for layer in range(self.n_layers):
                E_lyr = torch.sparse.mm(self.norm_adj_mat_sparse_tensor, E_lyr)
                all_layers_embedding.append(E_lyr)  # 갱신된 값 추가

            # 각 레이어의 임베딩을 평균내, 최종 임베딩 계산
            all_layers_embedding = torch.stack(all_layers_embedding)
            mean_layer_embedding = torch.mean(all_layers_embedding, axis=0)

            # 유저와 아이템의 임베딩을 분리해서 리턴
            return torch.split(mean_layer_embedding, [self.n_users, self.n_items]) + torch.split(self.E0.weight,
                                                                                                 [self.n_users,
                                                                                                  self.n_items])

        def forward(self, users, pos_items, neg_items):
            final_user_emb, final_item_emb, initial_user_emb, initial_item_emb = self.propagate_through_layers()

            return (
            final_user_emb[users], final_item_emb[pos_items], final_item_emb[neg_items], initial_user_emb[users],
            initial_item_emb[pos_items], initial_item_emb[neg_items])

    start = time.time()  # 시작 시간 저장

    # 모델 초기화
    lightGCN = LightGCN(train, n_users+1, n_items, config.n_layers, config.latent_dim)

    print("결과시간 :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    # epochs=config.epochs
    # batch_size=config.batch_size
    # lr=config.lr
    # top_k=config.top_k

    topk_epoch_10, topk_epoch_20=train_model(lightGCN,train, test, n_users, n_items, config.epochs, config.batch_size, config.lr, config.top_k,config.reg)

    topk_epoch_10_user = decoding(user_label_encoder, item_label_encoder,topk_epoch_10.iloc[25784]['user_id'], topk_epoch_20.iloc[25784]['topk_item'])
    nonfilter_list = []
    for i in topk_epoch_10_user['item_id']:
        nonfilter_list.append(i[0])
    print(nonfilter_list)

    topk_epoch_20_user = decoding(user_label_encoder, item_label_encoder,topk_epoch_20.iloc[25784]['user_id'], topk_epoch_20.iloc[25784]['topk_item'])
    filter_list = []
    for i in topk_epoch_20_user['item_id']:
        filter_list.append(i[0])
    print(filter_list)


    return nonfilter_list, filter_list


# user = 'subin'
# books = [9788963710006, 9788925516523, 9788952212306, 9788931005547, 9788952748096, 9788952745071,
#                      9788952214690, 9788974255565, 9788979199208, 9788987523163, 9788987523163, 9788987523163,
#                      9788982181030, 9788960900936, 9788960900936, 9788962911664, 9788938204127, 9788977875371,
#                      9788936456276, 9788936472115, 9788958072461, 9788939551299, 9788925532219, 9788977630758]
#
#
# run_GCN(user,books)