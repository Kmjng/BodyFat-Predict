# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:03:43 2024

@author: itwill
"""


file_2 = r"C:/ITWILL/MiddleProject/시도/증강_1차/Dataset.csv"
import pandas as pd 



df_f = pd.read_csv(file_2, encoding= 'euc-kr')
df_f.shape
#  (177, 23)
df_f.columns
df_f = df_f[['Age','BodyFat','Height','Weight','Chest','Abdomen','Hip','Thigh','Biceps','Ankle','Knee','Neck','Wrist']]
#  (177, 13)

# 불균형한 다중분류로 되어있어서 우선 삭제 

# 이진 분류 
def class_2_labeling(df):
    class_lst = [0,1] 
    # class 0 ; 비만 아님
    # class 1 ;  비만임 

    df['Class'] = None
    for i, values in df.iterrows():
        
         # 여성 30세 이상            
        if df['Age'][i] >= 30 :
             if df['BodyFat'][i] < 27 : 
                 df['Class'][i] = class_lst[0]
             elif  df['BodyFat'][i] >= 27 :
                 df['Class'][i] = class_lst[1]   
        # 여성 30세 미만          
        elif df['Age'][i] < 30 : 
             if df['BodyFat'][i] < 24 : 
                 df['Class'][i] = class_lst[0]
             elif  df['BodyFat'][i] >= 24 :
                 df['Class'][i] = class_lst[1]
    
    
    return df

df_f.BodyFat.min()

df_f = class_2_labeling(df_f)


df_f['Class'] = df_f['Class'].astype(int)

from collections import Counter


cnt2 = Counter(df_f['Class'])
cnt2 # Counter({0: 117, 1: 60})

len(df_f.columns) # 14 
###############
#### 종속변수(Class) 및 명목변수 삭제 
###############

df_f_no_class = df_f.drop(['Class'],axis = 1)

##########################################
#### 고유값분해를 이용한 데이터 증강 알고리즘 
##########################################
df_f.BodyFat.min() # 10.07

# BodyFat 분산 확인 (표준화 전)
df_f.BodyFat.var() # 30.021042979097345
df_f.BodyFat.mean() # 21.717821229050276

import numpy as np

###############
###데이터 MinMaxScaler
###############

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def standard_df(df):
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(df.values)
    standarded_df = pd.DataFrame(normalized_values, columns=df.columns, index=df.index)
    return standarded_df, scaler

def inverse_standard_df(standarded_df, scaler):

    original_values = scaler.inverse_transform(standarded_df.values)
    original_df = pd.DataFrame(original_values, columns=standarded_df.columns, index=standarded_df.index)
    return original_df

'''
def minmax_df(df):
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(df.values)
    normalized_df = pd.DataFrame(normalized_values, columns=df.columns, index=df.index)
    return normalized_df, scaler

def inverse_minmax_df(normalized_df, scaler):

    original_values = scaler.inverse_transform(normalized_df.values)
    original_df = pd.DataFrame(original_values, columns=normalized_df.columns, index=normalized_df.index)
    return original_df
'''
# minmax_m, scaler = minmax_df(df_m_no_class)
standarded_df, scaler = standard_df(df_f_no_class)

# BodyFat에 대한 표준화 확인 
standarded_df.BodyFat.min() # -2.131809153654107
standarded_df.BodyFat.mean() # 거의 0 
standarded_df.BodyFat.std() # 1.000002...


def calculate_covariance_matrix(x):
    cov_matrix = np.cov(x, rowvar=False)
    return cov_matrix

def calculate_eigen(cov_matrix):
    # 주성분 계산
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    # 내림차순으로 정렬
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_values, eigen_vectors, idx

def split_components(eigen_vectors, d):
    U1 = eigen_vectors[:, :d]  # 주성분(내림차순이니까)
    U2 = eigen_vectors[:, d:]  # 비주성분
    return U1, U2

def project_data(x, U1, U2):
    q = np.dot(x , U1) # (250,19) * (19,10)
    s = np.dot(x , U2)
    return q, s

def sample_q_prime(q_mean, q_variance, size):
    q_prime = np.random.normal(q_mean, np.sqrt(q_variance), size)
    # np.random.normal(평균값, 표준편차, 생성할 난수 갯수)
    # 정규분포를 따르는 난수 생성 함수
    # size=(250,10) 이면, 10개 열에 대한 난수 생성임.
    return q_prime

# generate_sample에서 난수로 생성된 q_prime 과 s를 합침
def generate_sample(q_prime, s, U1, U2): 
    sample = np.dot(q_prime, U1.T) + np.dot(s, U2.T) 
    return sample # (250,19)

def inverse_projection(sample, U):
    x_prime = np.dot(sample , U.T) 
    return x_prime

def augment_data(x, x_prime):
    x_aug = np.vstack((x, x_prime))
    return x_aug



###########
# 1. 데이터 생성

#x = minmax_m.values
x = standarded_df.values


# 2. 공분산 행렬 계산 (주대각선 1에 가까움)
cov_matrix_of_x = calculate_covariance_matrix(x)
cov_matrix_of_x.shape # 13,13


# 4. 고유값과 고유벡터 계산
eigen_values, eigen_vectors, idx = calculate_eigen(cov_matrix_of_x)

'''
주성분/ 비주성분 내림차순으로 배열됨 
이상한 점 : 내림차순 전/후가 거의 같음 => 기존 열 순서가 주성분 순서라는..
'''
idx 
'''
array([ 0,  1,  2,  3,  4,  7,  9, 10, 11, 12,  8,  6,  5]
'''


# 5. 주성분과 비주성분 분리
d = 8  # 주성분 개수
U1, U2 = split_components(eigen_vectors, d)
U1.shape # 12,8
U2.shape # 12,4

# 6. 주성분과 비주성분에 데이터 사영
q, s = project_data(x, U1, U2)

# 7. q'를 샘플링
q_mean = np.mean(q, axis=0)
q_mean # 10개 주성분 칼럼에 대한 평균
q_variance = np.var(q, axis=0)
q_variance # 10개 주성분 칼럼에 대한 분산
q_prime = sample_q_prime(q_mean, q_variance, size=(df_f.shape[0], d))

# 8. 샘플링된 q'와 s를 단순히 합침
sample = generate_sample(q_prime, s, U1, U2)
U1.shape # (12, 8)
U2.shape # (12, 4)
s.shape # (177, 4)
q_prime.shape # (177, 8)


# 9. 역사영하여 샘플링 데이터 x' 생성
x_prime = inverse_projection(sample, eigen_vectors)
x_prime.shape

# 정규성 난수로 생성된 데이터 x_prime 기초통계량 확인 
# BodyFat 0번째열
x_prime[0].min() # -1.9548576107316202
x_prime[0].mean() # -0.06408166907486013
x_prime[0].std() # 0.8610029608220321


# 10. 증강된 입력 데이터 x_aug 생성
x_aug = augment_data(x, x_prime)

x_aug.shape #  (354, 12)

# 결과 확인
print("원본 데이터 x의 형태:", x.shape)
print("데이터 증강 후 x_aug의 형태:", x_aug.shape)
'''
원본 데이터 x의 형태: (177, 13)
데이터 증강 후 x_aug의 형태: (354, 13)
'''

df_f_no_class.columns
x_aug = pd.DataFrame(x_aug, columns = df_f_no_class.columns)
x_aug.BodyFat.min() # -2.1798493891514537
x_aug.BodyFat.mean() # -0.0015080346384627092
x_aug.BodyFat.std() # 0.9330544750621741
#aug_m = inverse_minmax_df(x_aug, scaler)


# 역 표준화 
aug_m = inverse_standard_df(x_aug, scaler)
aug_m.shape # (358, 19)
aug_m.BodyFat.min() # 9.80751680636354
aug_m.describe().T
'''
          count        mean       std  ...         50%         75%         max
BodyFat   358.0   21.709582  5.098042  ...   21.545000   25.102500   37.042851
Age       358.0   20.527870  1.727673  ...   20.386312   21.857735   26.463301
Weight    358.0   60.107511  8.875560  ...   59.861283   66.403609   86.477563
Height    358.0  166.916372  5.686440  ...  166.704294  170.890294  180.300000
Neck      358.0   31.456133  1.287605  ...   31.500000   32.290842   36.000000
Chest     358.0   84.732176  6.564379  ...   85.000000   88.830941  100.883502
Abdomen   358.0   69.472625  5.250407  ...   69.154023   72.500000   96.000000
Hip       358.0   96.796244  7.142899  ...   96.815208  101.845734  116.850169
Thigh     358.0   51.330143  3.752383  ...   51.005633   53.924465   62.060423
Knee      358.0   35.689770  2.174772  ...   35.535291   37.054989   43.416101
Ankle     358.0   21.273487  1.056961  ...   21.114775   21.826437   25.000000
Biceps    358.0   26.592527  2.356651  ...   26.500000   28.000000   33.242015
Forearm   358.0   23.413607  1.915263  ...   23.443305   24.500000   29.356016
Wrist     358.0   15.637346  0.896707  ...   15.500000   16.296279   18.400000
BMI       358.0   21.498460  2.230462  ...   21.500000   22.998690   29.000000
AC_ratio  358.0    0.818597  0.073405  ...    0.810511    0.847665    1.534884
HT_ratio  358.0    1.890791  0.083990  ...    1.890304    1.944431    2.289474
WHtR      358.0    0.416399  0.043308  ...    0.415500    0.442777    0.564904
WWI       358.0    0.019828  0.003513  ...    0.019647    0.021703    0.032850

'''

aug_m = aug_m[aug_m['BodyFat'] > 10] 
# 생성된 데이터 하나가 10 이하
aug_m.shape # (350, 13)

aug_m.to_csv(r"C:/ITWILL/MiddleProject/시도/증강_1차/Oversampling_eigenvalue_decomposition_Female.csv", index = False)


##############################
######증강된 데이터 클래스 분류 (1) -  이진 분류  
##############################


aug_m #증강된 데이터 


aug_m = class_2_labeling(aug_m)
aug_m.shape
from collections import Counter

cnt = Counter(aug_m['Class'])
cnt # Counter({0: 220, 1: 130}) # 비율이 그대로 유지되는 듯 
aug_m['Class'] = aug_m['Class'].astype(int)

# 라벨링 후 덮어쓰기 
aug_m.to_csv(r"C:/ITWILL/MiddleProject/시도/증강_1차/Oversampling_eigenvalue_decomposition_Female.csv", index = False)


# 모델링 후  증강 전과 비교 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,classification_report

import pandas as pd
import numpy as np

# 로지스틱회귀 
model = LogisticRegression()

# 탐색할 하이퍼파라미터 그리드 정의
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 규제 강도
    'max_iter': [100, 200, 300, 400, 500],  # 최대 반복 횟수
}

# GridSearchCV를 사용하여 모델과 하이퍼파라미터 그리드 정의
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

####################
######증강 전 데이터 df_m
####################
# 그리드 서치를 사용하여 모델 학습

# 데이터 분할 
# X, y 분리
X = df_f[['Height','Weight','BMI']]
y = df_f['Class']


# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 0과 1을 각각 숫자 0과 1로 변환
y_train = y_train.astype(int)
y_test = y_test.astype(int)

grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", grid_search.best_params_)
# Best Hyperparameters: {'C': 1, 'max_iter': 100}

y_pred = grid_search.predict(X_test)

# 정확도 출력
print("Accuracy:", accuracy_score(y_test, y_pred))

# F1 score 출력  cf) 'average = 'weighted''는 다중 클래스 분류 문제에서 각 클래스에 대한 평가 지표를 계산할 때 사용되는 가중 평균 방법을 지정하는 매개변수
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# 분류 보고서 출력
print("Classification Report:")
print(classification_report(y_test, y_pred))
'''
Accuracy: 0.5
F1 Score: 0.5122133680273215
              precision    recall  f1-score   support

           0       0.33      0.29      0.31         7
           1       0.70      0.61      0.65        23
           2       0.17      0.20      0.18         5
           3       0.25      1.00      0.40         1

    accuracy                           0.50        36
   macro avg       0.36      0.52      0.39        36
weighted avg       0.54      0.50      0.51        36
'''

####################
######증강 후 데이터 aug_m
####################
# 그리드 서치를 사용하여 모델 학습

# 데이터 분할 
# X, y 분리
X = aug_m[['Height','Weight','BMI']]
y = aug_m['Class']


# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 0과 1을 각각 숫자 0과 1로 변환
y_train = y_train.astype(int)
y_test = y_test.astype(int)

grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", grid_search.best_params_)
# Best Hyperparameters: {'C': 0.1, 'max_iter': 100}
y_pred = grid_search.predict(X_test)

# 정확도 출력
print("Accuracy:", accuracy_score(y_test, y_pred))

# F1 score 출력  cf) 'average = 'weighted''는 다중 클래스 분류 문제에서 각 클래스에 대한 평가 지표를 계산할 때 사용되는 가중 평균 방법을 지정하는 매개변수
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# 분류 보고서 출력
print("Classification Report:")
print(classification_report(y_test, y_pred))

'''
Accuracy: 0.7361111111111112
F1 Score: 0.7079583515363332
              precision    recall  f1-score   support

           0       0.75      0.92      0.83        49
           1       0.67      0.35      0.46        23

    accuracy                           0.74        72
   macro avg       0.71      0.63      0.64        72
weighted avg       0.72      0.74      0.71        72
'''


