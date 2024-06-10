# -*- coding: utf-8 -*-

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,classification_report
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.inspection import permutation_importance
import shap # pip install shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# 데이터 불러오기
df_f = pd.read_csv(r"C:\Users\user\Downloads\Dataset.csv", encoding='CP949')
df_f.info()
df_f = df_f[['Height','Weight','Chest','Abdomen','Hip','Thigh','Biceps','Ankle','Knee','Neck','Wrist','Class']]
X = df_f.drop('Class',axis = 1)
y = df_f['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train.shape
X_test.shape

# smote 기법을 이용,  SMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None)
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
print("SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ", X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 값의 분포 :\n',y_train_over.value_counts() )
'''
 Class
1    126
0    126
'''
X_prime = X_train_over
y_prime = y_train_over

# LogisticRegression 모델 생성
model = LogisticRegression()

scaler = RobustScaler()
X_prime = scaler.fit_transform(X_prime)
X_prime = pd.DataFrame(data= X_prime, columns = X_train_over.columns)

# grid search cv를 통한 파라미터 최적화
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 규제 강도
    'max_iter': [100, 200, 300, 400, 500],  # 최대 반복 횟수
}



# GridSearchCV를 사용하여 모델과 하이퍼파라미터 그리드 정의
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# 그리드 서치를 사용하여 모델 학습
grid_search.fit(X_prime, y_prime)

# 최적의 하이퍼파라미터 출력

print("Best Hyperparameters:", grid_search.best_params_)
print("최적의 성능:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# feature importance

feature_importance = pd.DataFrame({'Feature': X_train_over.columns, 'Importance': best_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.title('Feature importance : LogisticRegression (Coeff.)')
plt.show()


# permutation_importance

result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=40)

per_imp_df = pd.DataFrame({'Feature': X.columns,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
per_imp_df = per_imp_df.sort_values('Importance', ascending=True)

ax = per_imp_df.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), yerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance with Standard Deviation (Logistic Regression)')
plt.show()


# SHAP summary plot 그리기

explainer = shap.Explainer(best_model, X_prime)
shap_values = explainer.shap_values(X_prime)

shap.summary_plot(shap_values, X_prime)
# SHAP 요약 플롯 생성

X_prime2 = X_prime[['Height', 'Weight', 'Chest', 'Abdomen' , 'Hip', 'Knee' ]]
X_test_SHAP = X_test[['Height', 'Weight', 'Chest', 'Abdomen' , 'Hip', 'Knee' ]]

"""# SHAP 토대로 다시 test"""

# LogisticRegression 모델 생성
model = LogisticRegression()

# grid search cv를 통한 파라미터 최적화
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 규제 강도
    'max_iter': [100, 200, 300, 400, 500],  # 최대 반복 횟수
}


# GridSearchCV를 사용하여 모델과 하이퍼파라미터 그리드 정의
grid_search2 = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# 그리드 서치를 사용하여 모델 학습
grid_search2.fit(X_prime2, y_prime)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", grid_search2.best_params_)
print("최적의 성능:", grid_search2.best_score_)
best_model2 = grid_search2.best_estimator_


# 트레인 데이터로 예측
X_prime2.columns
y_pre = best_model2.predict(X_prime2)

# 정확도 출력
print("train Accuracy:", accuracy_score(y_prime, y_pre)) 

# F1 score 출력  c
print("train F1 Score:", f1_score(y_prime, y_pre, average='weighted')) 

# 테스트 데이터로 예측
y_pred = best_model2.predict(X_test_SHAP)

# 정확도 출력
print("Accuracy:", accuracy_score(y_test, y_pred))

# F1 score 출력  
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# 분류 보고서 출력
print("Classification Report:")
print(classification_report(y_test, y_pred))

