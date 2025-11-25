# 랜덤 포레스트 회귀 분석 예제 코드
# 파일: student_habits_performance.csv 사용

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 데이터 불러오기
df = pd.read_csv("data.csv")

# 2. 타깃 변수와 입력 변수 분리
target = "exam_score"                     # 예측할 값
X = df.drop(columns=[target, "student_id"])  # student_id는 식별자라서 제거
y = df[target]

# 3. 범주형 / 수치형 변수 나누기
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("범주형 변수:", cat_cols)
print("수치형 변수:", num_cols)

# 4. 전처리 파이프라인 설정
#    - 범주형: One-Hot 인코딩
#    - 수치형: 그대로 사용(passthrough)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# 5. 랜덤 포레스트 회귀 모델 파이프라인 구성
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=300,      # 트리 개수
        random_state=42,       # 재현성
        n_jobs=-1              # 멀티코어 사용(가능하다면)
    ))
])

# 6. 학습용 / 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20%를 테스트 데이터로 사용
    random_state=42
)

# 7. 모델 학습
model.fit(X_train, y_train)

# 8. 예측
y_pred = model.predict(X_test)

# 9. 성능 평가 (MAE, R²)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (평균 절대 오차): {mae:.3f}")
print(f"R² (설명력): {r2:.3f}")
