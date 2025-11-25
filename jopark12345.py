# streamlit_student_rf_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import io

st.set_page_config(page_title="학생 성적 예측 (랜덤 포레스트)", layout="wide")
st.title("🎓 학생 생활 패턴 기반 성적 예측")

# -----------------------------
# 데이터 로딩 (CSV 포함)
# -----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("student_habits_performance.csv")
    except FileNotFoundError:
        # CSV가 없으면 샘플 데이터 생성
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "hours_studied": np.random.normal(5, 2, 100).clip(0),
            "sleep_hours": np.random.normal(7, 1, 100).clip(4, 10),
            "study_method": np.random.choice(["group", "solo", "online"], 100),
            "exam_score": np.random.normal(70, 10, 100).clip(0, 100),
        })
        return df

df = load_data()

if df.empty:
    st.error("CSV 파일이 없거나 데이터가 비어있습니다.")
    st.stop()

st.subheader("데이터 미리보기")
st.dataframe(df.head())
st.markdown(f"행: {df.shape[0]} / 열: {df.shape[1]}")

# -----------------------------
# 컬럼 선택 (특징/타깃)
# -----------------------------
cols = df.columns.tolist()

target = st.selectbox("타깃 변수 선택", options=cols, index=cols.index("exam_score") if "exam_score" in cols else 0)
id_col = st.selectbox("식별자 컬럼 제거", options=[None] + cols)

default_features = [c for c in cols if c not in [target, id_col]]
features = st.multiselect("특징 변수 선택", options=default_features, default=default_features)

if not features:
    st.error("최소 1개의 입력 변수가 필요합니다.")
    st.stop()

X = df[features].copy()
y = df[target].copy()

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------
# 모델 설정
# -----------------------------
st.sidebar.header("모델 설정")
n_estimators = st.sidebar.slider("트리 개수", 10, 1000, 300, 10)
train_size = st.sidebar.slider("학습 비율", 0.5, 0.95, 0.8, 0.05)
random_state = st.sidebar.number_input("랜덤 시드", value=42, format="%d")

def build_model(cat_cols, num_cols, n_estimators, random_state):
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))
    ])
    return model

# -----------------------------
# 모델 학습
# -----------------------------
train_button = st.button("모델 학습 시작")

if train_button:
    with st.spinner("학습 중..."):
        model = build_model(cat_cols, num_cols, n_estimators, random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("모델 학습 완료!")
        st.metric("MAE", f"{mae:.3f}")
        st.metric("R²", f"{r2:.3f}")

        # 실제 vs 예측
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--")
        ax.set_xlabel("실제 값")
        ax.set_ylabel("예측 값")
        ax.set_title("실제 vs 예측")
        st.pyplot(fig)

        # 피처 중요도 안전하게 가져오기
        rf = model.named_steps["rf"]
        preproc = model.named_steps["preprocess"]
        feature_names = []

        # 학습 후에만 get_feature_names_out 호출
        for name, transformer, cols_ in preproc.transformers:
            if transformer == "passthrough":
                feature_names.extend(cols_)
            else:
                feature_names.extend(transformer.get_feature_names_out(cols_))

        importances = rf.feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        st.subheader("피처 중요도")
        st.dataframe(fi.head(20))

        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.barh(fi["feature"][::-1], fi["importance"][::-1])
        st.pyplot(fig2)

        # 모델 다운로드
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button("모델 다운로드 (.pkl)", buffer, "rf_model.pkl")

        # 예측 결과 다운로드
        results = X_test.copy()
        results["y_true"] = y_test
        results["y_pred"] = y_pred
        st.download_button("예측 결과 다운로드 (.csv)", results.to_csv(index=False).encode("utf-8"), "predictions.csv")
