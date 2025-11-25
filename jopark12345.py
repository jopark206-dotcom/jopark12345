# streamlit_random_forest_app.py
# Streamlit 앱: 랜덤 포레스트 회귀 모델

import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="학생 습관 예측 (랜덤 포레스트)", layout="wide")

# -------------------------------------------------
# 유틸리티 함수
# -------------------------------------------------

def build_preprocess_and_model(cat_cols, num_cols, n_estimators, random_state=42):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
    ])

    return model


def get_feature_names_from_preprocessor(preprocessor, X_columns):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers:
        if transformer == "passthrough":
            feature_names.extend(cols)
        else:
            try:
                trans_feature_names = transformer.get_feature_names_out(cols)
            except Exception:
                trans_feature_names = cols
            feature_names.extend(list(trans_feature_names))
    return feature_names

# -------------------------------------------------
# 사이드바 설정
# -------------------------------------------------

st.sidebar.title("설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
use_example = st.sidebar.checkbox("샘플 데이터 사용 (student_habits_performance.csv)", value=True)

n_estimators = st.sidebar.slider("트리 개수 (n_estimators)", 10, 1000, 300, 10)
train_size = st.sidebar.slider("학습 비율", 0.5, 0.95, 0.8, 0.05)
random_state = st.sidebar.number_input("랜덤 시드", value=42, format="%d")

st.sidebar.markdown("---")
st.sidebar.markdown("CSV를 업로드하거나 data.csv 파일을 프로젝트에 넣어주세요.")

# -------------------------------------------------
# 데이터 로딩
# -------------------------------------------------

@st.cache_data
def load_data(uploaded_file, use_example):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if use_example:
        try:
            return pd.read_csv("student_habits_performance.csv")
        except FileNotFoundError:
            df = pd.DataFrame({
                "student_id": range(1, 101),
                "hours_studied": np.random.normal(5, 2, 100).clip(0),
                "sleep_hours": np.random.normal(7, 1, 100).clip(4, 10),
                "study_method": np.random.choice(["group", "solo", "online"], 100),
                "exam_score": np.random.normal(70, 10, 100).clip(0, 100),
            })
            return df

    return pd.DataFrame()


df = load_data(uploaded_file, use_example)

if df.empty:
    st.error("데이터가 없습니다. CSV 파일을 업로드하거나 샘플 데이터를 사용하세요.")
    st.stop()

# -------------------------------------------------
# 헤더
# -------------------------------------------------

st.title("🎯 학생 성취도 예측 — 랜덤 포레스트 회귀")
st.markdown("웹에서 바로 확인하는 EDA + 모델 학습 + 예측 + 다운로드 기능 제공")

# -------------------------------------------------
# 데이터 미리보기
# -------------------------------------------------

with st.expander("데이터 미리보기", expanded=True):
    st.dataframe(df.head())
    st.markdown(f"행: {df.shape[0]}  /  열: {df.shape[1]}")

cols = df.columns.tolist()

col1, col2 = st.columns([2, 1])

with col1:
    target = st.selectbox("타깃 변수 선택", options=cols, index=cols.index("exam_score") if "exam_score" in cols else 0)
    id_col = st.selectbox("식별자 컬럼 제거", options=[None] + cols)

with col2:
    default_features = [c for c in cols if c not in [target, id_col]]
    features = st.multiselect("특징 변수 선택", options=default_features, default=default_features)

if not features:
    st.error("최소 1개의 입력 변수가 필요합니다.")
    st.stop()

X = df[features].copy()
y = df[target].copy()

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

st.markdown("---")
st.markdown(
    f"**범주형 변수:** {cat_cols if cat_cols else '없음'}  
"
    f"**수치형 변수:** {num_cols if num_cols else '없음'}"
)

# -------------------------------------------------
# 모델 학습 버튼
# -------------------------------------------------

train_button = st.button("모델 학습 시작")

if train_button:
    with st.spinner("학습 중..."):
        model = build_preprocess_and_model(cat_cols, num_cols, n_estimators, random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_size, random_state=random_state
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("모델 학습 완료!")
        st.metric("MAE", f"{mae:.3f}")
        st.metric("R²", f"{r2:.3f}")

        # ------------------------
        # 실제 vs 예측 그래프
        # ------------------------
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--")
        ax.set_xlabel("실제 값")
        ax.set_ylabel("예측 값")
        ax.set_title("실제 vs 예측")
        st.pyplot(fig)

        # ------------------------
        # 피처 중요도
        # ------------------------
        rf = model.named_steps["rf"]
        preproc = model.named_steps["preprocess"]

        try:
            feature_names = get_feature_names_from_preprocessor(preproc, X.columns)
            importances = rf.feature_importances_

            fi = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi = fi.sort_values("importance", ascending=False).head(20)

            st.subheader("피처 중요도")
            st.dataframe(fi)

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.barh(fi["feature"][::-1], fi["importance"][::-1])
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"피처 중요도 계산 불가: {e}")

        # ------------------------
        # 모델 다운로드
        # ------------------------
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button("모델 다운로드 (.pkl)", buffer, "rf_model.pkl")

        # ------------------------
        # 예측 결과 다운로드
        # ------------------------
        results = X_test.copy()
        results["y_true"] = y_test
        results["y_pred"] = y_pred

        st.download_button(
            "예측 결과 다운로드 (.csv)",
            results.to_csv(index=False).encode("utf-8"),
            "predictions.csv"
        )

# -------------------------------------------------
# 바닥글
# -------------------------------------------------

st.markdown("---")
st.markdown("추가 기능(샘플링, 교차검증, SHAP 해석 등) 원하면 알려주세요.")

