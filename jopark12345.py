# streamlit_random_forest_app.py
# Streamlit 앱: 랜덤 포레스트 회귀 모델 (CSV 업로드 또는 기본 파일 사용)
# 사용법: 터미널에서 `streamlit run streamlit_random_forest_app.py`

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

# --- 유틸리티 함수 -------------------------------------------------

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
    """
    ColumnTransformer에서 최종 피처 이름을 추출합니다 (OneHot 처리된 이름 포함).
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers:
        if transformer == 'passthrough':
            # num passthrough
            feature_names.extend(cols)
        else:
            # transformer는 실제 estimator (OneHotEncoder)
            try:
                # sklearn >=1.0
                trans_feature_names = transformer.get_feature_names_out(cols)
            except Exception:
                # fallback
                if hasattr(transformer, 'get_feature_names'):
                    trans_feature_names = transformer.get_feature_names(cols)
                else:
                    trans_feature_names = cols
            feature_names.extend(list(trans_feature_names))
    return feature_names


# --- 레이아웃: 사이드바 (설정) --------------------------------------
st.sidebar.title("설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
use_example = st.sidebar.checkbox("샘플 데이터 사용 (data.csv)", value=True)

n_estimators = st.sidebar.slider("트리 개수 (n_estimators)", min_value=10, max_value=1000, value=300, step=10)
train_size = st.sidebar.slider("학습 비율", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
random_state = st.sidebar.number_input("랜덤 시드", value=42, format="%d")

st.sidebar.markdown("---")
st.sidebar.markdown("앱을 실행하려면 CSV 파일을 업로드하거나 `data.csv`가 앱과 동일한 디렉토리에 있어야 합니다.")

# --- 데이터 로딩 --------------------------------------------------

@st.cache_data
def load_data(uploaded_file, use_example):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if use_example:
        # 기본 경로: 같은 디렉토리의 data.csv
        try:
            return pd.read_csv("student_habits_performance.csv")
        except FileNotFoundError:
            # fallback: 최소한의 예시 데이터 생성
            df = pd.DataFrame({
                "student_id": range(1, 101),
                "hours_studied": np.random.normal(5, 2, 100).clip(0),
                "sleep_hours": np.random.normal(7, 1, 100).clip(4,10),
                "study_method": np.random.choice(["group", "solo", "online"], 100),
                "exam_score": np.random.normal(70, 10, 100).clip(0,100),
            })
            return df
    # 업로드 없고 예제 사용 안함 -> 빈 데이터프레임
    return pd.DataFrame()


df = load_data(uploaded_file, use_example)

if df.empty:
    st.error("데이터가 없습니다. CSV를 업로드하거나 '샘플 데이터 사용'을 체크하세요.")
    st.stop()

# --- 헤더 ----------------------------------------------------------
st.title("🎯 학생 성취도 예측 — 랜덤 포레스트 회귀 (Streamlit)")
st.markdown("간단한 웹 인터페이스로 데이터 확인 → 모델 학습 → 성능 확인까지 제공합니다.")

# --- 데이터 미리보기 및 컬럼 선택 --------------------------------
with st.expander("데이터 미리보기", expanded=True):
    st.dataframe(df.head(50))
    st.markdown(f"- 행: {df.shape[0]}, 열: {df.shape[1]}")

cols = df.columns.tolist()

col1, col2 = st.columns([2, 1])
with col1:
    target = st.selectbox("타깃 변수 선택 (예: exam_score)", options=cols, index=cols.index("exam_score") if "exam_score" in cols else 0)
    id_col = st.selectbox("식별자 컬럼 (제외)", options=[None] + cols, index=0)

with col2:
    default_features = [c for c in cols if c not in [target, id_col]]
    features = st.multiselect("입력 변수(특징) 선택", options=default_features, default=default_features)

if not features:
    st.error("최소 하나의 입력 변수를 선택해야 합니다.")
    st.stop()

X = df[features].copy()
y = df[target].copy()

# 자동으로 범주형 / 수치형 분류
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

st.markdown("---")
st.markdown(f"**범주형 변수:** {cat_cols if cat_cols else '없음'}  
**수치형 변수:** {num_cols if num_cols else '없음'}")

# --- 모델 학습 버튼 -----------------------------------------------
train_button = st.button("모델 학습 시작")

if train_button:
    with st.spinner("학습 중... 잠시만 기다려주세요"):
        model = build_preprocess_and_model(cat_cols, num_cols, n_estimators, random_state)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_state)

        # fit
        model.fit(X_train, y_train)

        # 예측 및 지표
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("학습 완료!")
        st.metric("MAE", f"{mae:.3f}")
        st.metric("R²", f"{r2:.3f}")

        # 예측 vs 실제 산점도
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
        ax.set_xlabel('실제 값')
        ax.set_ylabel('예측 값')
        ax.set_title('실제 vs 예측')
        st.pyplot(fig)

        # 피처 중요도 (랜덤포레스트의 feature_importances_ 사용)
        rf = model.named_steps['rf']
        preproc = model.named_steps['preprocess']
        try:
            feature_names = get_feature_names_from_preprocessor(preproc, X.columns)
        except Exception:
            # 실패 시 그냥 원래 columns 사용
            feature_names = X.columns.tolist()

        try:
            importances = rf.feature_importances_
            fi = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi = fi.sort_values(by="importance", ascending=False).head(30)

            st.subheader("피처 중요도 상위 30개")
            st.dataframe(fi.reset_index(drop=True))

            # 그래프
            fig2, ax2 = plt.subplots(figsize=(6, max(3, 0.2 * len(fi))))
            ax2.barh(fi['feature'][::-1], fi['importance'][::-1])
            ax2.set_title('Feature Importances')
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"피처 중요도 계산 불가: {e}")

        # 모델 다운로드 버튼 (직렬화)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        st.download_button("모델 다운로드 (.pkl)", data=buffer, file_name="rf_model.pkl")

        # 예측 결과 다운로드
        results = X_test.copy()
        results['y_true'] = y_test
        results['y_pred'] = y_pred
        csv_buf = results.to_csv(index=False).encode('utf-8')
        st.download_button("예측 결과 다운로드 (.csv)", data=csv_buf, file_name="predictions.csv")

# --- 바닥글: 간단 사용 팁 -----------------------------------------
st.markdown("---")
st.markdown("**팁:** CSV 파일에는 결측값/문자열이 섞여 있을 수 있으니 전처리를 추가하면 성능이 좋아집니다.")
st.markdown("필요하면 앱을 더 예쁘게 만들거나 하이퍼파라미터/교차검증 기능을 추가해드릴게요.")
