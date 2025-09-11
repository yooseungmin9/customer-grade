# app.py
# -*- coding: utf-8 -*-
"""
Streamlit 고객 등급(클러스터) 예측 앱
- 파이프라인: 로딩 → 정제 → KMeans(n=4) → LGBM 분류 학습 → 예측 UI → 시각화
- 입력 데이터: data/customer_data.csv (cp949 가정)
[PEP8][NumPy Docstring]
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from lightgbm import LGBMClassifier
import altair as alt

# -----------------------------
# 상수 및 공통 설정
# -----------------------------
SEED = 42
CLUSTER_K = 4  # 사용자 코드와 동일하게 4개 고정
ALT_THEME = "opaque"  # Altair 테마

st.set_page_config(page_title="고객 등급 예측(클러스터링+LGBM)", layout="wide")
alt.themes.enable(ALT_THEME)


# -----------------------------
# 데이터 로딩/정제
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str = "data/customer_data.csv") -> pd.DataFrame:
    """CSV 데이터를 로드합니다.

    Parameters
    ----------
    path : str
        CSV 경로. repo 내 data/customer_data.csv 가정.

    Returns
    -------
    pd.DataFrame
        원본 DataFrame
    """
    # 한글 칼럼 가정으로 cp949 기본
    return pd.read_csv(path, encoding="cp949")


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 칼럼만 선택합니다.

    Returns
    -------
    pd.DataFrame
        선택 칼럼 서브셋
    """
    sel = [
        "annual_income",
        "Recency",
        "Monetary",
        "Frequency",
        "num_purchase_store",
        "num_purchase_web",
        "num_purchase_discount",
    ]
    return df[sel].copy()


def drop_missing_and_top_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """결측 제거, annual_income 최댓값 한 점 제거(사용자 로직 유지)."""
    df = df.dropna().copy()
    max_val = df["annual_income"].max()
    return df[df["annual_income"] < max_val].copy()


def iqr_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """IQR(1.5*IQR)로 행 필터링. 마스크도 함께 반환.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        (정제된 DF, 사용된 불리언 마스크)
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    mask = ~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)
    return df[mask].reset_index(drop=True), mask


# -----------------------------
# 비지도(KMeans) → 라벨 생성
# -----------------------------
def fit_kmeans_labels(df_num: pd.DataFrame, k: int = CLUSTER_K) -> Tuple[np.ndarray, float, StandardScaler]:
    """표준화 후 KMeans로 클러스터 라벨 및 실루엣 점수 계산.

    Returns
    -------
    labels : np.ndarray
        클러스터 라벨
    sil : float
        실루엣 점수
    scaler : StandardScaler
        학습에 사용한 스케일러(추후 시각화/PCA에 재사용)
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_num.values)
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
    labels = kmeans.fit_predict(x_scaled)
    sil = silhouette_score(x_scaled, labels)
    return labels, sil, scaler


def make_tier_mapping(df_with_labels: pd.DataFrame) -> Dict[int, str]:
    """클러스터별 Monetary, Frequency 평균으로 등급명을 매핑.

    규칙
    ----
    score = 0.6*Monetary_mean + 0.4*Frequency_mean
    낮은 점수 → 브론즈, 높은 점수 → 플래티넘
    """
    means = (
        df_with_labels.groupby("cluster")[["Monetary", "Frequency"]]
        .mean()
        .assign(score=lambda d: 0.6 * d["Monetary"] + 0.4 * d["Frequency"])
        .sort_values("score")
    )
    tiers = ["브론즈 등급", "실버 등급", "골드 등급", "플래티넘 등급"]
    mapping = {int(c): tiers[i] for i, c in enumerate(means.index.tolist())}
    return mapping


# -----------------------------
# 지도→라벨 예측 모델(LGBM)
# -----------------------------
@st.cache_resource(show_spinner=False)
def train_lgbm(
        features: pd.DataFrame, labels: pd.Series
) -> Tuple[LGBMClassifier, Dict]:
    """LGBM 분류기 학습 + 랜덤 서치로 하이퍼파라미터 탐색(빠르게 축소).

    Returns
    -------
    model : LGBMClassifier
    best_params : dict
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    base = LGBMClassifier(random_state=SEED)
    search_space = {
        "n_estimators": randint(50, 150),
        "learning_rate": uniform(0.01, 0.2),
        "max_depth": randint(3, 16),
        "min_child_samples": randint(10, 100),
        "reg_alpha": uniform(0.0, 0.8),
        "reg_lambda": uniform(0.0, 0.8),
        "num_leaves": randint(7, 63),
    }

    gs = RandomizedSearchCV(
        estimator=base,
        param_distributions=search_space,
        n_iter=40,  # 실행시간을 고려해 40으로 축소
        n_jobs=-1,
        cv=3,
        random_state=SEED,
        verbose=0,
    )
    gs.fit(x_train, y_train)

    best_params = gs.best_params_
    model = LGBMClassifier(random_state=SEED, **best_params)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    meta = {
        "best_params": best_params,
        "train_score": train_score,
        "test_score": test_score,
    }
    return model, meta


# -----------------------------
# 시각화
# -----------------------------
def pca_scatter(df_num: pd.DataFrame, labels: np.ndarray) -> alt.Chart:
    """PCA 2D 산점도 Altair."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_num.values)
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(x_scaled)
    chart_df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": labels.astype(int)})

    return (
        alt.Chart(chart_df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("cluster:N", legend=alt.Legend(title="cluster")),
            tooltip=["x", "y", "cluster"],
        )
        .properties(height=420)
    )


# -----------------------------
# UI
# -----------------------------
def main() -> None:
    st.title("고객 등급 예측 서비스 · 클러스터링+LGBM")

    # 데이터 경로 설정(환경변수 허용)
    data_path = os.getenv("CUSTOMER_DATA_PATH", "data/customer_data.csv")
    st.caption(f"데이터 경로: `{data_path}`")

    # 1) 데이터 로드 및 전처리
    df_raw = load_data(data_path)
    df_sel = select_columns(df_raw)
    df_base = drop_missing_and_top_outlier(df_sel)
    df_clean, mask = iqr_clean(df_base)

    st.sidebar.header("입력 특성")
    st.sidebar.caption("아래 슬라이더 값으로 단건 예측을 수행합니다.")

    # 2) KMeans 라벨 생성
    labels, sil, scaler_for_view = fit_kmeans_labels(df_clean, k=CLUSTER_K)
    df_with_labels = df_clean.copy()
    df_with_labels["cluster"] = labels

    # 등급명 매핑 동적 생성
    cluster_to_tier = make_tier_mapping(df_with_labels)

    # 3) 지도학습 데이터 구성
    label_col = "cluster"
    feature_cols = df_with_labels.columns.difference([label_col]).tolist()
    X = df_with_labels[feature_cols].copy()
    y = df_with_labels[label_col].copy()

    # 4) LGBM 학습
    model, meta = train_lgbm(X, y)

    # 좌측 입력 위젯 범위 자동 생성
    inputs: Dict[str, float] = {}
    for col in feature_cols:
        v_min = float(df_with_labels[col].min())
        v_max = float(df_with_labels[col].max())
        v_mean = float(df_with_labels[col].mean())
        # 정수성 추정: 구매 횟수류는 정수 슬라이더로 노출
        if col.startswith("num_purchase") or col in ["Frequency", "Recency"]:
            inputs[col] = st.sidebar.slider(col, v_min, v_max, value=float(int(v_mean)), step=1.0)
        else:
            inputs[col] = st.sidebar.number_input(col, value=float(round(v_mean, 2)))

    predict_btn = st.sidebar.button("예측하기", use_container_width=True)

    # 메트릭 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("실루엣 점수(KMeans)", f"{sil:.3f}")
    c2.metric("LGBM 학습 정확도(train)", f"{meta['train_score']:.3f}")
    c3.metric("LGBM 테스트 정확도(test)", f"{meta['test_score']:.3f}")

    # 시각화
    st.subheader("PCA 2D 산점도")
    st.altair_chart(pca_scatter(df_clean, labels), use_container_width=True)

    # 등급 매핑 테이블
    st.subheader("클러스터 등급 매핑")
    map_df = (
        pd.DataFrame({"cluster": list(cluster_to_tier.keys()), "tier": list(cluster_to_tier.values())})
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    st.dataframe(map_df, use_container_width=True)

    # 예측
    if predict_btn:
        input_df = pd.DataFrame([inputs], columns=feature_cols)
        pred_cluster = int(model.predict(input_df)[0])
        tier_name = cluster_to_tier.get(pred_cluster, f"cluster {pred_cluster}")
        st.success(f"예측된 회원등급: **{tier_name}** (cluster={pred_cluster})")

        # 특성 중요도
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        st.subheader("특성 중요도")
        st.bar_chart(imp)


if __name__ == "__main__":
    main()

# -----------------------------
# 사용 예시(로컬)
# -----------------------------
# 1) repo 루트에 data/customer_data.csv 배치
# 2) 가상환경 생성 후 requirements 설치
#    pip install -r requirements.txt
# 3) 실행
#    streamlit run app.py