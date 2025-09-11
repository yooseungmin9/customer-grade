# app.py
# -*- coding: utf-8 -*-
"""
Streamlit 고객 등급(클러스터) 예측 앱
- 버튼 클릭으로 학습 시작, 진행률 표시
- 파이프라인: 로딩 → 정제 → KMeans(n=4) → LGBM 학습 → 예측 UI → 시각화
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
# 상수
# -----------------------------
SEED: int = 42
CLUSTER_K: int = 4
st.set_page_config(page_title="고객 등급 예측(클러스터링+LGBM)", layout="wide")
alt.themes.enable("opaque")

# -----------------------------
# 유틸: 데이터 로더(다중 인코딩 폴백)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """다중 인코딩 폴백으로 CSV/Excel을 로드."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    # 최후 수단
    return pd.read_csv(path, encoding="utf-8", sep=None, engine="python", errors="replace")

# -----------------------------
# 전처리
# -----------------------------
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    sel = [
        "annual_income","Recency","Monetary","Frequency",
        "num_purchase_store","num_purchase_web","num_purchase_discount",
    ]
    return df[sel].copy()

def drop_missing_and_top_outlier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    max_val = df["annual_income"].max()
    return df[df["annual_income"] < max_val].copy()

def iqr_clean(df: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    mask = ~((df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr))).any(axis=1)
    return df[mask].reset_index(drop=True)

# -----------------------------
# KMeans 라벨링
# -----------------------------
def fit_kmeans_labels(df_num: pd.DataFrame, k: int = CLUSTER_K) -> Tuple[np.ndarray, float]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_num.values)
    km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
    labels = km.fit_predict(x_scaled)
    sil = silhouette_score(x_scaled, labels)
    return labels, sil

def make_tier_mapping(df_with_labels: pd.DataFrame) -> Dict[int, str]:
    means = (
        df_with_labels.groupby("cluster")[["Monetary","Frequency"]]
        .mean()
        .assign(score=lambda d: 0.6*d["Monetary"] + 0.4*d["Frequency"])
        .sort_values("score")
    )
    tiers = ["브론즈 등급","실버 등급","골드 등급","플래티넘 등급"]
    return {int(c): tiers[i] for i, c in enumerate(means.index.tolist())}

# -----------------------------
# 지도학습(LGBM)
# -----------------------------
def train_lgbm(features: pd.DataFrame, labels: pd.Series, random_search: bool = False,
               n_iter: int = 15) -> Tuple[LGBMClassifier, Dict]:
    """LGBM 학습. random_search=True면 경량 랜덤서치."""
    xtr, xte, ytr, yte = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    if random_search:
        base = LGBMClassifier(random_state=SEED)
        space = {
            "n_estimators": randint(60, 140),
            "learning_rate": uniform(0.01, 0.15),
            "max_depth": randint(3, 12),
            "min_child_samples": randint(10, 80),
            "reg_alpha": uniform(0.0, 0.6),
            "reg_lambda": uniform(0.0, 0.6),
            "num_leaves": randint(7, 48),
        }
        gs = RandomizedSearchCV(base, space, n_iter=n_iter, cv=3, n_jobs=-1, random_state=SEED)
        gs.fit(xtr, ytr)
        model = LGBMClassifier(random_state=SEED, **gs.best_params_)
    else:
        model = LGBMClassifier(random_state=SEED)
    model.fit(xtr, ytr)
    meta = {
        "train_score": model.score(xtr, ytr),
        "test_score": model.score(xte, yte),
        "best_params": getattr(model, "get_params", lambda: {})(),
    }
    return model, meta

# -----------------------------
# 시각화
# -----------------------------
def pca_scatter(df_num: pd.DataFrame, labels: np.ndarray):
    x = StandardScaler().fit_transform(df_num.values)
    xy = PCA(n_components=2, random_state=SEED).fit_transform(x)
    chart_df = pd.DataFrame({"x": xy[:,0], "y": xy[:,1], "cluster": labels.astype(int)})
    return alt.Chart(chart_df).mark_circle(size=60, opacity=0.7).encode(
        x="x:Q", y="y:Q", color="cluster:N", tooltip=["x","y","cluster"]
    ).properties(height=420)

# -----------------------------
# 앱 메인
# -----------------------------
def main() -> None:
    st.title("고객 등급 예측 서비스 · 클러스터링+LGBM")
    data_path = os.getenv("CUSTOMER_DATA_PATH", "data/customer_data.csv")
    st.caption(f"데이터 경로: `{data_path}`")

    # 사이드바: 학습 제어
    st.sidebar.header("모델 학습")
    use_search = st.sidebar.checkbox("랜덤서치 사용(느림)", value=False)
    n_iter = st.sidebar.slider("랜덤서치 횟수", 5, 50, 15, 5)
    start_train = st.sidebar.button("학습 시작", use_container_width=True)

    # 데이터는 즉시 로드하여 PCA와 매핑을 먼저 보여줌
    df_raw = load_data(data_path)
    df = iqr_clean(drop_missing_and_top_outlier(select_columns(df_raw)))
    if df.empty:
        st.error("전처리 결과가 비었습니다. IQR 기준을 완화하거나 원본을 점검하세요.")
        return

    labels, sil = fit_kmeans_labels(df, k=CLUSTER_K)
    df_labeled = df.copy()
    df_labeled["cluster"] = labels
    tier_map = make_tier_mapping(df_labeled)

    # 상단 메트릭과 그래프는 즉시 표시
    c1, c2 = st.columns(2)
    c1.metric("실루엣 점수(KMeans)", f"{sil:.3f}")
    st.subheader("PCA 2D 산점도")
    st.altair_chart(pca_scatter(df, labels), use_container_width=True)

    st.subheader("클러스터 ↔ 등급 매핑")
    st.dataframe(
        pd.DataFrame({"cluster": list(tier_map.keys()), "tier": list(tier_map.values())})
        .sort_values("cluster"),
        use_container_width=True,
    )

    # 진행률 바
    progress = st.progress(0, text="대기 중")

    # 학습 버튼 클릭 시에만 모델 학습
    if start_train:
        progress.progress(10, text="데이터 분리 중")
        feature_cols = df.columns.tolist()
        X, y = df_labeled[feature_cols], df_labeled["cluster"]

        with st.spinner("모델 학습 중..."):
            progress.progress(40, text="모델 구성")
            model, meta = train_lgbm(X, y, random_search=use_search, n_iter=n_iter)
            progress.progress(85, text="평가 및 저장")
            # 학습 결과 세션에 저장하여 재사용
            st.session_state["model"] = model
            st.session_state["feature_cols"] = feature_cols
            st.session_state["tier_map"] = tier_map
            st.session_state["meta"] = meta
            progress.progress(100, text="학습 완료")

        # 성능 표시
        c1, c2 = st.columns(2)
        c1.metric("Train Acc", f"{meta['train_score']:.3f}")
        c2.metric("Test Acc", f"{meta['test_score']:.3f}")
        st.code(f"best_params = {meta['best_params']}", language="python")

    # 예측 UI: 학습 완료 후에만 노출
    if "model" in st.session_state:
        st.sidebar.header("입력 특성")
        inputs: Dict[str, float] = {}
        for col in st.session_state["feature_cols"]:
            vmin = float(df[col].min()); vmax = float(df[col].max()); vmean = float(df[col].mean())
            if col.startswith("num_purchase") or col in ["Frequency","Recency"]:
                inputs[col] = st.sidebar.slider(col, vmin, vmax, float(int(vmean)), step=1.0)
            else:
                inputs[col] = st.sidebar.number_input(col, value=float(round(vmean, 2)))
        if st.sidebar.button("예측하기", use_container_width=True):
            input_df = pd.DataFrame([inputs], columns=st.session_state["feature_cols"])
            pred = int(st.session_state["model"].predict(input_df)[0])
            tier = st.session_state["tier_map"].get(pred, f"cluster {pred}")
            st.success(f"예측된 회원등급: **{tier}** (cluster={pred})")
            # 특성 중요도
            imp = pd.Series(
                st.session_state["model"].feature_importances_,
                index=st.session_state["feature_cols"]
            ).sort_values(ascending=False)
            st.subheader("특성 중요도")
            st.bar_chart(imp)
    else:
        st.info("왼쪽 사이드바의 **학습 시작** 버튼을 눌러 모델을 학습하세요.")

if __name__ == "__main__":
    main()

# 사용 예시
# 1) repo 루트에 data/customer_data.csv 존재
# 2) pip install -r requirements.txt
# 3)
