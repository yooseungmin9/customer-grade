"""
Streamlit 고객 등급(클러스터) 예측 앱 - 개선 버전
- 한국어 컬럼명 지원
- 탭 기반 UI로 직관적인 경험 제공
- 학습 → 예측 → 시각화 분리
- 실시간 진행률 표시 및 상세 피드백
"""
from __future__ import annotations

import os
from typing import Dict, Tuple
import pickle
from datetime import datetime

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

# ═══════════════════════════════════════════════════════════════
# 설정 및 상수
# ═══════════════════════════════════════════════════════════════
SEED: int = 42
CLUSTER_K: int = 4

st.set_page_config(
    page_title="고객 등급 예측 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.themes.enable("opaque")

# 한국어 컬럼 매핑
COLUMN_NAME_KR = {
    "annual_income": "연소득(원)",
    "Recency": "최근구매경과일(일)",
    "Monetary": "총구매합산액(원)",
    "Frequency": "총구매빈도(회)",
    "num_purchase_store": "매장구매횟수(회)",
    "num_purchase_web": "온라인구매횟수(회)",
    "num_purchase_discount": "할인구매횟수(회)",
}

COLUMN_NAME_EN = {v: k for k, v in COLUMN_NAME_KR.items()}

TIER_NAMES = ["🥉 브론즈 등급", "🥈 실버 등급", "🥇 골드 등급", "💎 플래티넘 등급"]

# ═══════════════════════════════════════════════════════════════
# 데이터 로더
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """다중 인코딩 폴백으로 CSV/Excel 로드."""
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

    return pd.read_csv(path, encoding="utf-8", sep=None, engine="python", errors="replace")

# ═══════════════════════════════════════════════════════════════
# 전처리 함수들
# ═══════════════════════════════════════════════════════════════
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """필요 컬럼 선택."""
    sel = [
        "annual_income", "Recency", "Monetary", "Frequency",
        "num_purchase_store", "num_purchase_web", "num_purchase_discount",
    ]
    return df[sel].copy()

def drop_missing_and_top_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 제거 및 최대값 이상치 제거."""
    df = df.dropna().copy()
    max_val = df["annual_income"].max()
    return df[df["annual_income"] < max_val].copy()

def iqr_clean(df: pd.DataFrame) -> pd.DataFrame:
    """IQR 기반 이상치 제거."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    mask = ~((df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr))).any(axis=1)
    return df[mask].reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# KMeans 클러스터링
# ═══════════════════════════════════════════════════════════════
def fit_kmeans_labels(df_num: pd.DataFrame, k: int = CLUSTER_K) -> Tuple[np.ndarray, float]:
    """KMeans 클러스터링 및 실루엣 점수."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_num.values)
    km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
    labels = km.fit_predict(x_scaled)
    sil = silhouette_score(x_scaled, labels)
    return labels, sil

def make_tier_mapping(df_with_labels: pd.DataFrame) -> Dict[int, str]:
    """클러스터 → 등급 매핑 (Monetary & Frequency 기반)."""
    means = (
        df_with_labels.groupby("cluster")[["Monetary", "Frequency"]]
        .mean()
        .assign(score=lambda d: 0.6*d["Monetary"] + 0.4*d["Frequency"])
        .sort_values("score")
    )
    return {int(c): TIER_NAMES[i] for i, c in enumerate(means.index.tolist())}

# ═══════════════════════════════════════════════════════════════
# LGBM 학습
# ═══════════════════════════════════════════════════════════════
def train_lgbm(features: pd.DataFrame, labels: pd.Series, 
               random_search: bool = False, n_iter: int = 15) -> Tuple[LGBMClassifier, Dict]:
    """LGBM 분류 모델 학습."""
    xtr, xte, ytr, yte = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    if random_search:
        base = LGBMClassifier(random_state=SEED, verbose=-1)
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
        best_params = gs.best_params_
        model = LGBMClassifier(random_state=SEED, verbose=-1, **best_params)
    else:
        best_params = {}
        model = LGBMClassifier(random_state=SEED, verbose=-1)

    model.fit(xtr, ytr)

    meta = {
        "train_score": model.score(xtr, ytr),
        "test_score": model.score(xte, yte),
        "best_params": best_params,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return model, meta

# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════
def pca_scatter(df_num: pd.DataFrame, labels: np.ndarray, tier_map: Dict) -> alt.Chart:
    """PCA 2D 산점도."""
    x = StandardScaler().fit_transform(df_num.values)
    xy = PCA(n_components=2, random_state=SEED).fit_transform(x)

    chart_df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "cluster": labels.astype(int),
        "tier": [tier_map.get(int(c), f"cluster {c}") for c in labels]
    })

    return alt.Chart(chart_df).mark_circle(size=80, opacity=0.8).encode(
        x=alt.X("x:Q", title="주요성분 1"),
        y=alt.Y("y:Q", title="주요성분 2"),
        color=alt.Color("tier:N", title="등급"),
        tooltip=["tier:N", "cluster:N", "x:Q", "y:Q"]
    ).properties(height=450, title="고객 세분화 분포도")

def feature_importance_chart(model: LGBMClassifier, feature_cols: list) -> alt.Chart:
    """특성 중요도 차트."""
    imp = pd.Series(
        model.feature_importances_,
        index=[COLUMN_NAME_KR.get(c, c) for c in feature_cols]
    ).sort_values(ascending=True)

    return alt.Chart(imp.reset_index().rename(columns={"index": "특성", 0: "중요도"})).mark_barh().encode(
        x="중요도:Q",
        y=alt.Y("특성:N", sort="-x")
    ).properties(height=300, title="특성 중요도 분석")

def cluster_stats(df_labeled: pd.DataFrame, tier_map: Dict) -> pd.DataFrame:
    """등급별 통계."""
    stats = df_labeled.groupby("cluster").agg({
        "annual_income": ["mean", "count"],
        "Monetary": "mean",
        "Frequency": "mean",
    }).round(0)
    stats.columns = ["평균소득(원)", "고객수", "평균구매액(원)", "평균구매빈도"]
    stats.index = [tier_map.get(i, f"cluster {i}") for i in stats.index]
    return stats

# ═══════════════════════════════════════════════════════════════
# 세션 초기화
# ═══════════════════════════════════════════════════════════════
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
    st.session_state["model"] = None
    st.session_state["feature_cols"] = None
    st.session_state["tier_map"] = None
    st.session_state["meta"] = None

# ═══════════════════════════════════════════════════════════════
# 메인 앱
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    st.title("💼 고객 등급 예측 시스템")
    st.markdown("KMeans 클러스터링 + LightGBM으로 고객을 세분화하고 등급을 예측합니다.")

    # 데이터 경로 (환경변수 또는 기본값)
    data_path = os.getenv("CUSTOMER_DATA_PATH", "data/customer_data.csv")

    # ─────────────────────────────────────────────────────────
    # 1단계: 데이터 로드 및 전처리
    # ─────────────────────────────────────────────────────────
    try:
        df_raw = load_data(data_path)
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info(f"데이터 파일을 `{data_path}` 경로에 저장하세요.")
        return

    # 전처리
    df = iqr_clean(drop_missing_and_top_outlier(select_columns(df_raw)))

    if df.empty:
        st.error("❌ 전처리 후 데이터가 없습니다. 원본 데이터를 점검하세요.")
        return

    # 클러스터링
    labels, sil = fit_kmeans_labels(df, k=CLUSTER_K)
    df_labeled = df.copy()
    df_labeled["cluster"] = labels
    tier_map = make_tier_mapping(df_labeled)

    # ─────────────────────────────────────────────────────────
    # 탭 구성
    # ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 데이터 분석", "🤖 모델 학습", "🎯 고객 등급 예측"])

    # ─────────────────────────────────────────────────────────
    # 탭 1: 데이터 분석
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.header("고객 데이터 분석")

        col1, col2, col3 = st.columns(3)
        col1.metric("📈 총 고객 수", f"{len(df):,}")
        col2.metric("🎯 클러스터 수", CLUSTER_K)
        col3.metric("⭐ 실루엣 점수", f"{sil:.3f}")

        st.subheader("데이터 요약")
        summary_df = df.copy()
        summary_df.columns = [COLUMN_NAME_KR.get(c, c) for c in summary_df.columns]
        st.dataframe(summary_df.describe().round(0), use_container_width=True)

        st.subheader("고객 분포도 (PCA)")
        st.altair_chart(pca_scatter(df, labels, tier_map), use_container_width=True)

        st.subheader("등급별 통계")
        st.dataframe(cluster_stats(df_labeled, tier_map), use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # 탭 2: 모델 학습
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.header("LightGBM 모델 학습")

        col1, col2 = st.columns(2)
        with col1:
            use_search = st.checkbox("🔍 하이퍼파라미터 튜닝 활성화 (느림)", value=False)
        with col2:
            if use_search:
                n_iter = st.slider("튜닝 반복 횟수", 5, 50, 15, 5)
            else:
                n_iter = 0

        if st.button("🚀 모델 학습 시작", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📥 데이터 준비 중...")
                progress_bar.progress(10)

                feature_cols = df.columns.tolist()
                X, y = df_labeled[feature_cols], df_labeled["cluster"]

                status_text.text("🤖 모델 학습 중...")
                progress_bar.progress(30)

                model, meta = train_lgbm(X, y, random_search=use_search, n_iter=n_iter)

                status_text.text("💾 모델 저장 중...")
                progress_bar.progress(90)

                # 세션에 저장
                st.session_state["model"] = model
                st.session_state["feature_cols"] = feature_cols
                st.session_state["tier_map"] = tier_map
                st.session_state["meta"] = meta
                st.session_state["model_trained"] = True

                progress_bar.progress(100)
                status_text.text("✅ 학습 완료!")

                # 성능 결과
                st.success("모델 학습이 완료되었습니다! 예측 탭에서 사용할 수 있습니다.")

                col1, col2 = st.columns(2)
                col1.metric("✅ 학습 정확도", f"{meta['train_score']:.1%}")
                col2.metric("🧪 테스트 정확도", f"{meta['test_score']:.1%}")

                if meta["best_params"]:
                    st.subheader("🔧 최적 하이퍼파라미터")
                    st.json(meta["best_params"])

                st.info(f"학습 완료: {meta['trained_at']}")

            except Exception as e:
                st.error(f"❌ 학습 중 오류: {e}")

        else:
            st.info("위 버튼을 클릭하여 모델 학습을 시작하세요.")

    # ─────────────────────────────────────────────────────────
    # 탭 3: 고객 등급 예측
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.header("고객 등급 예측")

        if not st.session_state["model_trained"]:
            st.warning("⚠️ 먼저 **모델 학습** 탭에서 모델을 학습하세요.")
        else:
            st.success("✅ 모델이 준비되었습니다. 고객 정보를 입력해주세요.")

            # 입력 폼
            st.subheader("고객 정보 입력")

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)
            col7 = st.columns(1)[0]

            inputs: Dict[str, float] = {}

            feature_cols = st.session_state["feature_cols"]

            # 각 특성별 입력 위젯
            inputs_layout = [
                (col1, "annual_income"),
                (col2, "Recency"),
                (col3, "Monetary"),
                (col4, "Frequency"),
                (col5, "num_purchase_store"),
                (col6, "num_purchase_web"),
                (col7, "num_purchase_discount"),
            ]

            for col, col_en in inputs_layout:
                if col_en in feature_cols:
                    vmin = float(df[col_en].min())
                    vmax = float(df[col_en].max())
                    vmean = float(df[col_en].mean())
                    col_kr = COLUMN_NAME_KR.get(col_en, col_en)

                    if col_en.startswith("num_purchase") or col_en in ["Frequency", "Recency"]:
                        inputs[col_en] = col.slider(
                            col_kr,
                            min_value=int(vmin),
                            max_value=int(vmax),
                            value=int(vmean),
                            step=1
                        )
                    else:
                        inputs[col_en] = col.number_input(
                            col_kr,
                            min_value=vmin,
                            value=round(vmean, 0),
                            step=1000.0 if col_en == "annual_income" else 100.0
                        )

            # 예측 버튼
            if st.button("🎯 등급 예측하기", use_container_width=True, type="primary"):
                input_df = pd.DataFrame([inputs], columns=feature_cols)
                pred = int(st.session_state["model"].predict(input_df)[0])
                pred_proba = st.session_state["model"].predict_proba(input_df)[0]

                tier = st.session_state["tier_map"].get(pred, f"cluster {pred}")

                st.balloons()
                st.success(f"# {tier}")

                # 확률 분포
                st.subheader("예측 신뢰도")
                proba_df = pd.DataFrame({
                    "등급": [st.session_state["tier_map"].get(i, f"cluster {i}") for i in range(len(pred_proba))],
                    "확률": pred_proba
                })
                st.bar_chart(proba_df.set_index("등급"))

                # 특성 중요도
                st.subheader("의사결정 근거 (특성 중요도)")
                st.altair_chart(feature_importance_chart(st.session_state["model"], feature_cols), use_container_width=True)

if __name__ == "__main__":
    main()
