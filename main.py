import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

st.set_page_config(page_title="콘크리트 강도 안전 실습실", page_icon="🏗️", layout="wide")

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
# CSV의 열 순서: 시멘트, 고로슬래그, 플라이애시, 물, 고성능감수제,
#               굵은골재, 잔골재, 재령(일), 압축강도(MPa)  (총 9개)
COLS = [
    "시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제",
    "굵은골재", "잔골재", "재령(일)", "압축강도(MPa)",
]
INPUTS = COLS[:8]
TARGET = COLS[8]
LOCAL_FILE = "concrete.csv"  # main.py와 같은 폴더에 두면 자동으로 읽음

st.title("🏗️ 콘크리트 강도가 약해서 생기는 안전문제")
st.caption("배합(재료 양)과 재령(굳은 날짜)으로 콘크리트 압축강도를 예측하고, 기준에 못 미치는 '위험 배합'을 찾아보는 실습 앱")


@st.cache_data(show_spinner=False)
def read_url(url):
    return pd.read_csv(url)


def clean(raw):
    """열이 정확히 9개인 CSV만 받아서 한글 열 이름을 붙인다."""
    if raw.shape[1] != 9:
        return None
    df = raw.copy()
    df.columns = COLS
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df[df["시멘트"] > 0].reset_index(drop=True)


# ------------------------------------------------------------------
# 사이드바 ① 데이터 불러오기
# ------------------------------------------------------------------
with st.sidebar:
    st.header("① 데이터")
    url = st.text_input("CSV 주소(URL)", value="", placeholder="https://.../concrete.csv")
    up = st.file_uploader("또는 CSV 파일 올리기", type="csv")

raw = None
try:
    if up is not None:
        raw = pd.read_csv(up)
    elif url.strip():
        raw = read_url(url.strip())
    elif os.path.exists(LOCAL_FILE):
        raw = pd.read_csv(LOCAL_FILE)
except Exception as e:
    st.error(f"CSV를 읽지 못했어요: {e}")
    st.stop()

if raw is None:
    st.info(
        "데이터가 필요해요. 아래 세 가지 중 하나로 불러오세요.\n\n"
        "1. 왼쪽 사이드바에 CSV 주소(URL) 붙여넣기\n"
        "2. 왼쪽 사이드바에서 CSV 파일 올리기\n"
        f"3. `{LOCAL_FILE}` 파일을 main.py와 같은 폴더(깃허브 저장소)에 넣기\n\n"
        "CSV는 열이 정확히 9개여야 해요: 시멘트, 고로슬래그, 플라이애시, 물, 고성능감수제, "
        "굵은골재, 잔골재, 재령(일), 압축강도(MPa) 순서 (열 이름은 달라도 순서만 맞으면 돼요)."
    )
    st.stop()

df = clean(raw)
if df is None:
    st.error(f"열이 9개가 아니에요 (지금은 {raw.shape[1]}개). 위 순서의 9개 열만 있는 CSV를 사용하세요.")
    st.stop()

# ------------------------------------------------------------------
# 사이드바 ② 안전 기준, ③ 모델 설정
# ------------------------------------------------------------------
with st.sidebar:
    st.header("② 안전 기준")
    limit = st.slider("안전 기준 강도(MPa) — 이보다 약하면 '위험'", 10, 50, 24)
    only28 = st.checkbox("재령 28일 이상 배합만 사용", value=False)

    st.header("③ 모델 설정")
    feats = st.multiselect("입력 속성", INPUTS, default=INPUTS)
    test_pct = st.slider("테스트 데이터 비율(%)", 10, 40, 25, step=5)
    depth = st.slider("의사결정트리 깊이(질문 횟수)", 1, 6, 3)

if only28:
    df = df[df["재령(일)"] >= 28].reset_index(drop=True)
if len(df) < 30:
    st.error(f"데이터가 {len(df)}줄뿐이라 분석하기 어려워요. 조건을 바꿔 보세요.")
    st.stop()
if len(feats) == 0:
    st.warning("입력 속성을 하나 이상 골라 주세요.")
    st.stop()

df["위험"] = (df[TARGET] < limit).astype(int)  # 1 = 기준 미달(위험)
df["구분"] = df["위험"].map({0: "안전", 1: "위험"})
df["물시멘트비"] = df["물"] / df["시멘트"]

# ------------------------------------------------------------------
# 모델 학습 (탭에서 같이 쓰도록 미리 학습)
# ------------------------------------------------------------------
X = df[feats]
y_strength = df[TARGET]
y_risk = df["위험"]

can_cls = y_risk.nunique() == 2 and y_risk.value_counts().min() >= 2
X_tr, X_te, s_tr, s_te, r_tr, r_te = train_test_split(
    X, y_strength, y_risk,
    test_size=test_pct / 100,
    random_state=42,
    stratify=y_risk if can_cls else None,
)

lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, s_tr)
if can_cls:
    log = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(X_tr, r_tr)
    tree = DecisionTreeClassifier(
        max_depth=depth, class_weight="balanced", random_state=42
    ).fit(X_tr, r_tr)

tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터", "🔎 안전 문제 탐색", "🤖 예측 모델", "🧪 내 배합 진단"])

# ------------------------------------------------------------------
# 탭1: 데이터
# ------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("배합 수(줄)", f"{len(df):,}")
    c2.metric("평균 압축강도", f"{df[TARGET].mean():.1f} MPa")
    c3.metric("안전 기준", f"{limit} MPa")
    c4.metric("기준 미달(위험) 비율", f"{df['위험'].mean() * 100:.1f}%")

    st.subheader("처음 5줄")
    st.dataframe(df[COLS].head(), use_container_width=True)

    st.subheader("열 설명")
    st.dataframe(
        pd.DataFrame(
            {
                "열": COLS,
                "단위": ["kg/m³"] * 7 + ["일", "MPa"],
                "최솟값": [df[c].min() for c in COLS],
                "평균": [df[c].mean() for c in COLS],
                "최댓값": [df[c].max() for c in COLS],
            }
        ).round(1),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("강도가 약한 콘크리트는 왜 위험할까? (일반적인 설명)"):
        st.markdown(
            "- **내력 부족**: 기둥·보가 설계한 하중을 버티지 못해 균열, 처짐이 커질 수 있어요.\n"
            "- **내구성 저하**: 조직이 치밀하지 않으면 물·염분·이산화탄소가 스며들어 철근이 부식하기 쉬워요.\n"
            "- **시공 중 사고**: 거푸집을 너무 일찍 떼면(재령이 짧으면) 아직 약한 콘크리트가 무너질 수 있어요.\n"
            "- 그래서 현장에서는 시험으로 압축강도를 확인하고 기준에 못 미치면 보강·재시공을 검토해요."
        )

# ------------------------------------------------------------------
# 탭2: 안전 문제 탐색
# ------------------------------------------------------------------
with tab2:
    st.subheader("압축강도 분포와 안전 기준")
    fig = px.histogram(df, x=TARGET, nbins=40, color="구분",
                       color_discrete_map={"안전": "#2E86AB", "위험": "#D62828"})
    fig.add_vline(x=limit, line_dash="dash", line_color="black",
                  annotation_text=f"기준 {limit} MPa")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("물시멘트비와 강도")
        fig = px.scatter(df, x="물시멘트비", y=TARGET, color="구분", opacity=0.6,
                         color_discrete_map={"안전": "#2E86AB", "위험": "#D62828"})
        fig.add_hline(y=limit, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("물시멘트비 = 물 ÷ 시멘트. 물이 많을수록 강도가 낮아지는 경향이 보이는지 확인해 보세요.")
    with right:
        st.subheader("재령(굳은 기간)별 위험 비율")
        d = df.copy()
        d["재령 구간"] = pd.cut(
            d["재령(일)"], [0, 3, 7, 14, 28, 90, 400],
            labels=["1~3일", "4~7일", "8~14일", "15~28일", "29~90일", "91일~"],
        )
        rate = (d.groupby("재령 구간", observed=True)["위험"].mean() * 100).reset_index()
        rate.columns = ["재령 구간", "위험 비율(%)"]
        fig = px.bar(rate, x="재령 구간", y="위험 비율(%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("재령이 짧을수록 아직 덜 굳어서 기준 미달이 많아지는지 살펴보세요.")

    st.subheader("압축강도와 상관관계가 큰 속성")
    corr = df[INPUTS + ["물시멘트비"]].corrwith(df[TARGET]).sort_values().reset_index()
    corr.columns = ["속성", "상관계수"]
    fig = px.bar(corr, x="상관계수", y="속성", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("상관관계가 크다고 해서 원인이라는 뜻은 아니에요. 양수면 함께 커지고, 음수면 반대로 움직이는 경향이에요.")

# ------------------------------------------------------------------
# 탭3: 예측 모델
# ------------------------------------------------------------------
with tab3:
    st.subheader("① 선형회귀: 압축강도(MPa) 숫자 맞히기")
    pred = lin.predict(X_te)
    m1, m2, m3 = st.columns(3)
    m1.metric("평균 절대 오차(MAE)", f"{mean_absolute_error(s_te, pred):.2f} MPa")
    m2.metric("결정계수(R²)", f"{r2_score(s_te, pred):.3f}")
    m3.metric("테스트 데이터 수", f"{len(s_te)}")

    lo, hi = float(min(s_te.min(), pred.min())), float(max(s_te.max(), pred.max()))
    fig = px.scatter(x=s_te, y=pred, labels={"x": "실제 강도(MPa)", "y": "예측 강도(MPa)"}, opacity=0.7)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽한 예측",
                             line=dict(dash="dot", color="gray")))
    fig.add_hline(y=limit, line_dash="dash", line_color="#D62828")
    fig.add_vline(x=limit, line_dash="dash", line_color="#D62828")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("빨간 선은 안전 기준이에요. 실제로는 기준 미달(세로선 왼쪽)인데 예측은 기준 이상(가로선 위)인 점이 '놓친 위험'이에요.")

    coef = pd.DataFrame({"속성": feats, "영향(표준화 계수)": lin[-1].coef_}).sort_values("영향(표준화 계수)")
    fig = px.bar(coef, x="영향(표준화 계수)", y="속성", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("② 분류: 안전 / 위험 맞히기")
    if not can_cls:
        st.warning("지금 기준에서는 '안전'과 '위험' 중 한쪽이 거의 없어서 분류 모델을 만들 수 없어요. 사이드바에서 안전 기준을 바꿔 보세요.")
    else:
        def score(y_true, y_pred):
            return {
                "정확도": accuracy_score(y_true, y_pred),
                "재현율": recall_score(y_true, y_pred, zero_division=0),
                "정밀도": precision_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
            }

        majority = int(r_tr.mean() >= 0.5)
        p_log = log.predict(X_te)
        p_tree = tree.predict(X_te)
        table = pd.DataFrame(
            {
                "로지스틱 회귀": score(r_te, p_log),
                f"의사결정트리(깊이 {depth})": score(r_te, p_tree),
                "다수 클래스 기준 모델": score(r_te, [majority] * len(r_te)),
            }
        ).T.round(3)
        st.dataframe(table, use_container_width=True)
        st.caption("'위험'을 양성(1)으로 두었어요. 재현율 = 실제 위험 배합 중 모델이 찾아낸 비율 (놓치면 사고로 이어질 수 있어서 중요해요).")

        choice = st.radio("혼동행렬을 볼 모델", ["로지스틱 회귀", "의사결정트리"], horizontal=True)
        p = p_log if choice == "로지스틱 회귀" else p_tree
        tn, fp, fn, tp = confusion_matrix(r_te, p, labels=[0, 1]).ravel()
        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["실제 안전", "실제 위험"],
            columns=["예측 안전", "예측 위험"],
        )
        a, b = st.columns([1, 1])
        a.dataframe(cm, use_container_width=True)
        b.metric("놓친 위험 배합(FN)", f"{fn}개")
        b.metric("헛짚은 경보(FP)", f"{fp}개")

        with st.expander("의사결정트리 가지 그림"):
            dot = export_graphviz(
                tree, feature_names=feats, class_names=["안전", "위험"],
                filled=True, rounded=True, impurity=False,
            )
            st.graphviz_chart(dot)

# ------------------------------------------------------------------
# 탭4: 내 배합 진단
# ------------------------------------------------------------------
with tab4:
    st.subheader("배합을 직접 정해서 진단해 보기")
    st.caption("모델은 사이드바에서 고른 입력 속성만 사용해요. (학습용 예측이며, 실제 구조물의 안전 판단은 시험과 전문가의 확인이 필요해요.)")

    vals = {}
    cols = st.columns(2)
    for i, name in enumerate(feats):
        lo, hi = float(df[name].min()), float(df[name].max())
        default = float(df[name].median())
        if name == "재령(일)" and lo <= 28 <= hi:
            default = 28.0
        if lo == hi:
            vals[name] = lo
            continue
        vals[name] = cols[i % 2].slider(name, lo, hi, default)

    row = pd.DataFrame([vals])[feats]
    strength = float(lin.predict(row)[0])

    r1, r2 = st.columns(2)
    r1.metric("예상 압축강도", f"{strength:.1f} MPa", f"기준 대비 {strength - limit:+.1f} MPa")
    if can_cls:
        prob = float(log.predict_proba(row)[0][1])
        r2.metric("위험 확률(로지스틱 회귀)", f"{prob * 100:.0f}%")
        if prob >= 0.5 or strength < limit:
            st.error("⚠️ 기준 강도에 못 미칠 가능성이 커요. 물을 줄이거나 시멘트를 늘리거나, 충분히 굳은 뒤에 쓰는 것을 생각해 보세요.")
        else:
            st.success("✅ 이 배합은 기준 강도를 넘을 것으로 예측돼요.")
    else:
        if strength < limit:
            st.error("⚠️ 예상 강도가 기준보다 낮아요.")
        else:
            st.success("✅ 예상 강도가 기준 이상이에요.")
