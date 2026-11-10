import glob
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="콘크리트 강도 안전 실습실", page_icon="🏗️", layout="wide")
st.title("🏗️ 콘크리트 강도가 약해서 생기는 안전문제")
st.caption("국토안전관리원 콘크리트 압축강도 시험 자료로, 어떤 조건에서 강도가 기준에 못 미치는지 민감도 분석으로 살펴보는 앱")

NONE = "(사용 안 함)"
ID_WORDS = ["번호", "연번", "순번", "코드", "ID"]
FEAT_WORDS = ["반발경도", "재령", "타격각도", "측정", "보정", "계수", "설계기준강도"]


# ------------------------------------------------------------------
# 도우미 함수
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_bytes(b):
    """공공데이터 CSV는 인코딩이 제각각이라 여러 방식으로 시도한다."""
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc)
        except Exception:
            continue
    return None


def find(cols, words):
    for w in words:
        for c in cols:
            if w in c:
                return c
    return None


def make_model(name):
    if name == "선형회귀":
        return make_pipeline(StandardScaler(), LinearRegression())
    return RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42)


# ------------------------------------------------------------------
# 사이드바 ① 데이터
# ------------------------------------------------------------------
with st.sidebar:
    st.header("① 데이터")
    up = st.file_uploader("CSV 파일 올리기", type="csv")
    local = sorted(glob.glob("*.csv"))
    pick = None
    if up is None and local:
        idx = next((i for i, f in enumerate(local) if "콘크리트" in f), 0)
        pick = st.selectbox("저장소(같은 폴더) 안의 CSV", local, index=idx)

if up is not None:
    raw_bytes = up.getvalue()
elif pick is not None:
    with open(pick, "rb") as f:
        raw_bytes = f.read()
else:
    st.info(
        "CSV 파일이 필요해요. 두 가지 중 하나로 불러오세요.\n\n"
        "1. 왼쪽 사이드바에서 CSV 파일 올리기\n"
        "2. CSV 파일을 main.py와 같은 폴더(깃허브 저장소)에 넣기\n\n"
        "(내 컴퓨터의 C:\\ 경로는 웹 앱이 직접 읽을 수 없어요.)"
    )
    st.stop()

raw = read_bytes(raw_bytes)
if raw is None or raw.shape[1] < 3:
    st.error("CSV를 읽지 못했어요. 파일이 올바른 CSV인지 확인해 주세요.")
    st.stop()

raw.columns = [str(c).strip() for c in raw.columns]
df0 = raw.copy()
num_cols = []
for c in df0.columns:
    s = pd.to_numeric(df0[c].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    if s.notna().mean() >= 0.5 and s.nunique() > 1:
        df0[c] = s
        num_cols.append(c)
cat_cols = [c for c in df0.columns if c not in num_cols and 1 < df0[c].nunique() <= 15]

if len(num_cols) < 3:
    st.error("숫자로 된 열이 3개 미만이라 분석하기 어려워요.")
    st.stop()

# ------------------------------------------------------------------
# 사이드바 ② 열 고르기 / 안전 기준 / 모델
# ------------------------------------------------------------------
with st.sidebar:
    st.header("② 열 고르기")
    t_def = find(num_cols, ["코어강도", "추정강도", "압축강도", "코어"]) or num_cols[-1]
    target = st.selectbox("강도 열(예측할 값)", num_cols, index=num_cols.index(t_def))

    d_opts = [NONE] + [c for c in num_cols if c != target]
    d_def = find(d_opts[1:], ["설계기준강도", "설계강도", "기준강도", "설계"])
    design = st.selectbox("설계기준강도 열", d_opts, index=d_opts.index(d_def) if d_def else 0)

    use_design = False
    if design != NONE:
        use_design = st.radio(
            "위험 기준",
            ["설계기준강도보다 약하면 위험", "직접 정한 값보다 약하면 위험"],
        ) == "설계기준강도보다 약하면 위험"
    if not use_design:
        limit = st.number_input("안전 기준 강도(MPa)", 1.0, 200.0, 24.0, 1.0)

    cand = [c for c in num_cols if c != target and not any(w in c for w in ID_WORDS)]
    f_def = [c for c in cand if any(w in c for w in FEAT_WORDS)] or cand[:8]
    if use_design:
        f_def = [c for c in f_def if c != design]
    feats = st.multiselect("분석에 쓸 속성(숫자 열)", cand, default=f_def)
    if use_design:
        feats = [c for c in feats if c != design]
    st.caption("강도와 거의 같은 값을 가진 열(예: 추정강도)을 넣으면 의미 없는 모델이 돼요.")

    g_opts = [NONE] + cat_cols
    g_def = find(cat_cols, ["시설", "구분", "종류"])
    group = st.selectbox("시설물 구분 열(선택)", g_opts, index=g_opts.index(g_def) if g_def else 0)

    st.header("③ 모델 설정")
    model_name = st.radio("예측 모델", ["선형회귀", "랜덤포레스트"], horizontal=True)
    test_pct = st.slider("테스트 데이터 비율(%)", 10, 40, 25, step=5)

if len(feats) == 0:
    st.warning("분석에 쓸 속성을 하나 이상 골라 주세요.")
    st.stop()

# ------------------------------------------------------------------
# 분석용 데이터 만들기
# ------------------------------------------------------------------
need = [target] + feats + ([design] if use_design else [])
extra = [group] if group != NONE else []
d = df0[need + extra].copy().dropna(subset=need)
if use_design:
    d = d[d[design] > 0]
if group != NONE:
    d[group] = d[group].fillna("미상").astype(str)
d = d.reset_index(drop=True)

if len(d) < 30:
    st.error(f"빈 값을 빼고 나니 {len(d)}줄만 남았어요. 속성을 줄이거나 다른 열을 골라 보세요.")
    st.stop()

d["기준강도"] = d[design] if use_design else float(limit)
d["위험"] = (d[target] < d["기준강도"]).astype(int)  # 1 = 기준 미달(위험)
d["구분"] = d["위험"].map({0: "안전", 1: "위험"})

# ------------------------------------------------------------------
# 모델 학습
# ------------------------------------------------------------------
X_tr, X_te, s_tr, s_te = train_test_split(d[feats], d[target], test_size=test_pct / 100, random_state=42)
model = make_model(model_name).fit(X_tr, s_tr)
pred_te = model.predict(X_te)
r2 = r2_score(s_te, pred_te)

tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터", "🔎 안전 문제 탐색", "🤖 예측 모델", "🎛️ 민감도 분석"])

# ------------------------------------------------------------------
# 탭1: 데이터
# ------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("원본 줄 수", f"{len(raw):,}")
    c2.metric("분석에 쓴 줄 수", f"{len(d):,}")
    c3.metric(f"평균 {target}", f"{d[target].mean():.1f}")
    c4.metric("기준 미달(위험) 비율", f"{d['위험'].mean() * 100:.1f}%")

    st.subheader("분석에 쓴 데이터")
    st.dataframe(d[need + extra + ["기준강도", "구분"]].head(20), use_container_width=True)

    with st.expander("원본 열 목록과 빈 값 비율"):
        info = pd.DataFrame(
            {
                "열": raw.columns,
                "종류": ["숫자" if c in num_cols else "문자" for c in raw.columns],
                "비어 있는 비율(%)": [round(raw[c].isna().mean() * 100, 1) for c in raw.columns],
            }
        )
        st.dataframe(info, use_container_width=True, hide_index=True)
    st.caption("이 자료는 이미 지어진 시설물(교량, 댐, 터널 등)을 시험한 결과라서, 물·시멘트 같은 배합 정보는 들어 있지 않아요.")

# ------------------------------------------------------------------
# 탭2: 안전 문제 탐색
# ------------------------------------------------------------------
with tab2:
    color_map = {"안전": "#2E86AB", "위험": "#D62828"}
    st.subheader("강도 분포와 안전 기준")
    t2 = d.copy()
    if use_design:
        t2["강도비(실제 ÷ 설계기준)"] = t2[target] / t2["기준강도"]
        fig = px.histogram(t2, x="강도비(실제 ÷ 설계기준)", nbins=40, color="구분", color_discrete_map=color_map)
        fig.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="설계기준강도")
    else:
        fig = px.histogram(t2, x=target, nbins=40, color="구분", color_discrete_map=color_map)
        fig.add_vline(x=float(limit), line_dash="dash", line_color="black", annotation_text=f"기준 {limit}")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        xf = st.selectbox("가로축 속성", feats)
        fig = px.scatter(d, x=xf, y=target, color="구분", opacity=0.6, color_discrete_map=color_map)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**강도와의 상관계수**")
        corr = d[feats].corrwith(d[target]).dropna().sort_values().reset_index()
        corr.columns = ["속성", "상관계수"]
        fig = px.bar(corr, x="상관계수", y="속성", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("상관이 크다고 원인은 아니에요. 양수면 함께 커지는 경향이에요.")

    if group != NONE:
        st.subheader(f"{group}별 위험 비율")
        g = d.groupby(group)["위험"].agg(["mean", "count"]).reset_index()
        g["위험 비율(%)"] = g["mean"] * 100
        fig = px.bar(g, x=group, y="위험 비율(%)", hover_data=["count"])
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 탭3: 예측 모델
# ------------------------------------------------------------------
with tab3:
    st.subheader(f"{model_name}으로 {target} 예측하기")
    m1, m2, m3 = st.columns(3)
    m1.metric("결정계수(R²)", f"{r2:.3f}")
    m2.metric("평균 절대 오차(MAE)", f"{mean_absolute_error(s_te, pred_te):.2f}")
    m3.metric("테스트 데이터 수", f"{len(s_te)}")

    lo, hi = float(min(s_te.min(), pred_te.min())), float(max(s_te.max(), pred_te.max()))
    fig = px.scatter(x=s_te, y=pred_te, labels={"x": "실제 강도", "y": "예측 강도"}, opacity=0.7)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽한 예측", line=dict(dash="dot", color="gray")))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("'위험인가 안전인가' 맞히기 성적")
    lim_te = d.loc[X_te.index, "기준강도"]
    r_true = (s_te < lim_te).astype(int)
    r_pred = pd.Series(pred_te < lim_te.values, index=X_te.index).astype(int)
    a1, a2, a3 = st.columns(3)
    a1.metric("정확도", f"{accuracy_score(r_true, r_pred):.3f}")
    a2.metric("재현율(위험을 찾아낸 비율)", f"{recall_score(r_true, r_pred, zero_division=0):.3f}")
    a3.metric("정밀도", f"{precision_score(r_true, r_pred, zero_division=0):.3f}")
    tn, fp, fn, tp = confusion_matrix(r_true, r_pred, labels=[0, 1]).ravel()
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["실제 안전", "실제 위험"], columns=["예측 안전", "예측 위험"])
    st.dataframe(cm, use_container_width=True)
    st.caption(f"놓친 위험(실제 위험인데 안전이라 예측): {fn}건. 안전 문제에서는 이 숫자가 작을수록 좋아요.")

    if model_name == "선형회귀":
        vals, title = model[-1].coef_, "표준화 계수(양수면 클수록 강도 증가)"
    else:
        vals, title = model.feature_importances_, "변수 중요도"
    imp = pd.DataFrame({"속성": feats, title: vals}).sort_values(title)
    st.plotly_chart(px.bar(imp, x=title, y="속성", orientation="h"), use_container_width=True)

# ------------------------------------------------------------------
# 탭4: 민감도 분석
# ------------------------------------------------------------------
with tab4:
    st.subheader("민감도 분석: 무엇이 달라지면 강도가 기준에 가까워질까?")
    st.info(
        "반발경도 같은 값은 사람이 마음대로 바꾸는 조절값이 아니라 '강도 상태를 알려 주는 측정 신호'예요. "
        "그래서 결과는 '어떤 조건이 강도와 가장 민감하게 연결되는가'로 해석하세요."
    )
    if r2 < 0.5:
        st.warning(f"모델의 R²가 {r2:.2f}로 낮아요. 민감도 결과를 조심해서 해석하세요. (다른 모델이나 속성도 시도해 보세요.)")

    # 출발점 정하기
    mode = st.radio("출발점(기준 사례)", ["전체 중앙값", "데이터에서 한 건 고르기"], horizontal=True)
    if mode == "전체 중앙값":
        base = d[feats].median().astype(float)
        base_limit = float(d["기준강도"].median())
    else:
        i = int(st.number_input("행 번호(0부터)", min_value=0, max_value=len(d) - 1, value=0, step=1))
        base = d.loc[i, feats].astype(float)
        base_limit = float(d.loc[i, "기준강도"])
        st.dataframe(d.loc[[i], feats + [target, "기준강도"]], use_container_width=True)

    base_df = pd.DataFrame([base.to_dict()])[feats]
    p0 = float(model.predict(base_df)[0])
    safe_now = p0 >= base_limit

    b1, b2, b3 = st.columns(3)
    b1.metric("예상 강도", f"{p0:.1f}", f"기준 대비 {p0 - base_limit:+.1f}")
    b2.metric("기준 강도", f"{base_limit:.1f}")
    b3.metric("판정", "안전" if safe_now else "위험")

    def sweep(f, n=60):
        a, z = float(d[f].min()), float(d[f].max())
        grid = [a + (z - a) * k / (n - 1) for k in range(n)]
        frame = pd.concat([base_df] * n, ignore_index=True)
        frame[f] = grid
        return grid, model.predict(frame)

    # (A) 토네이도: 속성을 조금 올리거나 내렸을 때
    st.divider()
    st.markdown("### ① 속성을 조금씩 바꿔 보기 (토네이도 차트)")
    pct = st.slider("바꾸는 크기(속성의 일반적인 범위의 %)", 5, 30, 10, step=5)
    rows = []
    for f in feats:
        step = float(d[f].quantile(0.95) - d[f].quantile(0.05)) * pct / 100
        down, up_ = base_df.copy(), base_df.copy()
        down[f] -= step
        up_[f] += step
        rows.append(
            {
                "속성": f,
                f"내렸을 때": float(model.predict(down)[0]) - p0,
                f"올렸을 때": float(model.predict(up_)[0]) - p0,
            }
        )
    tor = pd.DataFrame(rows)
    tor["영향"] = tor[["내렸을 때", "올렸을 때"]].abs().max(axis=1)
    order = tor.sort_values("영향")["속성"].tolist()
    long = tor.melt(id_vars="속성", value_vars=["내렸을 때", "올렸을 때"], var_name="방향", value_name="강도 변화")
    fig = px.bar(long, x="강도 변화", y="속성", color="방향", orientation="h", barmode="relative")
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": order})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("막대가 길수록 그 속성이 조금만 변해도 예상 강도가 크게 흔들려요 (다른 속성은 출발점 값으로 고정).")

    # (B) 한 속성을 쭉 바꿔 보기
    st.divider()
    st.markdown("### ② 한 속성을 쭉 바꿔 보기")
    sf = st.selectbox("살펴볼 속성", feats, key="sweep_feat")
    grid, preds = sweep(sf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d[sf], y=d[target], mode="markers", name="실제 측정 자료", opacity=0.25))
    fig.add_trace(go.Scatter(x=grid, y=preds, mode="lines", name="모델 예측(다른 속성은 고정)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[float(base[sf])], y=[p0], mode="markers", name="출발점", marker=dict(size=14, symbol="star")))
    fig.add_hline(y=base_limit, line_dash="dash", line_color="#D62828", annotation_text="기준 강도")
    fig.update_layout(xaxis_title=sf, yaxis_title=target)
    st.plotly_chart(fig, use_container_width=True)

    ok = [g for g, p in zip(grid, preds) if p >= base_limit]
    if len(ok) == len(grid):
        st.success(f"{sf}이(가) 전 구간에서 바뀌어도 예상 강도가 기준 이상이에요.")
    elif not ok:
        st.warning(f"{sf}만 바꿔서는 예상 강도가 기준에 닿지 못해요. 다른 속성도 함께 봐야 해요.")
    elif preds[-1] > preds[0]:
        st.success(f"다른 조건이 같다면 {sf}이(가) 약 {min(ok):.1f} 이상일 때 예상 강도가 기준 이상이에요.")
    else:
        st.success(f"다른 조건이 같다면 {sf}이(가) 약 {max(ok):.1f} 이하일 때 예상 강도가 기준 이상이에요.")

    # (C) 속성 하나만 바꿔서 판정이 뒤집히려면
    st.divider()
    if safe_now:
        st.markdown("### ③ 안전 여유: 어디까지 변하면 기준 아래로 떨어질까?")
    else:
        st.markdown("### ③ 기준에 닿으려면: 속성 하나를 얼마나 바꿔야 할까?")
    need_rows = []
    for f in feats:
        g_, p_ = sweep(f)
        hits = [(abs(g - float(base[f])), g) for g, p in zip(g_, p_) if (p >= base_limit) != safe_now]
        rng = float(d[f].max() - d[f].min()) or 1.0
        row = {"속성": f, "현재값": float(base[f]), "판정이 바뀌는 값": None, "변화량": None, "범위 대비 이동(%)": None}
        if hits:
            g = min(hits)[1]
            row["판정이 바뀌는 값"] = g
            row["변화량"] = g - float(base[f])
            row["범위 대비 이동(%)"] = abs(g - float(base[f])) / rng * 100
        need_rows.append(row)
    need_df = pd.DataFrame(need_rows).sort_values("범위 대비 이동(%)").round(2)
    st.dataframe(need_df, use_container_width=True, hide_index=True)
    st.caption("위쪽일수록 조금만 변해도 판정이 바뀌는 속성이에요. 값이 비어 있으면 그 속성 하나만으로는 판정이 바뀌지 않아요.")

    # (D) 전체 중요도
    st.divider()
    st.markdown("### ④ 전체 데이터 기준 중요도 (순열 중요도)")
    with st.spinner("계산 중..."):
        pi = permutation_importance(model, X_te, s_te, n_repeats=5, random_state=42, scoring="r2")
    pdf = pd.DataFrame({"속성": feats, "R² 감소량": pi.importances_mean}).sort_values("R² 감소량")
    st.plotly_chart(px.bar(pdf, x="R² 감소량", y="속성", orientation="h"), use_container_width=True)
    st.caption("속성 하나의 값을 무작위로 섞었을 때 모델 성능(R²)이 얼마나 떨어지는지예요. 많이 떨어질수록 중요한 속성이에요.")

st.caption("학습용 분석이며, 실제 시설물의 안전 판단은 전문가의 시험과 평가가 필요해요.")
