import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# 콘크리트 강도 및 안전성 탐색 앱
# 실행: streamlit run main.py
# ---------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/lovepjw090627-debug/data/refs/heads/main/concrete_clean.csv"

st.set_page_config(
    page_title="콘크리트 강도 안전 분석",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏗️ 콘크리트 강도가 약해서 생기는 안전문제")
st.caption("측정 데이터 기반 강도 비교 · 민감도 분석 · 안전 개선 방향 탐색")

st.warning(
    "교육·탐색용 분석 앱입니다. 반발경도 환산값만으로 실제 구조물의 안전 여부나 사용 가능 여부를 판정할 수 없습니다. "
    "실제 시설물의 판단과 보강은 현장 조사, 공인 시험 및 구조기술 전문가의 검토가 필요합니다."
)

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8-sig")

try:
    raw = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결 또는 CSV 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()

# 데이터 열 이름
TARGET = "반발경도추정_재령보정강도평균"
DESIGN = "설계기준강도(Mpa)"
FACILITY = "시설물구분"
TYPE = "시설물종류"

required = [TARGET, DESIGN]
missing_required = [c for c in required if c not in raw.columns]
if missing_required:
    st.error(f"필수 열을 찾을 수 없습니다: {missing_required}")
    st.write("현재 CSV 열:", list(raw.columns))
    st.stop()

df = raw.copy()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for c in df.columns:
    if c not in numeric_cols:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() > 0 and converted.notna().sum() >= len(df) * 0.7:
            df[c] = converted
            numeric_cols.append(c)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df[DESIGN] = pd.to_numeric(df[DESIGN], errors="coerce")
df["강도차이(MPa)"] = df[TARGET] - df[DESIGN]
df["기준강도 대비 비율(%)"] = np.where(
    df[DESIGN] > 0, df[TARGET] / df[DESIGN] * 100, np.nan
)
df["탐색상 확인 필요"] = np.where(
    df["강도차이(MPa)"] < 0, "추정값이 설계기준 미만", "추정값이 설계기준 이상"
)

# 숫자형 데이터에서 비정상적으로 큰 값이 분석을 왜곡하지 않도록
# 시각화/모델용 데이터만 별도로 IQR 범위 기반 점검한다.
def iqr_filter(frame, columns):
    out = frame.copy()
    mask = pd.Series(True, index=out.index)
    for col in columns:
        s = pd.to_numeric(out[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 8:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            mask &= s.between(q1 - 3 * iqr, q3 + 3 * iqr) | s.isna()
    return out.loc[mask].copy()

st.sidebar.title("📌 분석 메뉴")
page = st.sidebar.radio(
    "페이지 선택",
    ["데이터 현황", "강도 분석", "민감도 분석", "안전성 진단", "안전 개선 방안"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"전체 측정 데이터: **{len(df):,}건**")
st.sidebar.caption("강도 단위는 원본 데이터의 MPa 표기를 따릅니다.")

valid_strength = df.dropna(subset=[TARGET, DESIGN]).copy()
if len(valid_strength) == 0:
    st.error("강도 분석에 사용할 수 있는 데이터가 없습니다.")
    st.stop()

# ---------------------- 데이터 현황 ----------------------
if page == "데이터 현황":
    st.subheader("📊 데이터 한눈에 보기")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 측정 건수", f"{len(df):,}건")
    c2.metric("데이터 열 수", f"{df.shape[1]}개")
    c3.metric("시설물 종류 수", f"{df[TYPE].nunique():,}종" if TYPE in df else "-")
    c4.metric("강도 정보가 있는 행", f"{len(valid_strength):,}건")

    c5, c6, c7 = st.columns(3)
    c5.metric("평균 설계기준강도", f"{valid_strength[DESIGN].mean():.2f} MPa")
    c6.metric("평균 추정 재령보정강도", f"{valid_strength[TARGET].mean():.2f} MPa")
    below = (valid_strength[TARGET] < valid_strength[DESIGN]).sum()
    c7.metric("추정값이 기준 미만인 행", f"{below:,}건")

    st.markdown("#### 데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### 열별 데이터 정보")
    info = pd.DataFrame({
        "열 이름": df.columns,
        "자료형": [str(df[c].dtype) for c in df.columns],
        "결측치 수": [int(df[c].isna().sum()) for c in df.columns],
        "고유값 수": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

    st.markdown("#### 결측치 현황")
    miss = df.isna().sum().reset_index()
    miss.columns = ["열 이름", "결측치 수"]
    miss = miss[miss["결측치 수"] > 0].sort_values("결측치 수", ascending=False)
    if len(miss):
        st.dataframe(miss, use_container_width=True, hide_index=True)
    else:
        st.success("결측치가 없습니다.")

# ---------------------- 강도 분석 ----------------------
elif page == "강도 분석":
    st.subheader("📈 설계기준강도와 추정강도 비교")
    st.caption("추정강도 = 반발경도추정_재령보정강도평균 열을 사용합니다.")
    a, b = st.columns(2)
    with a:
        st.metric("평균 설계기준강도", f"{valid_strength[DESIGN].mean():.2f} MPa")
    with b:
        st.metric("평균 추정 재령보정강도", f"{valid_strength[TARGET].mean():.2f} MPa")

    plot_df = valid_strength.copy()
    if FACILITY in plot_df.columns:
        options = ["전체"] + sorted(plot_df[FACILITY].dropna().astype(str).unique().tolist())
        selected = st.selectbox("시설물 구분", options)
        if selected != "전체":
            plot_df = plot_df[plot_df[FACILITY].astype(str) == selected]

    fig = px.scatter(
        plot_df, x=DESIGN, y=TARGET,
        color=FACILITY if FACILITY in plot_df.columns else None,
        hover_data=[c for c in [TYPE, "강도차이(MPa)", "기준강도 대비 비율(%)"] if c in plot_df],
        title="설계기준강도와 추정 재령보정강도",
        labels={DESIGN: "설계기준강도 (MPa)", TARGET: "추정 재령보정강도 (MPa)"},
    )
    lo = float(min(plot_df[DESIGN].min(), plot_df[TARGET].min()))
    hi = float(max(plot_df[DESIGN].max(), plot_df[TARGET].max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dash", color="red"))
    fig.add_annotation(x=hi, y=lo, text="점선 아래: 추정값이 설계기준보다 낮음",
                       showarrow=False, xanchor="right")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 시설물별 평균 강도 비교")
    if FACILITY in valid_strength.columns:
        group = valid_strength.groupby(FACILITY, dropna=False).agg(
            측정건수=(TARGET, "count"),
            평균설계기준강도=(DESIGN, "mean"),
            평균추정강도=(TARGET, "mean"),
        ).reset_index()
        group["평균강도차이"] = group["평균추정강도"] - group["평균설계기준강도"]
        st.dataframe(group.round(2), use_container_width=True, hide_index=True)
        melt = group.melt(id_vars=FACILITY,
                          value_vars=["평균설계기준강도", "평균추정강도"],
                          var_name="구분", value_name="강도(MPa)")
        fig2 = px.bar(melt, x=FACILITY, y="강도(MPa)", color="구분",
                      barmode="group", title="시설물 구분별 평균 강도")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------- 민감도 분석 ----------------------
elif page == "민감도 분석":
    st.subheader("🔬 민감도 분석")
    st.write(
        "어떤 측정 변수가 추정 재령보정강도와 함께 변화하는지 통계적으로 살펴봅니다. "
        "상관관계는 원인·결과를 증명하지 않으며, 변수 중요도도 실제 보강 효과를 뜻하지 않습니다."
    )

    candidate_features = [
        "반발경도(R0)", "측정치평균", "재령보정계수(an)",
        "타격각도", "습윤상태보정계수", "측정횟수", DESIGN
    ]
    features = [c for c in candidate_features if c in df.columns]
    if len(features) < 2:
        st.warning("민감도 분석에 필요한 숫자형 측정 변수가 충분하지 않습니다.")
    else:
        sens = df[features + [TARGET]].copy()
        for c in sens.columns:
            sens[c] = pd.to_numeric(sens[c], errors="coerce")
        sens = sens.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
        # 극단값 점검은 모델 안정성을 위한 탐색용 처리이며 원본 데이터는 변경하지 않음
        model_df = iqr_filter(sens, features + [TARGET])
        st.caption(f"민감도 분석에 사용 가능한 행: {len(model_df):,}건 / 강도값이 있는 행: {len(sens):,}건")

        corr = sens[features + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
        corr_df = corr.rename("스피어만 상관계수").reset_index().rename(columns={"index": "변수"})
        corr_df["절댓값"] = corr_df["스피어만 상관계수"].abs()
        corr_df = corr_df.sort_values("절댓값", ascending=False)

        st.markdown("#### 1. 상관관계 기반 민감도")
        st.caption("값이 +1에 가까우면 함께 증가하는 경향, -1에 가까우면 반대로 변화하는 경향입니다.")
        fig_corr = px.bar(
            corr_df.sort_values("스피어만 상관계수"),
            x="스피어만 상관계수", y="변수", orientation="h",
            color="스피어만 상관계수", color_continuous_scale="RdBu",
            range_color=[-1, 1], title="측정 변수와 추정강도의 상관관계"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.dataframe(corr_df[["변수", "스피어만 상관계수"]].round(3),
                     use_container_width=True, hide_index=True)

        st.markdown("#### 2. 머신러닝 변수 중요도")
        st.caption("랜덤 포레스트가 데이터의 추정강도 패턴을 학습한 뒤, 각 변수를 섞었을 때 예측 성능이 얼마나 달라지는지 평가합니다.")
        if len(model_df) >= 30:
            X = model_df[features]
            y = model_df[TARGET]
            if y.nunique() > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("rf", RandomForestRegressor(
                        n_estimators=250, random_state=42,
                        min_samples_leaf=2, n_jobs=-1
                    ))
                ])
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                perm = permutation_importance(
                    model, X_test, y_test, n_repeats=8,
                    random_state=42, scoring="r2"
                )
                imp = pd.DataFrame({
                    "변수": features,
                    "중요도": perm.importances_mean,
                    "표준편차": perm.importances_std
                }).sort_values("중요도", ascending=True)
                st.metric("검증 데이터 R²", f"{score:.3f}")
                st.plotly_chart(
                    px.bar(imp, x="중요도", y="변수", orientation="h",
                           error_x="표준편차", title="Permutation 중요도"),
                    use_container_width=True
                )
                st.dataframe(imp.sort_values("중요도", ascending=False).round(4),
                             use_container_width=True, hide_index=True)
                st.caption("R²가 낮거나 음수이면 모델 설명력이 부족할 수 있습니다. 중요도가 음수인 변수는 예측에 도움이 된다고 보기 어렵습니다.")
            else:
                st.info("추정강도 값이 모두 같아 모델을 학습할 수 없습니다.")
        else:
            st.info("머신러닝 민감도 분석에는 최소 30개 이상의 유효 행이 필요합니다.")

        st.markdown("#### 3. 변수별 분포 확인")
        chosen = st.selectbox("살펴볼 변수", features)
        chart_df = sens[[chosen, TARGET]].dropna()
        fig_sc = px.scatter(chart_df, x=chosen, y=TARGET, trendline="ols",
                            opacity=0.55,
                            labels={TARGET: "추정 재령보정강도 (MPa)"},
                            title=f"{chosen}와 추정강도의 관계")
        st.plotly_chart(fig_sc, use_container_width=True)

# ---------------------- 안전성 진단 ----------------------
elif page == "안전성 진단":
    st.subheader("🚦 추정값 기준 확인")
    st.write(
        "아래 구분은 데이터 탐색을 위한 단순 비교입니다. "
        "설계기준 미만이라는 이유만으로 곧바로 위험 또는 사용 불가라고 확정할 수는 없습니다."
    )
    below_df = valid_strength[valid_strength[TARGET] < valid_strength[DESIGN]].copy()
    above_df = valid_strength[valid_strength[TARGET] >= valid_strength[DESIGN]].copy()
    x1, x2, x3 = st.columns(3)
    x1.metric("전체 유효 측정", f"{len(valid_strength):,}건")
    x2.metric("추정값이 기준 미만", f"{len(below_df):,}건")
    x3.metric("기준 미만 비율", f"{len(below_df) / len(valid_strength) * 100:.1f}%")

    st.markdown("#### 추정강도와 설계기준의 차이")
    fig = px.histogram(valid_strength, x="강도차이(MPa)", nbins=35,
                       title="추정강도 - 설계기준강도 분포",
                       labels={"강도차이(MPa)": "강도 차이 (MPa)"})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 추가 확인이 필요한 측정 행")
    show_cols = [c for c in [
        FACILITY, TYPE, DESIGN, "반발경도(R0)", "측정치평균",
        TARGET, "강도차이(MPa)", "기준강도 대비 비율(%)"
    ] if c in below_df.columns]
    st.dataframe(
        below_df.sort_values("강도차이(MPa)")[show_cols],
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "확인 필요 데이터 CSV 다운로드",
        data=below_df[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="concrete_check_required.csv",
        mime="text/csv"
    )

    st.markdown("#### 데이터 품질 점검")
    quality_cols = [c for c in [
        "재령보정계수(an)", "반발경도(R0)", "측정치평균",
        "반발경도추정_환산강도평균", TARGET
    ] if c in df.columns]
    if quality_cols:
        qrows = []
        for col in quality_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            qrows.append({
                "변수": col,
                "결측": int(s.isna().sum()),
                "최솟값": s.min(),
                "중앙값": s.median(),
                "최댓값": s.max()
            })
        st.dataframe(pd.DataFrame(qrows).round(3), use_container_width=True, hide_index=True)
        st.caption("비정상적으로 큰 값이나 입력 오류가 의심되는 값은 원자료·측정 기록과 대조해야 합니다.")

# ---------------------- 안전 개선 방안 ----------------------
elif page == "안전 개선 방안":
    st.subheader("🛠️ 안전성을 높이기 위한 개선 방향")
    st.write("데이터에서 확인된 신호를 실제 안전관리 활동으로 연결하는 일반적인 점검 방향입니다.")

    below_rate = (valid_strength[TARGET] < valid_strength[DESIGN]).mean() * 100
    avg_gap = valid_strength["강도차이(MPa)"].mean()
    st.metric("설계기준 대비 평균 강도 차이", f"{avg_gap:+.2f} MPa")
    st.metric("추정값이 설계기준 미만인 행의 비율", f"{below_rate:.1f}%")

    st.markdown("#### 데이터에서 확인할 사항")
    st.markdown(
        "- 설계기준보다 추정강도가 낮게 나타난 측정 위치를 우선적으로 재확인합니다.\n"
        "- 같은 시설물 안에서 측정값의 편차가 큰 구간이 있는지 살펴봅니다.\n"
        "- 반발경도, 타격각도, 습윤상태 등 측정 조건이 기록되었는지 확인합니다.\n"
        "- 재령보정계수 등 계산에 사용된 값의 단위와 입력 오류 가능성을 점검합니다."
    )

    st.markdown("#### 권장되는 안전관리 절차")
    steps = [
        ("1. 현장 재점검", "균열, 박리, 누수, 철근 부식 흔적 등 외관 상태를 전문 인력이 확인합니다."),
        ("2. 측정 신뢰도 확인", "시험기 교정 상태, 표면 상태, 수분 상태, 타격 방향과 측정 위치를 확인하고 필요하면 재측정합니다."),
        ("3. 추가 시험", "필요한 경우 코어 압축강도 시험 등 직접적인 시험을 포함해 결과를 교차 확인합니다."),
        ("4. 구조 검토", "확인된 재료 강도와 구조물의 하중·배근·손상 상태를 종합해 구조기술 전문가가 안전성을 평가합니다."),
        ("5. 보수·보강 및 모니터링", "전문가가 설계한 보수·보강안을 적용하고 이후 균열·변형·누수 등을 정기적으로 관찰합니다."),
    ]
    for title, desc in steps:
        with st.expander(title, expanded=True):
            st.write(desc)

    st.info(
        "중요: 콘크리트에 물을 더 넣거나 임의로 재료 배합을 바꾸는 방식은 안전 개선책이 아닙니다. "
        "이미 시공된 구조물은 임의 보강하지 말고 전문 진단과 설계 절차를 따라야 합니다."
    )

    st.markdown("#### 참고 자료")
    st.markdown(
        "- [FHWA: Guide to Nondestructive Testing of Concrete](https://www.fhwa.dot.gov/pavement/pubs/006641.pdf)\n"
        "- [FEMA: Evaluation of Earthquake-Damaged Concrete and Masonry Wall Buildings](https://www.govinfo.gov/content/pkg/GOVPUB-FEM1-PURL-LPS117718/pdf/GOVPUB-FEM1-PURL-LPS117718.pdf)"
    )
    st.caption(
        "반발경도법은 콘크리트 표면의 경도와 균질성을 비교하고 강도를 추정하는 보조 방법입니다. "
        "실제 압축강도를 단독으로 확정하는 시험으로 사용해서는 안 됩니다."
    )

st.markdown("---")
st.caption("분석 데이터: concrete_clean.csv | 추정 강도와 설계기준의 단순 비교는 구조 안전 판정이 아닙니다.")
