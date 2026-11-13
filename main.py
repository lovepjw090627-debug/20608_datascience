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
