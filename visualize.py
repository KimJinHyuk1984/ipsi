import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

st.set_page_config(page_title="대학 입시결과 시각화 대시보드", layout="wide")

# 데이터 불러오기 및 컬럼 클린업
df = pd.read_excel("all_years_입시결과_통합.xlsx")
df = df.loc[:, ~df.columns.str.contains("Unnamed")]
df.columns = df.columns.str.strip()

# 사이드바: 대학명 → 중심전형 → 모집단위
대학s = sorted(df["대학명"].dropna().unique())
대학 = st.sidebar.selectbox("대학명 선택", 대학s)
중심전형s = sorted(df[df["대학명"] == 대학]["중심전형"].dropna().unique())
중심전형 = st.sidebar.selectbox("중심전형 선택", 중심전형s)
모집단위s = sorted(df[(df["대학명"] == 대학) & (df["중심전형"] == 중심전형)]["모집단위"].dropna().unique())
모집단위 = st.sidebar.selectbox("모집단위 선택", 모집단위s)

# 내신성적 입력 및 검증
내신성적 = None
내신_error = None
with st.sidebar:
    st.write("※ 내신 등급(평균)을 입력하세요 (0.0 ~ 9.9)")
    내신입력 = st.text_input("내신성적 (0~9.9)", value="", max_chars=4)
    if 내신입력:
        try:
            내신성적 = float(내신입력)
            if not (0 <= 내신성적 < 10):
                내신_error = True
        except ValueError:
            내신_error = True
        if 내신_error:
            내신성적 = None
            st.error("⚠️ 내신성적은 0.0 이상 9.9 미만의 숫자만 입력해야 합니다. 다시 입력하세요.")

# 데이터 필터
filtered = df[
    (df["대학명"] == 대학) &
    (df["중심전형"] == 중심전형) &
    (df["모집단위"] == 모집단위)
].copy()
filtered["연도"] = filtered["연도"].astype(int)
filtered = filtered.sort_values("연도")

def draw_plot(ycol, ylabel, yfmt=None, draw_line=False):
    st.subheader(f"{ylabel} 연도별 변화")

    x_full = [2021, 2022, 2023, 2024, 2025]
    base_cols = ["연도", "전형명"]
    if ycol != "모집인원":
        base_cols += [ycol, "모집인원"]
    else:
        base_cols += ["모집인원"]
    data = filtered[base_cols].dropna(subset=[ycol, "전형명"])
    data["연도"] = data["연도"].astype(int)

    # plotly/표준 색상군 많이 사용 (최대 20+색)
    color_list = (
        px.colors.qualitative.Plotly +
        px.colors.qualitative.Set1 +
        px.colors.qualitative.Pastel1 +
        px.colors.qualitative.Set2 +
        px.colors.qualitative.Dark2 +
        px.colors.qualitative.Set3
    )
    전형명_list = sorted(data["전형명"].unique())
    전형명_color_map = {v: color_list[i % len(color_list)] for i, v in enumerate(전형명_list)}

    fig = go.Figure()

    for 전형명 in 전형명_list:
        sub = data[data["전형명"] == 전형명].copy()
        color = 전형명_color_map[전형명]
        def safe_int(x):
            try: return str(int(x))
            except: return "정보없음"
        def safe_float(x):
            try: return str(round(float(x), 2))
            except: return "정보없음"
        value_text = sub[ycol].apply(safe_float if ycol != "모집인원" else safe_int)
        if ycol != "모집인원":
            모집인원_text = sub["모집인원"].apply(safe_int)
            customdata = pd.concat([
                sub["전형명"].reset_index(drop=True),
                모집인원_text.reset_index(drop=True)
            ], axis=1).values
            hovertemplate = (
                "연도: %{x}<br>"
                + ylabel + ": %{y}<br>"
                "세부전형명: %{customdata[0]}<br>"
                "모집인원: %{customdata[1]}<extra></extra>"
            )
        else:
            customdata = sub[["전형명"]].reset_index(drop=True).values
            hovertemplate = (
                "연도: %{x}<br>"
                + ylabel + ": %{y}<br>"
                "세부전형명: %{customdata[0]}<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=sub["연도"],
            y=sub[ycol],
            mode='markers+text',
            name=f"{전형명}",
            marker=dict(size=15, color=color, opacity=0.85, line=dict(width=1, color="white")),
            text=value_text,
            textposition='top center',
            showlegend=True,
            customdata=customdata,
            hovertemplate=hovertemplate
        ))

    if draw_line and (내신성적 is not None):
        fig.add_shape(
            type="line",
            x0=min(x_full)-0.5, x1=max(x_full)+0.5,
            y0=내신성적, y1=내신성적,
            line=dict(color="red", width=3, dash="dash"),
            xref='x', yref='y'
        )

    fig.update_layout(
        xaxis=dict(
            title="연도",
            tickmode='array',
            tickvals=x_full,
            ticktext=[str(year) for year in x_full]
        ),
        yaxis_title=ylabel,
        height=400,  # plot 크기를 50%로 축소!
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="세부 전형명",
        legend=dict(font=dict(size=11))
    )
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    st.plotly_chart(fig, use_container_width=True)

# ------------------- (연도별 라인/산점 그래프)
draw_plot("모집인원", "모집인원")
draw_plot("경쟁률", "경쟁률")
draw_plot("충원순위", "충원순위")
draw_plot("교과 50% cut", "교과 50% cut", draw_line=True)
draw_plot("교과 70% cut", "교과 70% cut", draw_line=True)

# --- 2026 예측 cut 그래프 (상한/하한 포함 세로선)
pred_df = pd.read_excel("2026 수시입결 예측_XGBoost.xlsx")
pred_df.columns = pred_df.columns.str.strip()

pred_filtered = pred_df[
    (pred_df["대학명"] == 대학) &
    (pred_df["중심전형"] == 중심전형) &
    (pred_df["모집단위"] == 모집단위)
]

if not pred_filtered.empty:
    x_cuts = ["50% cut", "70% cut"]

    y_pred = [
        pred_filtered["2026_교과50cut_예측"].values[0],
        pred_filtered["2026_교과70cut_예측"].values[0]
    ]
    y_lwr = [
        pred_filtered["2026_교과50cut_신뢰구간하한"].values[0],
        pred_filtered["2026_교과70cut_신뢰구간하한"].values[0]
    ]
    y_upr = [
        pred_filtered["2026_교과50cut_신뢰구간상한"].values[0],
        pred_filtered["2026_교과70cut_신뢰구간상한"].values[0]
    ]

    fig_vert = go.Figure()

    for i, label in enumerate(x_cuts):
        # 상한~하한 세로선 (카테고리명 두 번 사용)
        fig_vert.add_trace(go.Scatter(
            x=[label, label],
            y=[y_lwr[i], y_upr[i]],
            mode='lines',
            line=dict(color='dodgerblue', width=13),
            showlegend=False,
            hovertemplate=f"{label}<br>신뢰구간: {y_lwr[i]:.2f} ~ {y_upr[i]:.2f}<extra></extra>"
        ))
        # 예측값 마커 (큰 원)
        fig_vert.add_trace(go.Scatter(
            x=[label], y=[y_pred[i]],
            mode='markers+text',
            marker=dict(color='dodgerblue', size=15, symbol='circle'),
            text=[f"{y_pred[i]:.2f}"],
            textfont=dict(size=16),
            textposition="top center",
            showlegend=False,
            hovertemplate=f"{label}<br>예측값: {y_pred[i]:.2f}<extra></extra>"
        ))

    # 내신 입력값 가로선 (전체 cut에 걸쳐)
    if 내신성적 is not None:
        fig_vert.add_shape(
            type="line",
            x0=-0.5, x1=1.5,
            y0=내신성적, y1=내신성적,
            line=dict(color="red", width=7, dash="dash"),
            xref='x', yref='y'
        )
        fig_vert.add_trace(go.Scatter(
            x=x_cuts, y=[내신성적, 내신성적],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig_vert.update_layout(
        title="2026 예측 cut별 신뢰구간(by XGBoost)",
        xaxis=dict(
            title="컷 종류",
            type="category",
            tickvals=x_cuts,
            ticktext=x_cuts,
            categoryorder="array",
            categoryarray=x_cuts
        ),
        yaxis=dict(title="등급", rangemode='tozero'),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )

    st.plotly_chart(fig_vert, use_container_width=True)
else:
    st.warning("해당 조건의 2026 예측 데이터가 없습니다.")
