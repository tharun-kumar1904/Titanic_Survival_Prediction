import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
from src.data_preprocessing import preprocess

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# THEME STATE
# --------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# --------------------------------------------------
# LOAD MODEL & DATA
# --------------------------------------------------
@st.cache_resource
def load_model_data():
    model = load(os.path.join(PROJECT_ROOT, "models", "titanic_model.joblib"))
    feature_columns = load(os.path.join(PROJECT_ROOT, "models", "feature_columns.joblib"))
    data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "train.csv"))
    data["Age"] = data["Age"].fillna(data["Age"].median())
    return model, feature_columns, data

model, feature_columns, data = load_model_data()

# --------------------------------------------------
# THEME CONFIGURATION
# --------------------------------------------------
is_dark = st.session_state.theme == "dark"

if is_dark:
    bg_main = "#0f172a"
    bg_card = "#334155"
    text_primary = "#f1f5f9"
    text_secondary = "#cbd5e1"
    text_muted = "#94a3b8"
    border_color = "#475569"
    chart_template = "plotly_dark"
else:
    bg_main = "#f5f7fa"
    bg_card = "#ffffff"
    text_primary = "#0f172a"
    text_secondary = "#1f2933"
    text_muted = "#64748b"
    border_color = "#e2e8f0"
    chart_template = "plotly_white"

# --------------------------------------------------
# DYNAMIC CSS
# --------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * {{
        font-family: 'Inter', sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* reset any global opacity / blend on main containers */
    body, .stApp, .block-container {{
        opacity: 1 !important;
        mix-blend-mode: normal !important;
        filter: none !important;
    }}

    .stApp {{
        background: {bg_main};
    }}

    .main {{
        background: {bg_main};
    }}

    .main-header {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }}

    .main-header h1 {{
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }}

    .main-header p {{
        color: white;
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 500;
    }}

    /* METRIC CONTAINER CARD STYLE */
    div[data-testid="metric-container"] {{
        background: {bg_card} !important;
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid {border_color};
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
        filter: none !important;
    }}

    div[data-testid="metric-container"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    }}

    div[data-testid="metric-container"]:nth-child(1) {{
        border-left: 5px solid #6366f1;
    }}

    div[data-testid="metric-container"]:nth-child(2) {{
        border-left: 5px solid #8b5cf6;
    }}

    div[data-testid="metric-container"]:nth-child(3) {{
        border-left: 5px solid #ec4899;
    }}

    div[data-testid="metric-container"]:nth-child(4) {{
        border-left: 5px solid #f59e0b;
    }}

    div[data-testid="metric-container"]:nth-child(5) {{
        border-left: 5px solid #10b981;
    }}

    /* fully override metric internals */
    div[data-testid="stMetric"],
    div[data-testid="stMetric"] * {{
        opacity: 1 !important;
        mix-blend-mode: normal !important;
        filter: none !important;
    }}

    /* Override ALL text in metric containers */
    div[data-testid="metric-container"] p {{
        color: {text_secondary} !important;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
        filter: none !important;
    }}

    div[data-testid="metric-container"] span {{
        color: {text_primary} !important;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
        filter: none !important;
    }}

    /* Strongest possible override for any text/inline styles inside metric containers */
    div[data-testid="metric-container"],
    div[data-testid="metric-container"] *,
    div[data-testid="metric-container"] p,
    div[data-testid="metric-container"] span,
    div[data-testid="metric-container"] div {{
        color: {text_primary} !important;
        opacity: 1 !important;
        visibility: visible !important;
        mix-blend-mode: normal !important;
        filter: none !important;
        -webkit-text-fill-color: {text_primary} !important;
    }}

    /* Label text: force darker, slightly translucent for hierarchy */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {{
        color: {text_primary} !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 0.9 !important;
        mix-blend-mode: normal !important;
    }}

    /* Value text */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {{
        color: {text_primary} !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
    }}

    /* Delta text (if present) */
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] * {{
        color: {text_muted} !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        mix-blend-mode: normal !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {bg_card} !important;
        border-right: 3px solid {border_color};
    }}

    section[data-testid="stSidebar"] h3 {{
        color: {text_primary} !important;
        font-weight: 800;
    }}

    section[data-testid="stSidebar"] label {{
        color: {text_primary} !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }}

    section[data-testid="stSidebar"] p {{
        color: {text_secondary} !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] span[style*="rgba"] {{
        color: {text_primary} !important;
    }}

    /* Select/Dropdown styling */
    div[data-baseweb="select"] span {{
        color: {text_primary} !important;
    }}

    div[data-baseweb="select"] div {{
        color: {text_primary} !important;
    }}

    /* Radio button styling */
    div[data-baseweb="radio"] label {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    div[data-baseweb="radio"] span {{
        color: {text_primary} !important;
    }}

    /* Selectbox options visibility */
    div[data-baseweb="select"] > div > div {{
        color: {text_primary} !important;
    }}

    /* Dropdown menu items */
    ul[data-baseweb="menu"] li {{
        color: {text_primary} !important;
    }}

    ul[data-baseweb="menu"] li span {{
        color: {text_primary} !important;
    }}

    /* Radio button options */
    [role="radio"] + span {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    [role="radio"] ~ span {{
        color: {text_primary} !important;
    }}

    /* General text in form elements */
    div[data-baseweb="base-select"] [role="option"] {{
        color: {text_primary} !important;
    }}

    div[data-testid="stSelectbox"] div span {{
        color: {text_primary} !important;
    }}

    div[data-testid="stRadio"] div span {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background: {bg_card};
        padding: 1rem;
        border-radius: 16px;
        border: 2px solid {border_color};
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 700;
        background-color: {bg_main};
        color: {text_muted};
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }}

    h3, h4 {{
        color: {text_primary} !important;
        font-weight: 800;
    }}

    h3 {{
        font-size: 1.8rem;
    }}

    h4 {{
        font-size: 1.3rem;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }}

    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }}

    .stAlert {{
        border-radius: 12px;
        background: {bg_card} !important;
    }}

    div[data-baseweb="notification"] p {{
        color: {text_primary} !important;
        font-weight: 600;
    }}

    hr {{
        margin: 2.5rem 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, #ec4899, transparent);
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 16px;
        border: 2px solid {border_color};
    }}

    div[data-baseweb="select"] {{
        border: 2px solid {border_color} !important;
        background: {bg_card} !important;
    }}

    div[data-baseweb="select"] > div {{
        color: {text_primary} !important;
    }}

    div[data-baseweb="select"] span {{
        color: {text_primary} !important;
    }}

    div[data-baseweb="select"] input {{
        color: {text_primary} !important;
    }}

    /* Selectbox dropdown text */
    [data-baseweb="select"] [role="option"] {{
        color: {text_primary} !important;
        background-color: {bg_card} !important;
    }}

    [data-baseweb="select"] [role="option"]:hover {{
        background-color: {border_color} !important;
        color: {text_primary} !important;
    }}

    /* Selected value in selectbox */
    div[data-testid="stSelectbox"] {{
        color: {text_primary} !important;
    }}

    div[data-testid="stSelectbox"] div {{
        color: {text_primary} !important;
    }}

    div[data-testid="stSelectbox"] span {{
        color: {text_primary} !important;
    }}

    /* Override selectbox placeholder and value text */
    [data-baseweb="select"] > div > div > div {{
        color: {text_primary} !important;
    }}

    /* Make selectbox text visible - strongest override */
    div[role="combobox"] {{
        color: {text_primary} !important;
    }}

    div[role="combobox"] * {{
        color: {text_primary} !important;
    }}

    /* Selectbox input field text */
    input[type="text"][role="combobox"] {{
        color: {text_primary} !important;
    }}

    /* All text nodes in selectbox */
    [data-baseweb="select"] {{
        color: {text_primary} !important;
    }}

    [data-baseweb="select"] * {{
        color: {text_primary} !important;
    }}

    /* Force text color in select field */
    [data-baseweb="select"] [data-baseweb] {{
        color: {text_primary} !important;
    }}

    /* Selectbox value display */
    div[data-testid="stSelectbox"] [data-baseweb="select"] div {{
        color: {text_primary} !important;
        -webkit-text-fill-color: {text_primary} !important;
    }}

    /* Radio button styles */
    div[data-testid="stRadio"] {{
        color: {text_primary} !important;
    }}

    div[data-testid="stRadio"] label {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stRadio"] span {{
        color: {text_primary} !important;
    }}

    div[data-testid="stRadio"] [role="radio"] {{
        accent-color: #6366f1 !important;
    }}

    div[data-testid="stRadio"] [role="radio"] + label {{
        color: {text_primary} !important;
    }}

    div[data-baseweb="radio"] span {{
        color: {text_primary} !important;
    }}

    div[data-baseweb="radio"] label {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}

    div[data-baseweb="slider"] [role="slider"] {{
        background-color: #6366f1;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# THEME TOGGLE
# --------------------------------------------------
col_theme1, col_theme2, col_theme3 = st.columns([6, 1, 1])
with col_theme2:
    if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light", use_container_width=True):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🚢 Titanic Survival Intelligence</h1>
        <p>Advanced ML-powered analytics for predicting passenger survival probability</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 Passenger Filters")
    st.markdown("---")

    cls = st.multiselect(
        "🎫 Passenger Class",
        sorted(data["Pclass"].unique()),
        sorted(data["Pclass"].unique()),
        help="Filter by ticket class (1st, 2nd, or 3rd)",
    )

    gender = st.multiselect(
        "👤 Gender",
        sorted(data["Sex"].unique()),
        sorted(data["Sex"].unique()),
        help="Filter by passenger gender",
    )

    age_min, age_max = st.slider(
        "🎂 Age Range",
        int(data["Age"].min()),
        int(data["Age"].max()),
        (int(data["Age"].min()), int(data["Age"].max())),
        help="Filter passengers by age",
    )

    st.markdown("---")

    st.info(
        "**💡 Pro Tip:** Adjust filters to explore survival patterns across different passenger demographics!",
        icon="💡",
    )

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------
filtered = data.copy()

if cls:
    filtered = filtered[filtered["Pclass"].isin(cls)]
if gender:
    filtered = filtered[filtered["Sex"].isin(gender)]

filtered = filtered[(filtered["Age"] >= age_min) & (filtered["Age"] <= age_max)]

# --------------------------------------------------
# PREPROCESS & PREDICT
# --------------------------------------------------
if not filtered.empty:
    X_processed, _ = preprocess(
        filtered.drop(columns=["Survived"]),
        is_train=False,
        feature_columns=feature_columns,
    )
    X_model = X_processed.drop(columns=["PassengerId"])
    filtered["Survival Probability"] = model.predict_proba(X_model)[:, 1]
    filtered["Actual Survived"] = data.loc[filtered.index, "Survived"]

# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------
st.markdown("### 📊 Quick Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

total_passengers = len(filtered)
avg_age = round(filtered["Age"].mean(), 1) if not filtered.empty else 0
male_pct = f"{(filtered['Sex']=='male').mean()*100:.1f}" if not filtered.empty else "0"
female_pct = f"{(filtered['Sex']=='female').mean()*100:.1f}" if not filtered.empty else "0"
avg_survival = f"{filtered['Survival Probability'].mean()*100:.1f}" if not filtered.empty else "0"

with col1:
    st.markdown(
        f"""
        <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color};">
            <div style="color: {text_primary}; font-weight:700; font-size:0.95rem; opacity:1;">👥 Total Passengers</div>
            <div style="color: {text_primary}; font-weight:800; font-size:2.2rem; margin-top:0.4rem;">{total_passengers:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color};">
            <div style="color: {text_primary}; font-weight:700; font-size:0.95rem; opacity:1;">📅 Average Age</div>
            <div style="color: {text_primary}; font-weight:800; font-size:2.2rem; margin-top:0.4rem;">{avg_age} yrs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color};">
            <div style="color: {text_primary}; font-weight:700; font-size:0.95rem; opacity:1;">👨 Male</div>
            <div style="color: {text_primary}; font-weight:800; font-size:2.2rem; margin-top:0.4rem;">{male_pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color};">
            <div style="color: {text_primary}; font-weight:700; font-size:0.95rem; opacity:1;">👩 Female</div>
            <div style="color: {text_primary}; font-weight:800; font-size:2.2rem; margin-top:0.4rem;">{female_pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color};">
            <div style="color: {text_primary}; font-weight:700; font-size:0.95rem; opacity:1;">💚 Avg Survival</div>
            <div style="color: {text_primary}; font-weight:800; font-size:2.2rem; margin-top:0.4rem;">{avg_survival}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔮 Predictions", "💡 Insights", "🎯 Analysis"])

# ==============================
# TAB 1: OVERVIEW
# ==============================
with tab1:
    if filtered.empty:
        st.warning("⚠️ **No passengers match the selected filters.** Please adjust your criteria.", icon="⚠️")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Age Distribution")
            fig_age = px.histogram(
                filtered,
                x="Age",
                nbins=30,
                color_discrete_sequence=["#6366f1"],
                template=chart_template,
                labels={"Age": "Age (years)", "count": "Count"},
            )
            fig_age.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor=bg_card,
                paper_bgcolor=bg_card,
                font=dict(size=13, color=text_primary),
                xaxis=dict(
                    title=dict(text="<b>Age (years)</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=12, color=text_primary),
                    gridcolor=border_color,
                ),
                yaxis=dict(
                    title=dict(text="<b>Count</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=12, color=text_primary),
                    gridcolor=border_color,
                ),
            )
            fig_age.update_traces(marker_line_color="#4f46e5", marker_line_width=1.5)
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.markdown("#### 🎫 Class Distribution")
            class_counts = filtered["Pclass"].value_counts().sort_index()
            fig_class = px.pie(
                values=class_counts.values,
                names=[f"Class {i}" for i in class_counts.index],
                hole=0.45,
                color_discrete_sequence=["#6366f1", "#8b5cf6", "#ec4899"],
                template=chart_template,
            )
            fig_class.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(size=15, color="white"),
                marker=dict(line=dict(color="white", width=3)),
            )
            fig_class.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor=bg_card,
                font=dict(size=13, color=text_primary),
                legend=dict(font=dict(size=13, color=text_primary), bgcolor=bg_card),
            )
            st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 👥 Gender Distribution")
            gender_counts = filtered["Sex"].value_counts()
            fig_gender = go.Figure(
                data=[
                    go.Bar(
                        x=gender_counts.index,
                        y=gender_counts.values,
                        marker=dict(
                            color=["#6366f1", "#ec4899"],
                            line=dict(color=["#4f46e5", "#db2777"], width=2),
                        ),
                        text=gender_counts.values,
                        textposition="auto",
                        textfont=dict(size=20, color=text_primary, family="Inter"),
                    )
                ]
            )
            fig_gender.update_layout(
                showlegend=False,
                height=400,
                template=chart_template,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor=bg_card,
                paper_bgcolor=bg_card,
                font=dict(size=13, color=text_primary),
                xaxis=dict(
                    title=dict(text="<b>Gender</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=13, color=text_primary),
                    gridcolor=border_color,
                ),
                yaxis=dict(
                    title=dict(text="<b>Count</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=12, color=text_primary),
                    gridcolor=border_color,
                    range=[0, gender_counts.max() * 1.25],
                ),
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        with col4:
            st.markdown("#### ⚓ Actual Survival Rate")
            survival_rate = filtered["Actual Survived"].mean() * 100
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=survival_rate,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "<b>Survival Rate</b>", "font": {"size": 24, "color": text_primary}},
                    number={"suffix": "%", "font": {"size": 50, "color": text_primary}},
                    gauge={
                        "axis": {
                            "range": [None, 100],
                            "tickwidth": 2,
                            "tickcolor": text_muted,
                            "tickfont": {"color": text_primary, "size": 12},
                        },
                        "bar": {"color": "#10b981", "thickness": 0.8},
                        "bgcolor": bg_card,
                        "borderwidth": 3,
                        "bordercolor": border_color,
                        "steps": [
                            {"range": [0, 33], "color": "#fecaca"},
                            {"range": [33, 66], "color": "#fef08a"},
                            {"range": [66, 100], "color": "#bbf7d0"},
                        ],
                        "threshold": {
                            "line": {"color": "#6366f1", "width": 5},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor=bg_card,
                font=dict(color=text_primary),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================
# TAB 2: PREDICTIONS
# ==============================
with tab2:
    if filtered.empty:
        st.warning("⚠️ **No passengers match the selected filters.**", icon="⚠️")
    else:
        st.markdown("#### 🎯 Survival Probability Predictions")

        def get_risk_label(p):
            if p >= 0.7:
                return "🟢 High"
            elif p >= 0.4:
                return "🟡 Medium"
            else:
                return "🔴 Low"

        filtered["Survival Chance"] = filtered["Survival Probability"].apply(get_risk_label)

        display_df = (
            filtered[
                ["PassengerId", "Name", "Sex", "Age", "Pclass", "Survival Chance", "Survival Probability"]
            ]
            .sort_values("Survival Probability", ascending=False)
            .reset_index(drop=True)
        )

        st.dataframe(
            display_df.style.format({"Survival Probability": "{:.1%}", "Age": "{:.0f}"}).background_gradient(
                subset=["Survival Probability"], cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
            height=450,
        )

        st.markdown("---")
        st.markdown("#### 📊 Survival Category Breakdown")

        col1, col2, col3 = st.columns(3)

        high_risk = (filtered["Survival Probability"] >= 0.7).sum()
        med_risk = ((filtered["Survival Probability"] >= 0.4) & (filtered["Survival Probability"] < 0.7)).sum()
        low_risk = (filtered["Survival Probability"] < 0.4).sum()

        with col1:
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color}; text-align:center;">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.95rem;">🟢 High Survival (≥70%)</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.6rem; margin-top:0.4rem;">{high_risk} passengers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color}; text-align:center;">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.95rem;">🟡 Medium Survival (40-70%)</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.6rem; margin-top:0.4rem;">{med_risk} passengers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1.2rem; border-radius: 12px; border: 2px solid {border_color}; text-align:center;">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.95rem;">🔴 Low Survival (&lt;40%)</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.6rem; margin-top:0.4rem;">{low_risk} passengers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================
# TAB 3: INSIGHTS
# ==============================
with tab3:
    if filtered.empty:
        st.warning("⚠️ **No data available for insights.**", icon="⚠️")
    else:
        st.markdown("#### 🔍 Comparative Survival Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig_gender_box = px.box(
                filtered,
                x="Sex",
                y="Survival Probability",
                color="Sex",
                color_discrete_map={"male": "#6366f1", "female": "#ec4899"},
                template=chart_template,
                labels={"Survival Probability": "Survival Probability", "Sex": "Gender"},
            )
            fig_gender_box.update_layout(
                title=dict(text="<b>Survival by Gender</b>", font=dict(size=16, color=text_primary)),
                showlegend=False,
                height=450,
                plot_bgcolor=bg_card,
                paper_bgcolor=bg_card,
                font=dict(size=13, color=text_primary),
                xaxis=dict(
                    title=dict(text="<b>Gender</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=13, color=text_primary),
                    gridcolor=border_color,
                ),
                yaxis=dict(
                    title=dict(text="<b>Survival Probability</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=12, color=text_primary),
                    gridcolor=border_color,
                ),
            )
            fig_gender_box.update_traces(marker=dict(line=dict(width=2)))
            st.plotly_chart(fig_gender_box, use_container_width=True)

        with col2:
            fig_class_box = px.box(
                filtered,
                x="Pclass",
                y="Survival Probability",
                color="Pclass",
                color_discrete_sequence=["#6366f1", "#8b5cf6", "#ec4899"],
                template=chart_template,
                labels={"Survival Probability": "Survival Probability", "Pclass": "Class"},
            )
            fig_class_box.update_layout(
                title=dict(text="<b>Survival by Class</b>", font=dict(size=16, color=text_primary)),
                showlegend=False,
                height=450,
                plot_bgcolor=bg_card,
                paper_bgcolor=bg_card,
                font=dict(size=13, color=text_primary),
                xaxis=dict(
                    title=dict(text="<b>Passenger Class</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=13, color=text_primary),
                    gridcolor=border_color,
                ),
                yaxis=dict(
                    title=dict(text="<b>Survival Probability</b>", font=dict(size=14, color=text_primary)),
                    tickfont=dict(size=12, color=text_primary),
                    gridcolor=border_color,
                ),
            )
            fig_class_box.update_traces(marker=dict(line=dict(width=2)))
            st.plotly_chart(fig_class_box, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 🎯 Age vs Survival Probability")
        fig_scatter = px.scatter(
            filtered,
            x="Age",
            y="Survival Probability",
            color="Sex",
            size="Pclass",
            color_discrete_map={"male": "#6366f1", "female": "#ec4899"},
            template=chart_template,
            opacity=0.8,
            labels={"Age": "Age (years)", "Survival Probability": "Survival", "Sex": "Gender", "Pclass": "Class"},
        )
        fig_scatter.update_layout(
            height=450,
            plot_bgcolor=bg_card,
            paper_bgcolor=bg_card,
            font=dict(size=13, color=text_primary),
            xaxis=dict(
                title=dict(text="<b>Age (years)</b>", font=dict(size=14, color=text_primary)),
                tickfont=dict(size=12, color=text_primary),
                gridcolor=border_color,
            ),
            yaxis=dict(
                title=dict(text="<b>Survival Probability</b>", font=dict(size=14, color=text_primary)),
                tickfont=dict(size=12, color=text_primary),
                gridcolor=border_color,
            ),
            legend=dict(
                bgcolor=bg_card,
                bordercolor=border_color,
                borderwidth=2,
                font=dict(size=12, color=text_primary),
            ),
        )
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color="white")))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================
# TAB 4: ANALYSIS
# ==============================
with tab4:
    if filtered.empty:
        st.warning("⚠️ **No data available for analysis.**", icon="⚠️")
    else:
        st.markdown("#### 📋 Statistical Summary by Gender & Class")

        summary_stats = (
            filtered.groupby(["Sex", "Pclass"])
            .agg({"Survival Probability": ["mean", "min", "max", "count"]})
            .round(3)
        )
        summary_stats.columns = ["Mean", "Min", "Max", "Count"]

        st.dataframe(
            summary_stats.style.format("{:.3f}", subset=["Mean", "Min", "Max"]),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### 🎯 Key Findings")

        if "male" in filtered["Sex"].values and "female" in filtered["Sex"].values:
            avg_male_survival = filtered[filtered["Sex"] == "male"]["Survival Probability"].mean()
            avg_female_survival = filtered[filtered["Sex"] == "female"]["Survival Probability"].mean()

            col1, col2 = st.columns(2)

            with col1:
                female_bg = "#fdf2f8" if not is_dark else "#2d1b2e"
                female_title_color = "#831843" if not is_dark else "#f472b6"
                female_value_color = text_primary
                female_label_color = text_secondary
                
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, {female_bg} 0%, {female_bg} 100%);
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid #ec4899;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <h4 style='color: {female_title_color}; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>👩 Female Passengers</h4>
                        <p style='color: {female_value_color}; font-size: 1.8rem; font-weight: 800; margin: 0;'>{avg_female_survival:.1%}</p>
                        <p style='color: {female_label_color}; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Average Survival Rate</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                male_bg = "#eff6ff" if not is_dark else "#1e3a5f"
                male_title_color = "#1e3a8a" if not is_dark else "#93c5fd"
                male_value_color = text_primary
                male_label_color = text_secondary
                
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, {male_bg} 0%, {male_bg} 100%);
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid #6366f1;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <h4 style='color: {male_title_color}; margin: 0 0 0.5rem 0; font-size: 1.1rem;'>👨 Male Passengers</h4>
                        <p style='color: {male_value_color}; font-size: 1.8rem; font-weight: 800; margin: 0;'>{avg_male_survival:.1%}</p>
                        <p style='color: {male_label_color}; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Average Survival Rate</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### 🎫 Survival by Passenger Class")

        cols = st.columns(len(sorted(filtered["Pclass"].unique())))
        colors = ["#6366f1", "#8b5cf6", "#ec4899"]

        for idx, pclass in enumerate(sorted(filtered["Pclass"].unique())):
            avg_class_survival = filtered[filtered["Pclass"] == pclass]["Survival Probability"].mean()
            count_class = len(filtered[filtered["Pclass"] == pclass])

            with cols[idx]:
                class_bg = "#f8fafc" if not is_dark else "#1e293b"
                class_title_color = text_primary
                class_value_color = text_primary
                class_label_color = text_secondary
                
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, {class_bg} 0%, {class_bg} 100%);
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid {colors[idx]};
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;'>
                        <h4 style='color: {class_title_color}; margin: 0 0 0.5rem 0; font-size: 1rem;'>Class {pclass}</h4>
                        <p style='color: {class_value_color}; font-size: 2rem; font-weight: 800; margin: 0;'>{avg_class_survival:.1%}</p>
                        <p style='color: {class_label_color}; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>{count_class} passengers</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### 📊 Detailed Passenger Data with Sorting Options")

        # Sorting options with custom HTML and buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<p style='color: {text_primary}; font-weight: 700; margin: 0 0 0.8rem 0;'>🔄 Sort By</p>", unsafe_allow_html=True)
            sort_options = ["Survival Probability", "Age", "Name", "Passenger Class", "Gender"]
            
            # Create custom button group for sort by
            sort_cols = st.columns(len(sort_options))
            sort_by = "Survival Probability"  # default
            for idx, option in enumerate(sort_options):
                with sort_cols[idx]:
                    if st.button(option, key=f"sort_by_{idx}", use_container_width=True):
                        sort_by = option

        with col2:
            st.markdown(f"<p style='color: {text_primary}; font-weight: 700; margin: 0 0 0.8rem 0;'>📈 Sort Order</p>", unsafe_allow_html=True)
            order_cols = st.columns(2)
            sort_order = "Descending"  # default
            
            with order_cols[0]:
                if st.button("Descending ↓", key="sort_desc", use_container_width=True):
                    sort_order = "Descending"
            
            with order_cols[1]:
                if st.button("Ascending ↑", key="sort_asc", use_container_width=True):
                    sort_order = "Ascending"

        # Determine sort column
        sort_column_map = {
            "Survival Probability": "Survival Probability",
            "Age": "Age",
            "Name": "Name",
            "Passenger Class": "Pclass",
            "Gender": "Sex"
        }
        
        sort_column = sort_column_map[sort_by]
        ascending = "Ascending" in sort_order

        # Create display dataframe with sorting
        display_analysis_df = (
            filtered[
                ["PassengerId", "Name", "Sex", "Age", "Pclass", "Survival Probability", "Actual Survived"]
            ]
            .sort_values(sort_column, ascending=ascending)
            .reset_index(drop=True)
        )

        # Format and display
        st.dataframe(
            display_analysis_df.style.format({
                "Survival Probability": "{:.1%}",
                "Age": "{:.0f}",
                "Actual Survived": "{:.0f}"
            }).background_gradient(
                subset=["Survival Probability"], cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
            height=450,
        )

        # Summary statistics
        st.markdown("---")
        st.markdown("#### 📈 Quick Stats for Current View")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1rem; border-radius: 12px; border: 2px solid {border_color};">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.9rem;">📊 Total Shown</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.8rem;">{len(display_analysis_df)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            avg_surv = display_analysis_df["Survival Probability"].mean()
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1rem; border-radius: 12px; border: 2px solid {border_color};">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.9rem;">💚 Avg Survival</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.8rem;">{avg_surv:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            avg_age_view = display_analysis_df["Age"].mean()
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1rem; border-radius: 12px; border: 2px solid {border_color};">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.9rem;">📅 Avg Age</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.8rem;">{avg_age_view:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            actual_survivors = display_analysis_df["Actual Survived"].sum()
            st.markdown(
                f"""
                <div style="background: {bg_card}; padding: 1rem; border-radius: 12px; border: 2px solid {border_color};">
                    <div style="color: {text_primary}; font-weight:700; font-size:0.9rem;">✅ Survived</div>
                    <div style="color: {text_primary}; font-weight:800; font-size:1.8rem;">{int(actual_survivors)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0;'>
        <p style='font-size: 1rem; color: #64748b; font-weight: 600;'>
            🚢 <strong style='color: #6366f1;'>Titanic Survival Intelligence</strong>
        </p>
        <p style='font-size: 0.9rem; color: #94a3b8; margin-top: 0.5rem;'>
            Powered by Machine Learning & Advanced Analytics | Built with Streamlit & Plotly
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
