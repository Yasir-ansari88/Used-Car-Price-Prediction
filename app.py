import io
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="VahanValue | Car Valuation Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = "artifacts"

COLORS = {
    "bg": "#F4F5F1",
    "surface": "#FFFFFF",
    "surface_alt": "#ECEEE8",
    "border": "#DCE1D9",
    "ink": "#14201E",
    "ink_muted": "#5B6B65",
    "primary": "#123832",
    "primary_dark": "#0A2420",
    "accent": "#E2A33B",
    "accent_soft": "#F7E4BB",
    "secondary": "#2F6F6B",
    "danger": "#A6461F",
}
CHART_PALETTE = [COLORS["primary"], COLORS["accent"], COLORS["secondary"], COLORS["danger"], "#7C8B5B"]


def style_fig(fig, height=340):
    fig.update_layout(
        font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["ink"], size=12),
        title_font=dict(family="Space Grotesk, sans-serif", size=16, color=COLORS["primary"]),
        plot_bgcolor=COLORS["surface"],
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=48, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=CHART_PALETTE,
    )
    fig.update_xaxes(gridcolor=COLORS["surface_alt"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["surface_alt"], zerolinecolor=COLORS["border"])
    return fig


GAUGE_MARK_SVG = (
    '<svg width="{size}" height="{size}" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="17" cy="17" r="15" stroke="{ring}" stroke-width="2.2" opacity="0.35"/>'
    '<circle cx="17" cy="17" r="15" stroke="{accent}" stroke-width="2.6" stroke-dasharray="62 100" stroke-linecap="round" transform="rotate(-208 17 17)"/>'
    '<line x1="17" y1="17" x2="23.5" y2="10.5" stroke="{needle}" stroke-width="2.4" stroke-linecap="round"/>'
    '<circle cx="17" cy="17" r="2.6" fill="{needle}"/>'
    "</svg>"
)


def gauge_mark(size=30, ring="#F4F5F1", accent="#E2A33B", needle="#F4F5F1"):
    return GAUGE_MARK_SVG.format(size=size, ring=ring, accent=accent, needle=needle)


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{ background: {COLORS["bg"]}; }}
    h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; color: {COLORS["ink"]}; }}

    

    .app-header {{
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
        border-radius: 14px; padding: 22px 28px; margin-bottom: 22px;
    }}
    .app-header-left {{ display: flex; align-items: center; gap: 14px; }}
    .app-brand {{ font-size: 22px; font-weight: 700; color: #F4F5F1; line-height: 1.1; }}
    .app-tagline {{ font-size: 12.5px; color: #C9D3CC; letter-spacing: .03em; margin-top: 2px; }}
    .app-badge {{
        font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: {COLORS["primary_dark"]};
        background: {COLORS["accent"]}; padding: 6px 12px; border-radius: 20px; font-weight: 600;
        letter-spacing: .02em;
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase;
        letter-spacing: .12em; color: {COLORS["secondary"]}; font-weight: 600; margin-bottom: 4px;
    }}

    .kpi-card {{
        background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px;
        padding: 16px 18px; height: 100%;
    }}
    .kpi-label {{
        font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .08em;
        font-size: 10.5px; color: {COLORS["ink_muted"]}; font-weight: 600;
    }}
    .kpi-value {{
        font-family: 'Space Grotesk', sans-serif; font-size: 25px; font-weight: 700;
        color: {COLORS["ink"]}; margin-top: 4px; line-height: 1.15;
    }}
    .kpi-sub {{ font-size: 11.5px; color: {COLORS["ink_muted"]}; margin-top: 3px; }}

    .result-hero {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
        border-radius: 16px; padding: 28px 32px; text-align: left;
    }}
    .result-hero-label {{
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-transform: uppercase;
        letter-spacing: .1em; color: {COLORS["accent"]}; font-weight: 600;
    }}
    .result-hero-value {{
        font-family: 'Space Grotesk', sans-serif; font-size: 44px; font-weight: 700; color: #FFFFFF;
        margin-top: 6px;
    }}
    .result-hero-sub {{ font-size: 12.5px; color: #C9D3CC; margin-top: 8px; }}

    section[data-testid="stSidebar"] {{ background: {COLORS["primary"]}; }}
    section[data-testid="stSidebar"] * {{ color: #EDF1EC !important; }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}
    /* Selectbox's own closed-state value and inline <code> badges keep their
       native light background — force dark text there so it isn't
       white-on-white / white-on-light-grey (unreadable). Streamlit 1.60's
       selectbox renders as a react-aria <input> element, so target that
       directly (older [data-baseweb="select"] markup no longer exists). */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] input {{
        color: {COLORS["ink"]} !important;
        background: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-rac] {{
        background: #FFFFFF !important;
        border-color: {COLORS["border"]} !important;
    }}
    section[data-testid="stSidebar"] code {{
        color: {COLORS["ink"]} !important;
        background: {COLORS["surface_alt"]};
        padding: 1px 6px;
        border-radius: 4px;
    }}
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{ margin-bottom: 2px; }}
    /* Navigation radio (sidebar) — bigger touch targets, clear active item */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
        padding: 9px 10px; border-radius: 8px; margin-bottom: 2px; width: 100%;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
        background: rgba(226,163,59,0.16);
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] p {{
        font-size: 14.5px; font-weight: 500;
    }}

    .stButton > button, .stDownloadButton > button {{
        background: {COLORS["primary"]}; color: #F4F5F1; border: none; border-radius: 8px;
        font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; padding: 0.55rem 1.1rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {COLORS["accent"]}; color: {COLORS["primary_dark"]};
    }}

    .info-card {{
        background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px;
        padding: 18px 20px; height: 100%;
    }}
    .info-step {{
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
        color: {COLORS["primary_dark"]}; background: {COLORS["accent"]}; border-radius: 20px;
        padding: 2px 10px; margin-bottom: 8px; letter-spacing: .06em;
    }}
    .info-title {{
        font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700;
        color: {COLORS["ink"]}; margin-bottom: 6px;
    }}
    .info-body {{ font-size: 13.5px; color: {COLORS["ink_muted"]}; line-height: 1.55; }}
    .about-hero {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
        border-radius: 16px; padding: 28px 32px;
    }}
    .about-hero-title {{
        font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: #FFFFFF;
    }}
    .about-hero-body {{ font-size: 14.5px; color: #C9D3CC; margin-top: 8px; line-height: 1.6; max-width: 900px; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label, value, sub=""):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def section_eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def info_card(title, body, step=None, min_height=0):
    badge = f'<div class="info-step">{step}</div>' if step else ""
    style = f' style="min-height:{min_height}px;"' if min_height else ""
    st.markdown(
        f'<div class="info-card"{style}>{badge}<div class="info-title">{title}</div>'
        f'<div class="info-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    models = joblib.load(f"{ARTIFACT_DIR}/trained_models.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
    encoders = joblib.load(f"{ARTIFACT_DIR}/encoders.joblib")
    with open(f"{ARTIFACT_DIR}/meta.json") as f:
        meta = json.load(f)
    results_table = pd.read_json(f"{ARTIFACT_DIR}/results_table.json", orient="index")
    clean_data = pd.read_parquet(f"{ARTIFACT_DIR}/clean_data.parquet")
    return models, scaler, encoders, meta, results_table, clean_data


models, scaler, encoders, meta, results_table, clean_data = load_artifacts()
FEATURE_COLUMNS = meta["feature_column"]
BEST_MODEL_NAME = meta["best_model_name"]
TWO_WORD_BRANDS = ["Land Rover", "Alfa Romeo", "Aston Martin"]

USD_TO_INR = 83.0


def format_indian(n: float):
    n = int(round(n))
    sign = "-" if n < 0 else ""
    n = abs(n)
    s = str(n)
    if len(s) <= 3:
        return sign + s
    last3, rest = s[-3:], s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return sign + ",".join(parts) + "," + last3


def fmt_num(x: float):
    if st.session_state.get("currency", "₹ INR") == "₹ INR":
        return format_indian(x)
    return f"{x:,.0f}"


def money(amount_usd: float):
    if st.session_state.get("currency", "₹ INR") == "₹ INR":
        return f"₹{format_indian(amount_usd * USD_TO_INR)}"
    return f"${amount_usd:,.0f}"


def to_display_currency(series_usd: pd.Series):
    if st.session_state.get("currency", "₹ INR") == "₹ INR":
        return series_usd * USD_TO_INR
    return series_usd


def extract_brand_name(name: str):
    name = str(name)
    for b in TWO_WORD_BRANDS:
        if name.startswith(b):
            return b
    return name.split()[0]


def clean_raw_dataframe(data: pd.DataFrame):
    df = data.copy()

    df["miles"] = (
        df["miles"].astype(str).str.replace(",", "", regex=False).str.replace("miles", "", regex=False).str.strip()
    )
    df["miles"] = pd.to_numeric(df["miles"], errors="coerce")

    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    color_split = df["color"].astype(str).str.split(",", expand=True)
    df["exterior_color"] = color_split[0].str.replace("exterior", "", regex=False).str.strip()
    if 1 in color_split.columns:
        df["interior_color"] = color_split[1].str.replace("interior", "", regex=False).str.strip()
    else:
        df["interior_color"] = "Unknown"
    df["exterior_color"] = df["exterior_color"].fillna("Unknown")
    df["interior_color"] = df["interior_color"].fillna("Unknown")

    df["accidents"] = df["condition"].astype(str).str.extract(r"(\d+|No) accident")[0]
    df["accidents"] = df["accidents"].replace("No", "0")
    df["accidents"] = pd.to_numeric(df["accidents"], errors="coerce").fillna(0).astype(int)

    df["owners"] = df["condition"].astype(str).str.extract(r"(\d+) Owner")[0]
    df["owners"] = pd.to_numeric(df["owners"], errors="coerce").fillna(encoders["owners_median"])

    df["car_age"] = meta["current_year"] - df["year"]
    df["miles_per_year"] = df["miles"] / df["car_age"].replace(0, 1)

    df["brand"] = df["name"].astype(str).apply(extract_brand_name)
    df["model"] = df.apply(lambda r: r["name"][len(r["brand"]):].strip(), axis=1)

    return df


def encode_features(df: pd.DataFrame):
    overall_avg = encoders["overall_avg"]
    df["brand_encoded"] = df["brand"].map(encoders["brand_avg_price"]).fillna(overall_avg)
    df["model_encoder"] = df["model"].map(encoders["model_avg_price"]).fillna(overall_avg)
    df["exterior_encoder_color"] = df["exterior_color"].map(encoders["exterior_avg_price"]).fillna(overall_avg)
    df["interior_encoder_color"] = df["interior_color"].map(encoders["interior_avg_price"]).fillna(overall_avg)
    return df[FEATURE_COLUMNS]


def predict_price(df_features: pd.DataFrame, model_name: str):
    model = models[model_name]
    needs_scaling = bool(results_table.loc[model_name, "needs_scaling"])
    X = scaler.transform(df_features) if needs_scaling else df_features.values
    log_pred = model.predict(X)
    return np.expm1(log_pred)


def build_row_from_form(brand, model_name_car, year, miles, ext_color, int_color, accidents, owners):
    full_name = f"{brand} {model_name_car}".strip()
    raw_row = pd.DataFrame(
        [
            {
                "name": full_name,
                "year": year,
                "miles": f"{miles} miles",
                "color": f"{ext_color} exterior, {int_color} interior",
                "condition": (
                    (f"{accidents} accident{'s' if accidents != 1 else ''} reported" if accidents > 0 else "No accidents reported")
                    + f", {owners} Owner{'s' if owners != 1 else ''}"
                ),
            }
        ]
    )
    return raw_row


with st.sidebar:
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
        f'{gauge_mark(32)}<div>'
        f'<div style="font-family:\'Space Grotesk\',sans-serif; font-size:18px; font-weight:700;">VahanValue</div>'
        f'<div style="font-size:10.5px; opacity:0.75; letter-spacing:.05em;">VALUATION ENGINE</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)

    st.radio("Currency", ["₹ INR", "$ USD"], horizontal=True, key="currency", label_visibility="collapsed")
    st.caption(f"Rate: $1 = ₹{USD_TO_INR:,.1f}")

    st.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)

CURRENCY_SYMBOL = "₹" if st.session_state.get("currency", "₹ INR") == "₹ INR" else "$"
clean_data_display = clean_data.copy()
clean_data_display["price"] = to_display_currency(clean_data["price"])

with st.sidebar:
    st.markdown('<div class="eyebrow">Navigate</div>', unsafe_allow_html=True)
    section = st.radio(
        "Navigate",
        [
            "Dataset Overview",
            "Charts",
            "Model Details",
            "Single Car Prediction",
            "Bulk Prediction",
            "About",
        ],
        label_visibility="collapsed",
    )

st.markdown(
    f'<div class="app-header"><div class="app-header-left">{gauge_mark(38)}<div>'
    f'<div class="app-brand">VahanValue</div>'
    f'<div class="app-tagline">Data-driven car valuation, one listing or ten thousand</div>'
    f'</div></div>'
    f'<div class="app-badge">MODEL · {BEST_MODEL_NAME.upper()} · R² {results_table.loc[BEST_MODEL_NAME, "R2"]:.2f}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

if section == "Dataset Overview":
    section_eyebrow("Dataset")
    st.markdown("#### Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Listings", f"{meta['n_rows_raw']:,}")
    with c2:
        kpi_card("Used In Training", f"{meta['n_rows_clean']:,}")
    with c3:
        kpi_card("Brands Covered", f"{len(meta['brands'])}")
    with c4:
        kpi_card(f"Median Price ({CURRENCY_SYMBOL})", fmt_num(clean_data_display["price"].median()))

    st.write("")
    st.markdown("###### Full cleaned dataset (preview)")
    st.dataframe(
        clean_data_display[
            ["name", "brand", "model", "year", "miles", "exterior_color", "interior_color", "accidents", "owners", "price"]
        ],
        height=340,
    )

    st.markdown("###### Statistical summary")
    st.dataframe(clean_data_display[["price", "miles", "car_age", "accidents", "owners"]].describe().round(2))

elif section == "Charts":
    section_eyebrow("Exploratory Analysis")
    st.markdown("#### Charts")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(clean_data_display, x="price", nbins=40, title=f"Price Distribution ({CURRENCY_SYMBOL})")
        st.plotly_chart(style_fig(fig1), width='stretch', config={"displayModeBar": False})

        fig3 = px.scatter(clean_data_display, x="car_age", y="price", opacity=0.45, title="Car Age vs Price", color_discrete_sequence=[COLORS["secondary"]])
        st.plotly_chart(style_fig(fig3), width='stretch', config={"displayModeBar": False})

        top_brands = clean_data_display["brand"].value_counts().head(10).index
        fig5 = px.box(clean_data_display[clean_data_display["brand"].isin(top_brands)], x="brand", y="price", title="Price by Brand (Top 10)")
        st.plotly_chart(style_fig(fig5), width='stretch', config={"displayModeBar": False})

    with col2:
        fig2 = px.scatter(clean_data_display, x="miles", y="price", opacity=0.45, title="Miles vs Price", color_discrete_sequence=[COLORS["accent"]])
        st.plotly_chart(style_fig(fig2), width='stretch', config={"displayModeBar": False})

        numeric_cols = ["price", "miles", "car_age", "accidents", "owners"]
        corr = clean_data_display[numeric_cols].corr()
        fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale=[COLORS["danger"], "#FFFFFF", COLORS["primary"]], title="Correlation Matrix")
        st.plotly_chart(style_fig(fig4), width='stretch', config={"displayModeBar": False})


        top_models = clean_data_display["model_grp"].value_counts().head(10).index
        fig6 = px.box(clean_data_display[clean_data_display["model_grp"].isin(top_models)], x="model_grp", y="price", title="Price by Model (Top 10)")
        st.plotly_chart(style_fig(fig6), width='stretch', config={"displayModeBar": False})


elif section == "Model Details":
    section_eyebrow("Models")
    st.markdown("#### Model Details")

    k1, k2 = st.columns(2)
    with k1:
        kpi_card("Best Model", BEST_MODEL_NAME, f"R² = {results_table.loc[BEST_MODEL_NAME, 'R2']:.3f}")
    with k2:
        kpi_card(f"Typical Error ({CURRENCY_SYMBOL})", money(results_table.loc[BEST_MODEL_NAME, "MAE"]), "Mean absolute error on test data")

    st.write("")
    st.markdown("###### All trained models compared")
    results_display = results_table[["MAE", "RMSE", "R2"]].copy()
    results_display["MAE"] = to_display_currency(results_display["MAE"]).round(0).astype(int)
    results_display["RMSE"] = to_display_currency(results_display["RMSE"]).round(0).astype(int)
    results_display = results_display.rename(columns={"MAE": f"MAE ({CURRENCY_SYMBOL})", "RMSE": f"RMSE ({CURRENCY_SYMBOL})", "R2": "R²"})
    st.dataframe(results_display, width='stretch')
    st.caption("Lower MAE/RMSE = more accurate. Higher R² = more accurate (max 1.0).")

    st.write("")
    fig_models = px.bar(
        results_table.reset_index().rename(columns={"index": "model"}),
        x="model", y="R2", title="Model Accuracy (R²) Comparison",
        color_discrete_sequence=[COLORS["primary"]],
    )
    st.plotly_chart(style_fig(fig_models, height=320), width='stretch', config={"displayModeBar": False})

    st.markdown("###### Features used by every model")
    st.code("\n".join(FEATURE_COLUMNS), language="text")


elif section == "Single Car Prediction":
    section_eyebrow("Valuation")
    st.markdown("#### Predict price for one car")

    with st.container(border=True):
        model_choice = st.selectbox("Model used for prediction", results_table.index.tolist(), index=results_table.index.tolist().index(BEST_MODEL_NAME))

        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Brand", meta["brands"])
            available_models = meta["models_by_brand"].get(brand, [])
            model_name_car = st.selectbox("Model", available_models) if available_models else st.text_input("Model (type manually)", "")
            year = st.number_input("Year", min_value=1990, max_value=meta["current_year"], value=meta["year_max"])
        with c2:
            miles = st.number_input("Miles driven", min_value=0, max_value=400000, value=int(meta["miles_median"]), step=1000)
            ext_color = st.selectbox("Exterior color", meta["exterior_colors"])
            int_color = st.selectbox("Interior color", meta["interior_colors"])
        with c3:
            accidents = st.number_input("Accidents reported", min_value=0, max_value=10, value=0)
            owners = st.number_input("Number of owners", min_value=1, max_value=10, value=1)

        predict_clicked = st.button(" Predict Price", type="primary", use_container_width="True")

    if predict_clicked:
        raw_row = build_row_from_form(brand, model_name_car, year, miles, ext_color, int_color, accidents, owners)
        feat_df = clean_raw_dataframe(raw_row)
        X = encode_features(feat_df)
        pred = predict_price(X, model_choice)[0]

        st.write("")
        rc1, rc2 = st.columns([1.4, 1])
        with rc1:
            st.markdown(
                f'<div class="result-hero"><div class="result-hero-label">Estimated Market Value</div>'
                f'<div class="result-hero-value">{money(pred)}</div>'
                f'<div class="result-hero-sub">{brand} {model_name_car} · {year} · {miles:,} miles</div></div>',
                unsafe_allow_html=True,
            )
        with rc2:
            kpi_card("Model", model_choice, f"R² = {results_table.loc[model_choice, 'R2']:.3f}")
            st.write("")
            kpi_card("Typical Error (MAE)", money(results_table.loc[model_choice, "MAE"]), "Average deviation on test data")


elif section == "Bulk Prediction":
    section_eyebrow("Batch Valuation")
    st.markdown("#### Predict prices for multiple cars at once")
    st.markdown(
       "Upload file with extention (**.csv**, **.xlsx**, **.json**) with columns: "
               "`name`, `year`, `miles`, `color`, `condition`"
    )

    bulk_model_choice = st.selectbox(
        "Model used for bulk prediction",
        results_table.index.tolist(),
        index=results_table.index.tolist().index(BEST_MODEL_NAME),
        key="bulk_model_choice",
    )

    with st.expander("Download a sample template"):
        sample = pd.DataFrame(
            {
                "name": ["Honda Civic", "Toyota RAV4", "Chevrolet Silverado 1500"],
                "year": [2020, 2022, 2021],
                "miles": ["37,190 miles", "32,879 miles", "15,138 miles"],
                "color": ["Blue exterior, Black interior", "Silver exterior, Unknown interior", "White exterior, Black interior"],
                "condition": ["No accidents reported, 1 Owner", "No accidents reported, 1 Owner", "1 accident reported, 1 Owner"],
            }
        )
        st.dataframe(sample)
        st.download_button("Download sample_template.csv", sample.to_csv(index=False).encode("utf-8"), "sample_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                bulk_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                bulk_df = pd.read_excel(uploaded_file)
            else:
                bulk_df = pd.read_json(uploaded_file)

            required_cols = {"name", "year", "miles", "color", "condition"}
            missing = required_cols - set(bulk_df.columns)
            if missing:
                st.error(f"Cloumns are missing: {sorted(missing)}")
            else:
                st.markdown(f"**{len(bulk_df)} cars uploaded.** Preview:")
                st.dataframe(bulk_df.head(10))

                if st.button("Predict Prices for All", type="primary", use_container_width= "Truem"):
                    with st.spinner("Predicting..."):
                        feat_df = clean_raw_dataframe(bulk_df)
                        X = encode_features(feat_df)
                        preds = predict_price(X, bulk_model_choice)

                        pred_col = f"predicted_price_{CURRENCY_SYMBOL}"
                        result_df = bulk_df.copy()
                        result_df[pred_col] = np.round(to_display_currency(pd.Series(preds)), 0).astype(int).values

                        viz_df = result_df.copy()
                        viz_df["brand"] = feat_df["brand"].values
                        viz_df["miles_numeric"] = feat_df["miles"].values
                        viz_df["accidents"] = feat_df["accidents"].values

                    st.success(f"Done! Predicted prices for {len(result_df)} cars using {bulk_model_choice}.")
                    st.dataframe(result_df, height=360)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download results", csv_bytes, "predicted_prices.csv", "text/csv", type="primary", use_container_width=True)

                    st.write("")
                    st.markdown("<hr>", unsafe_allow_html=True)
                    section_eyebrow("Batch Insights")
                    st.markdown("#### Prediction Analytics")

                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        kpi_card("Cars Valued", f"{len(viz_df):,}")
                    with k2:
                        kpi_card(f"Average Value ({CURRENCY_SYMBOL})", fmt_num(viz_df[pred_col].mean()))
                    with k3:
                        kpi_card(f"Highest ({CURRENCY_SYMBOL})", fmt_num(viz_df[pred_col].max()), viz_df.loc[viz_df[pred_col].idxmax(), "name"])
                    with k4:
                        kpi_card(f"Total Portfolio Value ({CURRENCY_SYMBOL})", fmt_num(viz_df[pred_col].sum()))

                    st.write("")
                    bc1, bc2 = st.columns(2)

                    with bc1:
                        fig_dist = px.histogram(
                            viz_df, x=pred_col, nbins=min(30, max(5, len(viz_df) // 2 or 5)),
                            title=f"Predicted Price Distribution ({CURRENCY_SYMBOL})",
                            color_discrete_sequence=[COLORS["primary"]],
                        )
                        st.plotly_chart(style_fig(fig_dist), width='stretch', config={"displayModeBar": False})

                        top10 = viz_df.nlargest(min(10, len(viz_df)), pred_col).sort_values(pred_col)
                        fig_top = px.bar(
                            top10, x=pred_col, y="name", orientation="h",
                            title="Top Valued Cars in This Batch",
                            color_discrete_sequence=[COLORS["accent"]],
                        )
                        st.plotly_chart(style_fig(fig_top), width='stretch', config={"displayModeBar": False})

                    with bc2:
                        fig_scatter = px.scatter(
                            viz_df, x="miles_numeric", y=pred_col,
                            color="brand" if viz_df["brand"].nunique() > 1 else None,
                            title="Miles Driven vs Predicted Price",
                            labels={"miles_numeric": "Miles"},
                        )
                        st.plotly_chart(style_fig(fig_scatter), width='stretch', config={"displayModeBar": False})

                        if viz_df["brand"].nunique() > 1:
                            avg_by_brand = (
                                viz_df.groupby("brand")[pred_col].mean().sort_values(ascending=False).reset_index()
                            )
                            fig_brand = px.bar(
                                avg_by_brand, x="brand", y=pred_col,
                                title="Average Predicted Price by Brand",
                                color_discrete_sequence=[COLORS["secondary"]],
                            )
                            st.plotly_chart(style_fig(fig_brand), width='stretch', config={"displayModeBar": False})
                        else:
                            fig_acc = px.box(
                                viz_df, x="accidents",
                                y=pred_col, title="Predicted Price by Accident Count",
                                color_discrete_sequence=[COLORS["secondary"]],
                            )
                            st.plotly_chart(style_fig(fig_acc), width='stretch', config={"displayModeBar": False})

        except Exception as e:
            st.error(f"Error: {e}")


elif section == "About":
    section_eyebrow("About")
    st.markdown("#### About VahanValue")
    

    best_r2 = results_table.loc[BEST_MODEL_NAME, "R2"]
    st.markdown(
        f'<div class="about-hero"><div class="about-hero-title">Data-driven used-car valuation</div>'
        f'<div class="about-hero-body">VahanValue estimates the market price of a used car from its brand, model, age, '
        f'mileage, colours, accident history and ownership history. It is built on {meta["n_rows_raw"]:,} real '
        f'web-scraped listings and five machine-learning models trained and compared against each other. '
        f'The best performer, <b>{BEST_MODEL_NAME}</b>, explains about {best_r2 * 100:.0f}% of the price variation '
        f'on unseen test data. Value one car in seconds, or upload a file and value thousands at once.</div></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Listings Scraped", f"{meta['n_rows_raw']:,}", f"{meta['n_rows_clean']:,} used after cleaning")
    with k2:
        kpi_card("Brands", f"{len(meta['brands'])}", f"Model years {meta['year_min']}–{meta['year_max']}")
    with k3:
        kpi_card("Models Trained", f"{len(results_table)}", "Compared on the same test set")
    with k4:
        kpi_card("Best Accuracy (R²)", f"{best_r2:.3f}", BEST_MODEL_NAME)

    st.write("")
    section_eyebrow("Pipeline")
    st.markdown("#### How it works")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        info_card("Collect", "Used-car listings were scraped from the web: name, year, miles, colour, condition and asking price.", "STEP 1", min_height=190)
    with s2:
        info_card("Clean & engineer", "Text fields are parsed into numbers. Brand, model, colours, accidents, owners, car age and miles per year are derived.", "STEP 2", min_height=190)
    with s3:
        info_card("Train & compare", "Five regression models are trained on a log-transformed price and scored on a held-out 20% test set.", "STEP 3", min_height=190)
    with s4:
        info_card("Predict", "Your input goes through the identical cleaning and encoding steps, then the chosen model returns a price.", "STEP 4", min_height=190)

    st.write("")
    section_eyebrow("Data")
    st.markdown("#### Dataset & preparation")
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown("**Raw columns**")
            st.markdown(
                "- `name` — brand and model together\n"
                "- `year` — model year\n"
                "- `miles` — odometer reading\n"
                "- `color` — exterior and interior colour in one field\n"
                "- `condition` — accident history and number of owners\n"
                "- `price` — asking price (the value we predict)"
            )
    with d2:
        with st.container(border=True):
            st.markdown("**Cleaning steps**")
            st.markdown(
                "- Text such as `37,190 miles` and `$21,998` converted to plain numbers\n"
                "- Brand split from model (handles two-word brands like Land Rover)\n"
                "- Cheapest 1% and most expensive 1% of listings removed as outliers\n"
                "- Rare brands (under 15 listings) grouped as `Other`\n"
                "- Missing owner counts filled with the median\n"
                "- 80% / 20% train–test split with a fixed random seed"
            )

    st.write("")
    st.markdown("###### Features the models learn from")
    feature_notes = {
        "miles": "Odometer reading",
        "car_age": f"{meta['current_year']} minus model year",
        "miles_per_year": "Miles divided by car age, a usage-intensity signal",
        "accidents": "Number of reported accidents",
        "owners": "Number of previous owners",
        "brand_encoded": "Average log-price of the brand in training data",
        "model_encoder": "Average log-price of the model in training data",
        "exterior_encoder_color": "Average log-price of the exterior colour",
        "interior_encoder_color": "Average log-price of the interior colour",
    }
    feat_df_about = pd.DataFrame(
        {"Feature": FEATURE_COLUMNS, "Meaning": [feature_notes.get(f, "") for f in FEATURE_COLUMNS]}
    )
    st.dataframe(feat_df_about, hide_index=True, width="stretch")
    st.caption(
        "Brand, model and colour are converted to numbers with target (mean) encoding, "
        "computed on the training split only so the test score stays honest."
    )

    st.write("")
    section_eyebrow("Models")
    st.markdown("#### Models compared")
    model_notes = {
        "Linear Regression": ("Straight-line baseline", "Uses scaled features"),
        "KNN Regressor": ("Averages the 7 most similar cars", "Uses scaled features"),
        "Decision Tree": ("Rule-based splits, max depth 8", "No scaling needed"),
        "Random Forest": ("400 decision trees voting together", "No scaling needed"),
        "XG Boost": ("400 boosted trees, learning rate 0.05, depth 5", "No scaling needed"),
    }
    rows = []
    for name in results_table.index:
        desc, scaling = model_notes.get(name, ("", ""))
        rows.append(
            {
                "Model": name + (" ★" if name == BEST_MODEL_NAME else ""),
                "How it works": desc,
                "Preprocessing": scaling,
                f"MAE ({CURRENCY_SYMBOL})": int(round(float(to_display_currency(pd.Series([results_table.loc[name, "MAE"]])).iloc[0]))),
                "R²": round(float(results_table.loc[name, "R2"]), 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("★ marks the best model. MAE is the average miss between predicted and actual price; lower is better. R² closer to 1 is better.")

    st.write("")
    section_eyebrow("Guide")
    st.markdown("#### Using the app")
    g1, g2, g3 = st.columns(3)
    with g1:
        info_card(
            "Single Car Prediction",
            "Pick brand, model, year, miles, colours, accidents and owners, choose a model, and click Predict Price to get one valuation.",
            min_height=150,
        )
    with g2:
        info_card(
            "Bulk Prediction",
            "Upload a .csv, .xlsx or .json file. Every row is valued at once, charts are generated for the batch, and results download as CSV, XLSX or JSON.",
            min_height=150,
        )
    with g3:
        info_card(
            "Exploring the data",
            "Dataset Overview, Charts and Model Details show what the models were trained on and how each one performed.",
            min_height=150,
        )

    st.write("")
    st.markdown("###### Bulk upload format")
    sample_about = pd.DataFrame(
        {
            "name": ["Honda Civic", "Toyota RAV4"],
            "year": [2020, 2022],
            "miles": ["37,190 miles", "32,879 miles"],
            "color": ["Blue exterior, Black interior", "Silver exterior, Unknown interior"],
            "condition": ["No accidents reported, 1 Owner", "1 accident reported, 1 Owner"],
        }
    )
    st.dataframe(sample_about, hide_index=True, width="stretch")
    st.caption("All five columns are required. A ready-made template can be downloaded from the Bulk Prediction page.")

    st.write("")
    section_eyebrow("Good to know")
    st.markdown("#### Currency, accuracy and limits")
    n1, n2 = st.columns(2)
    with n1:
        with st.container(border=True):
            st.markdown("**Currency**")
            st.markdown(
                f"The source listings are priced in US dollars, so every model predicts in USD. "
                f"For display, values are converted to rupees at a fixed rate of \\$1 = ₹{USD_TO_INR:,.1f} "
                f"(changeable in `app.py`). Use the toggle in the sidebar to switch between ₹ and \\$."
            )
    with n2:
        with st.container(border=True):
            st.markdown("**Accuracy and limits**")
            mae_best = results_table.loc[BEST_MODEL_NAME, "MAE"]
            st.markdown(
                f"On the test set the best model is typically off by about {money(mae_best)}. "
                "Cars of a brand, model or colour not seen in training fall back to the overall average, "
                "so estimates for them are less reliable. Predictions are estimates, not a formal appraisal "
                "or a guaranteed selling price."
            )

    st.write("")
    st.markdown("###### Built with")
    st.markdown("`Python` · `Streamlit` · `pandas` · `scikit-learn` · `XGBoost` · `Plotly` · `joblib`")
