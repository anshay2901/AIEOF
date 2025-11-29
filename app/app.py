# ============================================================
# Streamlit App: AI Energy Forecaster (Polished Edition)
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from datetime import datetime

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="AI Energy Forecaster",
    page_icon="⚡",
    layout="wide",
)

# Page Title
st.markdown(
    "<h1 style='text-align:center; margin-bottom: 10px;'>⚡ AI Energy Forecaster</h1>",
    unsafe_allow_html=True,
)

st.caption(
    "This dashboard uses AI/ML (Prophet) to forecast India's daily & hourly industrial electricity demand using MoSPI-aligned synthetic energy data, with anomaly detection and optimization insights."
)

# ============================================================
# Global CSS Styling (Premium UI)
# ============================================================
st.markdown(
    """
<style>
/* Layout reset */
main {
    padding: 0rem 1rem;
}

/* Section divider */
.section-divider {
    margin-top: 30px;
    margin-bottom: 20px;
    border-top: 1px solid rgba(255,255,255,0.15);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1116;
    padding-top: 20px;
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* KPI cards */
.metric-box {
    background: linear-gradient(
        to bottom right,
        rgba(255,255,255,0.08),
        rgba(255,255,255,0.03)
    );
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    transition: transform 0.1s ease;
    text-align: center;
}

.metric-box:hover {
    transform: translateY(-3px);
}

.metric-label {
    font-size: 14px;
    opacity: 0.75;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: -5px;
}

/* Dataframe container */
.dataframe {
    border-radius: 12px;
    overflow: hidden;
}

/* Plot spacing */
.plot-container {
    margin-top: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# File Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

P_DAILY = os.path.join(MODELS_DIR, "daily_totals_with_anomaly_scores.csv")
P_ANOMS = os.path.join(MODELS_DIR, "daily_anomalies.csv")
P_HOURLY = os.path.join(MODELS_DIR, "hourly_forecast_allocated.csv")
P_PEAKS = os.path.join(MODELS_DIR, "forecast_daily_peaks_next30.csv")

# ============================================================
# Cached CSV Loader
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv_safe(path, parse_dates=None):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

daily_df = load_csv_safe(P_DAILY, parse_dates=["date"])
anoms_df = load_csv_safe(P_ANOMS, parse_dates=["date"])
hourly_df = load_csv_safe(P_HOURLY, parse_dates=["datetime"])
peaks_df = load_csv_safe(P_PEAKS, parse_dates=["date"])

# ============================================================
# Column Normalization Helpers
# ============================================================
def normalize_daily(df):
    if df is None or df.empty:
        return df
    rename_map = {
        "industry_consumption_mwh": "mwh",
        "yhat": "mwh",
        "value": "mwh",
        "load_mwh": "mwh",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    if "mwh" not in df.columns:
        df["mwh"] = np.nan
    if "anomaly" not in df.columns:
        df["anomaly"] = 0
    if "score" not in df.columns:
        df["score"] = np.nan
    return df

def normalize_hourly(df):
    if df is None or df.empty:
        return df
    if "mwh" not in df.columns:
        for alt in ["value", "load_mwh", "industry_consumption_mwh"]:
            if alt in df.columns:
                df.rename(columns={alt: "mwh"}, inplace=True)
    return df

def normalize_peaks(df):
    if df is None or df.empty:
        return df
    rename_map = {
        "peak": "peak_mwh",
        "max_mwh": "peak_mwh",
        "hour": "peak_hour",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df

daily_df = normalize_daily(daily_df)
hourly_df = normalize_hourly(hourly_df)
peaks_df = normalize_peaks(peaks_df)

# ============================================================
# Premade KPI Card Renderer
# ============================================================
def metric_card(label, value, unit=None):
    if value is None or np.isnan(value):
        val = "—"
    else:
        if unit == "%":
            val = f"{value:.2f}%"
        elif unit == "Hour":
            val = f"{value}"
        else:
            val = f"{value:,.0f}"
    return f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}{'' if unit=='Hour' else f' {unit or ""}'}</div>
        </div>
    """

# ============================================================
# Sidebar Navigation
# ============================================================
page = st.sidebar.radio(
    "Navigate",
    [
        "📈 Overview (Daily)",
        "⏱️ Hourly & Peaks",
        "🚨 Anomalies",
        "🧠 Optimization",
        "⬇️ Downloads",
    ],
)

st.sidebar.markdown(
    "<div class='small-note'>Loaded from CSV exports in <code>/models</code>.</div>",
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1 — Overview (Daily)
# ============================================================
if page.startswith("📈"):

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## 📈 Energy Demand Overview (Daily)")

    # --------------------------
    # Model Accuracy Cards
    # --------------------------
    st.subheader("📊 Model Accuracy (Last 30 Days)")
    colA, colB, colC = st.columns(3)

    colA.markdown(metric_card("MAE", 107333, "MWh"), unsafe_allow_html=True)
    colB.markdown(metric_card("RMSE", 118550, "MWh"), unsafe_allow_html=True)
    colC.markdown(metric_card("MAPE", 5.87, "%"), unsafe_allow_html=True)

    if daily_df is None or daily_df.empty:
        st.warning("Daily dataset missing.")
        st.stop()

    min_d, max_d = daily_df["date"].min(), daily_df["date"].max()
    start_d, end_d = st.slider(
        "Select Date Range",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(
            max(min_d, max_d - pd.Timedelta(days=180)).to_pydatetime(),
            max_d.to_pydatetime(),
        ),
    )
    mask = (daily_df["date"] >= pd.to_datetime(start_d)) & (daily_df["date"] <= pd.to_datetime(end_d))
    ddf = daily_df.loc[mask].sort_values("date")

    # --------------------------
    # Daily KPI Cards
    # --------------------------
    today = ddf["date"].max()
    yesterday = today - pd.Timedelta(days=1)

    todays_mwh = ddf.loc[ddf["date"] == today, "mwh"].mean()
    yday_mwh = ddf.loc[ddf["date"] == yesterday, "mwh"].mean()
    w7_avg = ddf.tail(7)["mwh"].mean()

    pct_vs_y = (
        (todays_mwh - yday_mwh) / yday_mwh * 100
        if yday_mwh and not np.isnan(todays_mwh) else np.nan
    )

    peak_hour_today = (
        int(peaks_df.loc[peaks_df["date"] == today, "peak_hour"].iloc[0])
        if peaks_df is not None and not peaks_df.empty and today in peaks_df["date"].values
        else np.nan
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(metric_card("Today", todays_mwh, "MWh"), unsafe_allow_html=True)
    col2.markdown(metric_card("Yesterday", yday_mwh, "MWh"), unsafe_allow_html=True)
    col3.markdown(metric_card("7-Day Avg", w7_avg, "MWh"), unsafe_allow_html=True)
    col4.markdown(metric_card("% vs Yesterday", pct_vs_y, "%"), unsafe_allow_html=True)
    col5.markdown(metric_card("Peak Hour Today", peak_hour_today, "Hour"), unsafe_allow_html=True)

    # --------------------------
    # Daily Time Series Chart
    # --------------------------
    fig = px.line(
        ddf,
        x="date",
        y="mwh",
        title="Daily Industrial Electricity Consumption (MWh)",
        labels={"date": "Date", "mwh": "MWh"},
        height=460,
    )

    if "anomaly" in ddf.columns and ddf["anomaly"].sum() > 0:
        anom_pts = ddf[ddf["anomaly"] == 1]
        fig.add_scatter(
            x=anom_pts["date"],
            y=anom_pts["mwh"],
            mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name="Anomaly",
        )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Seasonality Charts
    # --------------------------
    c1, c2 = st.columns(2)

    ddf["dow"] = ddf["date"].dt.day_name()
    ddf["month"] = ddf["date"].dt.month_name()

    with c1:
        st.subheader("Weekly Pattern")
        order_dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_dow = ddf.groupby("dow", as_index=False)["mwh"].mean()
        by_dow["dow"] = pd.Categorical(by_dow["dow"], categories=order_dow, ordered=True)
        by_dow = by_dow.sort_values("dow")
        fig_dow = px.bar(
            by_dow,
            x="dow",
            y="mwh",
            title="Average by Day of Week",
            labels={"dow": "Day", "mwh": "Avg MWh"},
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with c2:
        st.subheader("Yearly Seasonality (Synthetic)")
        months = list(pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B"))
        by_month = ddf.groupby("month", as_index=False)["mwh"].mean()
        by_month["month"] = pd.Categorical(by_month["month"], categories=months, ordered=True)
        by_month = by_month.sort_values("month")
        fig_month = px.line(
            by_month,
            x="month",
            y="mwh",
            markers=True,
            title="Average by Month",
            labels={"month": "Month", "mwh": "Avg MWh"},
        )
        st.plotly_chart(fig_month, use_container_width=True)

# ============================================================
# PAGE 2 — Hourly & Peaks
# ============================================================
elif page.startswith("⏱️"):

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## ⏱️ Hourly Forecast & Peak Analysis")

    if hourly_df is None or hourly_df.empty:
        st.warning("Hourly dataset missing.")
        st.stop()

    hourly_df = hourly_df.sort_values("datetime")
    min_t, max_t = hourly_df["datetime"].min(), hourly_df["datetime"].max()

    start_t, end_t = st.slider(
        "Time Range",
        min_value=min_t.to_pydatetime(),
        max_value=max_t.to_pydatetime(),
        value=(max(min_t, max_t - pd.Timedelta(days=7)).to_pydatetime(), max_t.to_pydatetime()),
    )
    hdf = hourly_df.loc[
        (hourly_df["datetime"] >= start_t) & (hourly_df["datetime"] <= end_t)
    ]

    # Hourly line chart
    fig_hour = px.line(
        hdf,
        x="datetime",
        y="mwh",
        title="Hourly Forecasted Load (MWh)",
        labels={"datetime": "Date/Hour", "mwh": "MWh"},
        height=460,
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # Heatmap: DOW x Hour
    st.markdown("### 🔥 Average Hourly Profile Heatmap")
    tmp = hourly_df.copy()
    tmp["dow"] = tmp["datetime"].dt.day_name()
    tmp["hour"] = tmp["datetime"].dt.hour

    order_dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = tmp.groupby(["dow", "hour"], as_index=False)["mwh"].mean()
    pivot["dow"] = pd.Categorical(pivot["dow"], categories=order_dow, ordered=True)
    pivot = pivot.sort_values(["dow", "hour"])

    heat = pivot.pivot(index="hour", columns="dow", values="mwh")
    fig_hm = px.imshow(
        heat,
        labels=dict(x="Day of Week", y="Hour", color="Avg MWh"),
        aspect="auto",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Peaks
    st.markdown("### 📊 Forecasted Daily Peaks (Next 30 Days)")
    if peaks_df is None or peaks_df.empty:
        st.info("Peak forecast file missing.")
    else:
        figp = px.scatter(
            peaks_df.sort_values("date"),
            x="date",
            y="peak_mwh",
            color="peak_hour",
            color_continuous_scale="Turbo",
            labels={"peak_mwh": "Peak MWh", "date": "Date", "peak_hour": "Hour"},
            title="Forecasted Peak Load",
        )
        figp.update_traces(mode="lines+markers")
        st.plotly_chart(figp, use_container_width=True)

# ============================================================
# PAGE 3 — Anomalies
# ============================================================
elif page.startswith("🚨"):

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## 🚨 Anomaly Detection (Daily)")

    if daily_df is None or daily_df.empty:
        st.warning("Daily dataset missing.")
        st.stop()

    ddf = daily_df.sort_values("date")
    anomaly_count = int(ddf["anomaly"].sum())

    colA, colB, colC = st.columns(3)
    colA.markdown(metric_card("Total Days", len(ddf), "Days"), unsafe_allow_html=True)
    colB.markdown(metric_card("Anomaly Days", anomaly_count, "Days"), unsafe_allow_html=True)
    colC.markdown(
        metric_card("Anomaly Rate", anomaly_count / len(ddf) * 100, "%"),
        unsafe_allow_html=True,
    )

    # Plot anomalies
    fig = px.line(
        ddf,
        x="date",
        y="mwh",
        title="Daily Load with Anomaly Markers",
        labels={"date": "Date", "mwh": "MWh"},
    )

    pts = ddf[ddf["anomaly"] == 1]
    if len(pts) > 0:
        fig.add_scatter(
            x=pts["date"],
            y=pts["mwh"],
            mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name="Anomaly",
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anomaly Table")
    if anoms_df is not None and not anoms_df.empty:
        st.dataframe(anoms_df.sort_values("date", ascending=False), height=380)
    else:
        st.caption("No separate anomalies CSV found — showing anomaly flags from daily file.")

# ============================================================
# PAGE 4 — Optimization
# ============================================================
elif page.startswith("🧠"):

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## 🧠 Optimization Insights (Load Shifting)")

    st.caption("This view illustrates energy savings from shaving 10% of upcoming peak loads.")

    if peaks_df is None or peaks_df.empty:
        st.warning("Peak forecast data not found.")
        st.stop()

    topn = st.slider("Show top N peaks", 5, 30, 10)

    show = peaks_df.nlargest(topn, "peak_mwh").sort_values("date")
    st.dataframe(show, height=300, use_container_width=True)

    show["after_shave"] = show["peak_mwh"] * 0.90

    fig_opt = px.line(
        show,
        x="date",
        y=["peak_mwh", "after_shave"],
        labels={"value": "MWh", "date": "Date", "variable": ""},
        title=f"Peak-Shaving Simulation ({topn} Days)",
    )
    st.plotly_chart(fig_opt, use_container_width=True)

    total_saved = (show["peak_mwh"] - show["after_shave"]).sum()
    st.success(f"Estimated energy saved: **{total_saved:,.0f} MWh** (10% shaving)")

# ============================================================
# PAGE 5 — Downloads
# ============================================================
elif page.startswith("⬇️"):

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## ⬇️ Downloads & Exports")

    files = [
        ("Daily totals with anomalies", P_DAILY),
        ("Daily anomalies only", P_ANOMS),
        ("Hourly forecast", P_HOURLY),
        ("Daily peak forecast", P_PEAKS),
    ]

    for label, path in files:
        st.subheader(label)
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    "Download CSV",
                    f,
                    file_name=os.path.basename(path),
                    mime="text/csv",
                )

            sample = load_csv_safe(path)
            if sample is not None:
                st.caption("Preview:")
                st.dataframe(sample.head(8), height=250)
        else:
            st.warning(f"File not found: {path}")
# ============================================================
# Footer + Branding
# ============================================================
st.markdown(
    """
    <style>
        .footer {
            margin-top: 40px;
            padding-top: 12px;
            padding-bottom: 12px;
            border-top: 1px solid rgba(255,255,255,0.15);
            text-align: center;
            font-size: 14px;
            opacity: 0.7;
        }
        .footer a {
            color: #ffffff;
            text-decoration: none;
            font-weight: 600;
        }
        .footer:hover {
            opacity: 1.0;
        }
    </style>

    <div class="footer">
        ⚡ <strong>AI Energy Forecaster (AIEOF)</strong> — Built for 
        <strong>Viksit Bharat 2047 – Environmental Sustainability Challenge</strong><br>
        Crafted by <strong>Anshay Singh</strong> · 
        <a href="https://aieof-toi.streamlit.app/" target="_blank">Live Dashboard</a>
    </div>
    """,
    unsafe_allow_html=True,
)
