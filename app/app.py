# Streamlit app: AI Energy Forecaster
# Folder assumptions (relative to this file):
#   ../models/daily_totals_with_anomaly_scores.csv
#   ../models/daily_anomalies.csv
#   ../models/hourly_forecast_allocated.csv
#   ../models/forecast_daily_peaks_next30.csv

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

#############################################
# Page config
#############################################
st.set_page_config(
    page_title="AI Energy Forecaster",
    page_icon="⚡",
    layout="wide",
)
st.caption(
    "This dashboard uses AI/ML (Prophet) to forecast India's daily & hourly industrial electricity demand using MoSPI-aligned synthetic energy data, with anomaly detection and optimization insights."
)


st.markdown(
    """
    <style>
    .small-note {font-size: 0.85rem; color:#666;}
    .metric-ok {color:#0a7b0a;}
    .metric-warn {color:#b36b00;}
    .metric-bad {color:#a00000;}
    </style>
    """,
    unsafe_allow_html=True
)

#############################################
# Paths
#############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

P_DAILY = os.path.join(MODELS_DIR, "daily_totals_with_anomaly_scores.csv")
P_ANOMS = os.path.join(MODELS_DIR, "daily_anomalies.csv")
P_HOURLY = os.path.join(MODELS_DIR, "hourly_forecast_allocated.csv")
P_PEAKS = os.path.join(MODELS_DIR, "forecast_daily_peaks_next30.csv")

#############################################
# Data loaders (cached)
#############################################
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
        st.error(f"Failed to read {os.path.basename(path)}: {e}")
        return None

daily_df   = load_csv_safe(P_DAILY, parse_dates=["date"])
anoms_df   = load_csv_safe(P_ANOMS, parse_dates=["date"])
hourly_df  = load_csv_safe(P_HOURLY, parse_dates=["datetime"])
peaks_df   = load_csv_safe(P_PEAKS, parse_dates=["date"])

#############################################
# Friendly column normalization
#############################################
def normalize_daily(df):
    if df is None or df.empty:
        return df
    # Expected: columns ["date", "mwh", "anomaly", "score"]
    # Provide fallbacks if names vary
    c = {c.lower(): c for c in df.columns}
    if "date" not in c:
        # try 'ds'
        if "ds" in c:
            df.rename(columns={c["ds"]:"date"}, inplace=True)
    if "mwh" not in c:
        # try common variants
        for alt in ["industry_consumption_mwh", "yhat", "value", "load_mwh"]:
            if alt in df.columns:
                df.rename(columns={alt:"mwh"}, inplace=True)
                break
    # optional flags
    if "anomaly" not in df.columns:
        df["anomaly"] = 0
    if "score" not in df.columns:
        df["score"] = np.nan
    return df

def normalize_hourly(df):
    if df is None or df.empty: return df
    # Expected: ["datetime","mwh"]
    if "datetime" not in df.columns:
        # try 'ts'
        if "ts" in df.columns:
            df.rename(columns={"ts":"datetime"}, inplace=True)
    if "mwh" not in df.columns:
        for alt in ["mw", "value", "load_mwh"]:
            if alt in df.columns:
                df.rename(columns={alt:"mwh"}, inplace=True)
                break
    return df

def normalize_peaks(df):
    if df is None or df.empty: return df
    # Expected: ["date","peak_mwh","peak_hour"]
    if "peak_mwh" not in df.columns:
        for alt in ["peak", "max_mwh", "max_load"]:
            if alt in df.columns:
                df.rename(columns={alt:"peak_mwh"}, inplace=True)
                break
    if "peak_hour" not in df.columns:
        for alt in ["hour", "peak_hr"]:
            if alt in df.columns:
                df.rename(columns={alt:"peak_hour"}, inplace=True)
                break
    return df

daily_df  = normalize_daily(daily_df)
hourly_df = normalize_hourly(hourly_df)
peaks_df  = normalize_peaks(peaks_df)

#############################################
# Sidebar Navigation
#############################################
st.sidebar.title("⚡ AI Energy Forecaster")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview (Daily)",
        "Hourly & Peaks",
        "Anomalies",
        "Optimization",
        "Downloads"
    ],
)

st.sidebar.markdown(
    "<div class='small-note'>Data shown from exported CSVs in <code>/models</code>.</div>",
    unsafe_allow_html=True
)

#############################################
# Helper: KPI cards
#############################################
def kpi_row(cols, labels, values, suffix=" MWh"):
    for col, label, value in zip(cols, labels, values):
        if value is None or np.isnan(value):
            col.metric(label, "—")
        else:
            col.metric(label, f"{value:,.0f}{suffix}")

#############################################
# PAGE: Overview (Daily)
#############################################
if page == "Overview (Daily)":
    st.header("📈 Energy Demand Overview (Daily)")
   # ==============================================
    # 🌟 Model Accuracy Metrics (Improved Styling)
    # ==============================================

    st.subheader("📊 Model Accuracy (Last 30 Days)")

    # CSS for beautiful metric boxes
    st.markdown("""
        <style>
            .metric-box {
                background-color: rgba(255,255,255,0.05);
                padding: 18px;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.1);
                text-align: center;
            }
            .metric-value {
                font-size: 28px;
                font-weight: 600;
                margin-top: -10px;
            }
            .metric-label {
                font-size: 14px;
                opacity: 0.7;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-label">MAE</div>
                <div class="metric-value">107,333 MWh</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">118,550 MWh</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-label">MAPE</div>
                <div class="metric-value">5.87%</div>
            </div>
        """, unsafe_allow_html=True)


    if daily_df is None or daily_df.empty:
        st.warning("Daily dataset not found. Make sure the notebook saved `daily_totals_with_anomaly_scores.csv`.")
        st.stop()

    # Date filter
    min_d, max_d = daily_df["date"].min(), daily_df["date"].max()
    start_d, end_d = st.slider(
        "Date range", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(),
        value=(max(min_d, max_d - pd.Timedelta(days=180)).to_pydatetime(), max_d.to_pydatetime()),
    )
    mask = (daily_df["date"] >= pd.to_datetime(start_d)) & (daily_df["date"] <= pd.to_datetime(end_d))
    ddf = daily_df.loc[mask].sort_values("date")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    today = ddf["date"].max()
    yday = today - pd.Timedelta(days=1)

    todays_mwh = float(ddf.loc[ddf["date"]==today, "mwh"].mean()) if not ddf.loc[ddf["date"]==today].empty else np.nan
    yday_mwh   = float(ddf.loc[ddf["date"]==yday,   "mwh"].mean()) if not ddf.loc[ddf["date"]==yday].empty else np.nan
    w7_avg     = float(ddf.tail(7)["mwh"].mean()) if len(ddf) >= 7 else np.nan
    pct_vs_y   = ((todays_mwh - yday_mwh)/yday_mwh*100) if (not np.isnan(todays_mwh) and not np.isnan(yday_mwh) and yday_mwh!=0) else np.nan
    peak_hour_today = int(peaks_df.loc[peaks_df["date"]==today, "peak_hour"].iloc[0]) if (peaks_df is not None and not peaks_df.empty and today in peaks_df["date"].values) else np.nan

    kpi_row(
        (col1, col2, col3, col4, col5),
        ("Today’s Demand", "Yesterday", "7-day Avg", "% vs Yesterday", "Peak Hour Today"),
        (todays_mwh, yday_mwh, w7_avg, pct_vs_y, peak_hour_today),
        suffix=""  # Default suffix set below per field
    )
    # Pretty suffix per metric
    col1.caption("MWh"); col2.caption("MWh"); col3.caption("MWh")
    col4.caption("%");   col5.caption("Hour (0–23)")

    # Main daily chart
    fig = px.line(
        ddf, x="date", y="mwh",
        title="Daily Industrial Electricity Consumption (MWh)",
        labels={"date": "Date", "mwh": "Consumption (MWh)"},
        height=450
    )
    # Mark anomalies if present
    if "anomaly" in ddf.columns and ddf["anomaly"].sum() > 0:
        anom_pts = ddf[ddf["anomaly"] == 1]
        fig.add_scatter(
            x=anom_pts["date"], y=anom_pts["mwh"],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=8, symbol="x")
        )
    st.plotly_chart(fig, use_container_width=True)

    # Seasonality quick views (weekly + monthly)
    c1, c2 = st.columns(2)
    ddf["dow"] = ddf["date"].dt.day_name()
    ddf["month"] = ddf["date"].dt.month_name()

    with c1:
        by_dow = ddf.groupby("dow", as_index=False)["mwh"].mean()
        # Order weekdays
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_dow["dow"] = pd.Categorical(by_dow["dow"], categories=order, ordered=True)
        by_dow = by_dow.sort_values("dow")
        st.subheader("Weekly Pattern")
        st.plotly_chart(px.bar(by_dow, x="dow", y="mwh", labels={"mwh":"Avg MWh","dow":"Day of Week"},
                               title="Average Consumption by Day of Week"), use_container_width=True)

    with c2:
        by_month = ddf.groupby("month", as_index=False)["mwh"].mean()
        # Order months
        months_order = list(pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B"))
        by_month["month"] = pd.Categorical(by_month["month"], categories=months_order, ordered=True)
        by_month = by_month.sort_values("month")
        st.subheader("Yearly Seasonality")
        st.plotly_chart(px.line(by_month, x="month", y="mwh", markers=True,
                                labels={"mwh":"Avg MWh","month":"Month"},
                                title="Average Consumption by Month"), use_container_width=True)

#############################################
# PAGE: Hourly & Peaks
#############################################
elif page == "Hourly & Peaks":
    st.header("⏱️ Hourly Forecast & Peak Analysis")
    if hourly_df is None or hourly_df.empty:
        st.warning("Hourly forecast file not found. Expected `hourly_forecast_allocated.csv`.")
        st.stop()

    hourly_df = hourly_df.sort_values("datetime")
    # Range picker (last 7 days by default if available)
    min_t, max_t = hourly_df["datetime"].min(), hourly_df["datetime"].max()
    default_start = max(min_t, max_t - pd.Timedelta(days=7))
    start_t, end_t = st.slider(
        "Time range", min_value=min_t.to_pydatetime(), max_value=max_t.to_pydatetime(),
        value=(default_start.to_pydatetime(), max_t.to_pydatetime())
    )
    hmask = (hourly_df["datetime"] >= pd.to_datetime(start_t)) & (hourly_df["datetime"] <= pd.to_datetime(end_t))
    hdf = hourly_df.loc[hmask]

    # Hourly line
    fig = px.line(
        hdf, x="datetime", y="mwh",
        title="Hourly Forecasted Load (MWh)",
        labels={"datetime":"Date/Hour","mwh":"MWh"},
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap by Day-of-Week x Hour
    st.subheader("Average Hourly Profile (Heatmap)")
    tmp = hourly_df.copy()
    tmp["dow"] = tmp["datetime"].dt.day_name()
    tmp["hour"] = tmp["datetime"].dt.hour
    pivot = tmp.groupby(["dow","hour"], as_index=False)["mwh"].mean()
    # Order axes
    order_dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot["dow"] = pd.Categorical(pivot["dow"], categories=order_dow, ordered=True)
    pivot = pivot.sort_values(["dow","hour"])

    heat = pivot.pivot(index="hour", columns="dow", values="mwh")
    fig_hm = px.imshow(
        heat,
        labels=dict(x="Day of Week", y="Hour", color="Avg MWh"),
        aspect="auto",
        title="Average Hourly Consumption by Day-of-Week"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Peaks
    st.subheader("Forecasted Daily Peaks (Next 30 Days)")
    if peaks_df is None or peaks_df.empty:
        st.info("No peaks file found (`forecast_daily_peaks_next30.csv`).")
    else:
        figp = px.scatter(
            peaks_df.sort_values("date"),
            x="date", y="peak_mwh",
            color="peak_hour",
            color_continuous_scale="Turbo",
            labels={"peak_mwh":"Peak MWh","date":"Date","peak_hour":"Peak Hour"},
            title="Forecasted Daily Peak Load (Next 30 Days)",
        )
        figp.update_traces(mode="lines+markers")
        st.plotly_chart(figp, use_container_width=True)

#############################################
# PAGE: Anomalies
#############################################
elif page == "Anomalies":
    st.header("🚨 Anomaly Detection (Daily)")
    if daily_df is None or daily_df.empty:
        st.warning("Daily dataset not found.")
        st.stop()

    ddf = daily_df.sort_values("date")
    anomaly_count = int(ddf["anomaly"].sum()) if "anomaly" in ddf.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Days", f"{len(ddf):,}")
    c2.metric("Anomaly Days", f"{anomaly_count:,}")
    c3.metric("Anomaly Rate", f"{(anomaly_count/len(ddf)*100 if len(ddf)>0 else 0):.2f}%")

    # Plot with anomaly markers
    fig = px.line(ddf, x="date", y="mwh",
                  title="Daily MWh with Anomalies Highlighted",
                  labels={"date":"Date","mwh":"MWh"})
    if "anomaly" in ddf.columns:
        pts = ddf[ddf["anomaly"] == 1]
        fig.add_scatter(
            x=pts["date"], y=pts["mwh"],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=8, symbol="x")
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Table")
    if anoms_df is not None and not anoms_df.empty:
        st.dataframe(anoms_df.sort_values("date", ascending=False), use_container_width=True, height=400)
    else:
        st.caption("No dedicated `daily_anomalies.csv` found. Showing markers from the main daily file instead.")

#############################################
# PAGE: Optimization
#############################################
elif page == "Optimization":
    st.header("🧠 Optimization Hints (Load Shifting)")
    st.caption("Illustrative view based on the 'peak shaving' visualization you produced in the notebook.")

    # If you have your “top 10 shaved” dataset saved, you can read and show it here.
    st.info(
        "To enable an interactive optimization panel, export a CSV with columns "
        "`date, peak_mwh, after_shave_mwh` and I’ll wire it to a delta-savings view."
    )

    if peaks_df is not None and not peaks_df.empty:
        st.subheader("Top Upcoming Peaks")
        topn = st.slider("Show top N future peaks", 5, 30, 10)
        show = peaks_df.nlargest(topn, "peak_mwh").sort_values("date")
        st.dataframe(show, use_container_width=True, height=300)

        # Quick delta vs hypothetical 10% shave
        show["after_10pct_shave"] = show["peak_mwh"] * 0.90
        fig_opt = px.line(
            show,
            x="date", y=["peak_mwh","after_10pct_shave"],
            labels={"value":"Peak MWh","date":"Date","variable":""},
            title=f"Peak-Shaving Illustration on Top {topn} Upcoming Peaks"
        )
        st.plotly_chart(fig_opt, use_container_width=True)

        total_saved = (show["peak_mwh"] - show["after_10pct_shave"]).sum()
        st.success(f"Illustrative savings from 10% shaving on top {topn} future peaks: **{total_saved:,.0f} MWh**")

#############################################
# PAGE: Downloads
#############################################
elif page == "Downloads":
    st.header("⬇️ Downloads & Exports")

    files = [
        ("Daily totals with anomaly scores", P_DAILY),
        ("Daily anomalies only", P_ANOMS),
        ("Hourly forecast (allocated)", P_HOURLY),
        ("Forecasted daily peaks (next 30 days)", P_PEAKS),
    ]
    for label, path in files:
        with st.container():
            st.subheader(label)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button("Download CSV", f, file_name=os.path.basename(path), mime="text/csv")
                prev = load_csv_safe(path)
                if prev is not None:
                    st.caption(f"Preview of {os.path.basename(path)}")
                    st.dataframe(prev.head(10), use_container_width=True)
            else:
                st.warning(f"Missing file: `{os.path.basename(path)}`")
