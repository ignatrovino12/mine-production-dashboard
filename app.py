import streamlit as st
import gspread
import pandas as pd
from pdf_utils import build_pdf_report, prepare_dataframe_for_display
from anomalies_utils import detect_iqr, detect_zscore, detect_moving_average, detect_grubbs
from chart_utils import build_production_chart

st.set_page_config(page_title="Mine Dashboard", layout="wide")
st.title("Mine Production Dashboard")

SHEET_ID = "1WkZl92-C1rTQNsfYDvFA8tFEabfPjswo1CSgCPShZnk"

@st.cache_resource
def connect_to_sheets():
    credentials = dict(st.secrets["gcp_service_account"])
    return gspread.service_account_from_dict(credentials)

try:
    refresh_column, generate_column, download_column, _ = st.columns([1.3, 1.8, 1.8, 8], gap="small",)
    with refresh_column:
        if st.button("Refresh data"):
            st.session_state.pop("mine_production_report_pdf", None)
            st.cache_data.clear()
            st.rerun()
    with generate_column:
        generate_report = st.button("Generate PDF report")
    with download_column:
        report_download_placeholder = st.empty()

    with st.spinner("Connecting to Google Sheets..."):
        client = connect_to_sheets()
        spreadsheet = client.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet("Data")
        rows = worksheet.get_all_values()

    df = pd.DataFrame(rows[1:], columns=rows[0])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Mine ID"] = pd.to_numeric(df["Mine ID"], errors="coerce")
    df["Final Production"] = pd.to_numeric(df["Final Production"].str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=["Date", "Mine ID", "Mine Name", "Final Production"])

    mine_daily = df.groupby(
        ["Date", "Mine ID", "Mine Name"],
        as_index=False)["Final Production"].sum(min_count=1)

    total_daily = mine_daily.groupby(
        "Date", as_index=False)["Final Production"].sum(min_count=1)

    mine_stats = mine_daily.groupby(
        ["Mine ID", "Mine Name"]
    )["Final Production"].agg(
        mean_daily_output="mean",
        standard_deviation="std",
        median="median",
        interquartile_range=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index()

    total_stats = pd.DataFrame([{
        "Mine ID": "All",
        "Mine Name": "Total Output",
        "mean_daily_output": total_daily["Final Production"].mean(),
        "standard_deviation": total_daily["Final Production"].std(),
        "median": total_daily["Final Production"].median(),
        "interquartile_range": total_daily["Final Production"].quantile(0.75) - total_daily["Final Production"].quantile(0.25),

    }])

    stats = pd.concat([mine_stats, total_stats], ignore_index=True)

    st.subheader("Production Statistics by Mine")
    st.dataframe(prepare_dataframe_for_display(stats), width="stretch", hide_index=True)

    st.subheader("Anomaly Detection")

    st.markdown("1. IQR rule")
    iqr_multiplier = st.number_input(
        "IQR multiplier (k)",
        min_value=0.1,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

    iqr_results = []
    iqr_anomalies = pd.DataFrame()

    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        anomalies = detect_iqr(
            mine_daily[mine_daily["Mine ID"] == mine_id],
            iqr_multiplier,
            mine_id,
            mine_name)
        
        if not anomalies.empty:
            iqr_results.append(anomalies)

    anomalies = detect_iqr(
        total_daily,
        iqr_multiplier,
        "All",
        "Total Output")
    
    if not anomalies.empty:
        iqr_results.append(anomalies)

    if iqr_results:
        iqr_anomalies = pd.concat(iqr_results, ignore_index=True)
        st.write(f"Detected anomalies: {len(iqr_anomalies)}")
        st.dataframe(
            prepare_dataframe_for_display(iqr_anomalies[
                [
                    "Date",
                    "Mine ID",
                    "Mine Name",
                    "Final Production",
                    "Anomaly Type",
                    "Lower Bound",
                    "Upper Bound"
                ]
            ]),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No anomalies detected using the IQR rule.")

    st.markdown("2. Z-score Anomaly Detection")

    z_threshold = st.number_input(
        "Z-score threshold",
        min_value=0.5,
        max_value=5.0,
        value=3.0,
        step=0.1
    )

    z_results = []
    z_anomalies = pd.DataFrame()

    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        anomalies = detect_zscore(
            mine_daily[mine_daily["Mine ID"] == mine_id],
            z_threshold,
            mine_id,
            mine_name
        )

        if not anomalies.empty:
            z_results.append(anomalies)

    anomalies = detect_zscore(
        total_daily,
        z_threshold,
        "All",
        "Total Output"
    )

    if not anomalies.empty:
        z_results.append(anomalies)

    if z_results:
        z_anomalies = pd.concat(z_results, ignore_index=True)
        st.write(f"Detected anomalies: {len(z_anomalies)}")
        st.dataframe(
            prepare_dataframe_for_display(z_anomalies[
                [
                    "Date",
                    "Mine ID",
                    "Mine Name",
                    "Final Production",
                    "Z-score",
                    "Anomaly Type"
                ]
            ]),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No anomalies detected using the Z-score method.")

    st.markdown("3. Moving Average Anomaly Detection")

    ma_window = st.number_input(
        "Moving average window (days)",
        min_value=2,
        max_value=30,
        value=7,
        step=1
    )

    ma_threshold = st.number_input(
        "Percentage distance threshold (%)",
        min_value=1.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    ma_results = []
    ma_anomalies = pd.DataFrame()

    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        anomalies = detect_moving_average(
            mine_daily[mine_daily["Mine ID"] == mine_id],
            ma_window,
            ma_threshold,
            mine_id,
            mine_name
        )

        if not anomalies.empty:
            ma_results.append(anomalies)

    anomalies = detect_moving_average(
        total_daily,
        ma_window,
        ma_threshold,
        "All",
        "Total Output"
    )

    if not anomalies.empty:
        ma_results.append(anomalies)

    if ma_results:
        ma_anomalies = pd.concat(ma_results, ignore_index=True)
        st.write(f"Detected anomalies: {len(ma_anomalies)}")
        st.dataframe(
            prepare_dataframe_for_display(ma_anomalies[
                [
                    "Date",
                    "Mine ID",
                    "Mine Name",
                    "Final Production",
                    "Moving Average",
                    "Distance (%)",
                    "Anomaly Type"
                ]
            ]),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No anomalies detected using the moving average method.")
    
    st.markdown("4. Grubbs' Test")

    alpha = st.number_input(
        "Significance level (alpha)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.3f"
    )

    grubbs_results = []
    grubbs_anomalies = pd.DataFrame()

    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        anomalies = detect_grubbs(
            mine_daily[mine_daily["Mine ID"] == mine_id],
            alpha,
            mine_id,
            mine_name
        )

        if not anomalies.empty:
            grubbs_results.append(anomalies)

    anomalies = detect_grubbs(
        total_daily,
        alpha,
        "All",
        "Total Output"
    )

    if not anomalies.empty:
        grubbs_results.append(anomalies)

    if grubbs_results:
        grubbs_anomalies = pd.concat(
            grubbs_results,
            ignore_index=True
        )
        st.write(f"Detected anomalies: {len(grubbs_anomalies)}")
        st.dataframe(
            prepare_dataframe_for_display(grubbs_anomalies[
                [
                    "Date",
                    "Mine ID",
                    "Mine Name",
                    "Final Production",
                    "Anomaly Type",
                ]
            ]),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No anomalies detected using Grubbs' test.")

    st.subheader("Production Charts")

    chart_type = st.selectbox( "Chart type", ["Line", "Bar", "Stacked"])
    trend_degree = st.selectbox( "Polynomial trendline degree", [1, 2, 3, 4])
    chart_scope = st.selectbox( "Production data", ["Total Output", "Individual Mine", "All Individual Mines"])
    anomaly_method = st.selectbox( "Highlight anomalies from", ["IQR", "Z-score", "Moving Average", "Grubbs"])

    anomaly_frames = {
        "IQR": iqr_anomalies,
        "Z-score": z_anomalies,
        "Moving Average": ma_anomalies,
        "Grubbs": grubbs_anomalies
    }

    selected_anomalies = anomaly_frames[anomaly_method]

    if chart_scope == "Total Output":
        chart_data = total_daily.copy()
    elif chart_scope == "Individual Mine":
        mine_options = mine_daily[["Mine ID", "Mine Name"]].drop_duplicates()
        mine_choices = mine_options.to_dict("records")
        selected_mine = st.selectbox(
            "Select mine",
            mine_choices,
            format_func=lambda mine: f"{mine['Mine Name']} (ID {mine['Mine ID']:g})"
        )
        selected_id = selected_mine["Mine ID"]
        chart_data = mine_daily[mine_daily["Mine ID"] == selected_id].copy()
    elif chart_scope == "All Individual Mines":
        chart_data = mine_daily.copy()

    fig = build_production_chart(
        chart_data,
        chart_scope,
        chart_type,
        selected_anomalies,
        anomaly_method,
        trend_degree,
        selected_id if chart_scope == "Individual Mine" else None,
    )

    st.plotly_chart(fig, width="stretch")

    if generate_report:
        try:
            st.session_state["mine_production_report_pdf"] = build_pdf_report(
                stats,
                total_daily,
                mine_daily,
                anomaly_frames,
                iqr_multiplier,
                z_threshold,
                ma_window,
                ma_threshold,
                alpha,
                trend_degree,

            )
        except Exception as report_error:
            st.error(f"Could not generate the PDF report: {report_error}")

    report_pdf = st.session_state.get("mine_production_report_pdf")
    if report_pdf:
        report_download_placeholder.download_button(
            "Download PDF report",
            data=report_pdf,
            file_name="Weyland-Yutani Corporation mines.pdf",
            mime="application/pdf",
            key="download_mine_production_report",
        )

except Exception as e:
    st.error(f"Connection failed: {e}")
