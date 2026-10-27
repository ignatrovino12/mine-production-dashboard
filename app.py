import streamlit as st
import gspread
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from html import escape
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import zscore
from scipy.stats import t
import numpy as np
from outliers import smirnov_grubbs as grubbs
import plotly.graph_objects as go
from plotly.colors import qualitative


def prepare_dataframe_for_display(data):
    display_data = data.copy()
    if "Mine ID" in display_data.columns:
        display_data["Mine ID"] = display_data["Mine ID"].astype("string")

    numeric_columns = display_data.select_dtypes(include="number").columns
    display_data[numeric_columns] = display_data[numeric_columns].round(2)
    return display_data


def build_pdf_report(
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
):
    page_size = landscape(A4)
    buffer = BytesIO()
    report_title = "Mine Production Dashboard Report"
    document = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        title=report_title,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
    ))
    styles.add(ParagraphStyle(
        name="SmallCell",
        parent=styles["BodyText"],
        fontSize=7,
        leading=8,
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["SmallCell"],
        textColor=colors.white,
        fontName="Helvetica-Bold",
    ))
    story = [
        Paragraph(report_title, styles["ReportTitle"]),
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"| Data range: {total_daily['Date'].min():%Y-%m-%d} to "
            f"{total_daily['Date'].max():%Y-%m-%d}",
            styles["Normal"],
        ),
        Spacer(1, 8),
        Paragraph("Production Statistics", styles["Heading1"]),
    ]

    def add_table(frame, columns, numeric_columns=()):
        if frame.empty:
            story.append(Paragraph("No records.", styles["Normal"]))
            return

        available_columns = [column for column in columns if column in frame.columns]
        table_data = [[
            Paragraph(escape(str(column)), styles["TableHeader"])
            for column in available_columns
        ]]
        for _, row in frame[available_columns].iterrows():
            cells = []
            for column, value in row.items():
                if pd.isna(value):
                    text = ""
                elif column == "Date":
                    text = pd.to_datetime(value).strftime("%Y-%m-%d")
                elif column in numeric_columns:
                    text = f"{value:,.2f}" if isinstance(value, (int, float, np.number)) else str(value)
                else:
                    text = str(value)
                cells.append(Paragraph(escape(text), styles["SmallCell"]))
            table_data.append(cells)

        usable_width = page_size[0] - 30 * mm
        table = Table(
            table_data,
            colWidths=[usable_width / len(available_columns)] * len(available_columns),
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F7")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Daily output summary by mine and total output.",
        styles["BodyText"],
    ))
    add_table(
        stats,
        [
            "Mine ID",
            "Mine Name",
            "mean_daily_output",
            "standard_deviation",
            "median",
            "interquartile_range",
        ],
        numeric_columns={
            "Mine ID",
            "mean_daily_output",
            "standard_deviation",
            "median",
            "interquartile_range",
        },
    )

    story.append(Paragraph("Detection Settings", styles["Heading1"]))
    story.append(Paragraph(
        f"IQR multiplier: {iqr_multiplier:g} | Z-score threshold: {z_threshold:g} | "
        f"Moving-average window: {ma_window} days | Moving-average distance: "
        f"{ma_threshold:g}% | Grubbs significance level: {alpha:g} | "
        f"Trendline degree: {trend_degree}",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 8))

    def add_trendline(axis, data, color="orange"):
        valid = data.dropna(subset=["Final Production"]).sort_values("Date")
        if len(valid) <= trend_degree:
            return
        x = np.arange(len(valid))
        values = valid["Final Production"].to_numpy()
        trend = np.polyval(np.polyfit(x, values, trend_degree), x)
        axis.plot(
            valid["Date"],
            trend,
            color=color,
            linestyle="--",
            label="Polynomial trendline",
        )

    def make_chart(data, title, series_column, series_name_column=None):
        figure, axis = plt.subplots(figsize=(10, 3))
        if series_name_column is None:
            axis.plot(data["Date"], data["Final Production"], label=series_column, color="#2878B5")
            add_trendline(axis, data)
            chart_events = all_events[
                (all_events["Mine ID"] == "All")
                & (all_events["Mine Name"] == "Total Output")
            ] if not all_events.empty else all_events
        else:
            for (mine_id, mine_name), mine_data in data.groupby(["Mine ID", series_name_column]):
                label = f"{mine_name} (ID {mine_id:g})"
                mine_line, = axis.plot(
                    mine_data["Date"],
                    mine_data["Final Production"],
                    label=label,
                )
                add_trendline(axis, mine_data, color=mine_line.get_color())
            chart_events = all_events[
                (all_events["Mine ID"] != "All")
            ] if not all_events.empty else all_events

        if not chart_events.empty:
            for anomaly_type, color in (("High", "#C00000"), ("Low", "#1F77B4")):
                marked = chart_events[chart_events["Anomaly Type"] == anomaly_type]
                if not marked.empty:
                    axis.scatter(
                        marked["Date"],
                        marked["Final Production"],
                        color=color,
                        marker="^" if anomaly_type == "High" else "v",
                        s=30,
                        label=f"{'Spike' if anomaly_type == 'High' else 'Drop'} (detected)",
                        zorder=4,
                    )

        axis.set_title(title)
        axis.set_xlabel("Date")
        axis.set_ylabel("Production")
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        event_labels = {"Spike (detected)", "Drop (detected)"}
        mine_items = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label not in event_labels and label != "Polynomial trendline"
        ]
        event_items = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label in event_labels
        ]
        trendline_items = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label == "Polynomial trendline"
        ][:1]
        legend_items = mine_items + event_items + trendline_items
        if legend_items:
            axis.legend(
                [handle for handle, _ in legend_items],
                [label for _, label in legend_items],
                loc="best",
                fontsize=7,
            )
        figure.autofmt_xdate()
        figure.tight_layout()
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=140)
        plt.close(figure)
        image_buffer.seek(0)
        story.append(Image(
            image_buffer,
            width=245 * mm,
            height=73.5 * mm,
            hAlign="CENTER",
        ))
        story.append(Spacer(1, 4))

    all_events = []
    for method, anomalies in anomaly_frames.items():
        if anomalies.empty:
            continue
        events = anomalies.copy()
        events["Detection Method"] = method
        all_events.append(events)
    all_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    story.append(PageBreak())
    story.append(Paragraph("Production Charts", styles["Heading1"]))
    make_chart(total_daily, "Total Daily Production", "Total Output")
    make_chart(mine_daily, "Daily Production by Mine", "Mine", "Mine Name")

    story.append(PageBreak())
    story.append(Paragraph("Anomaly Detection Results", styles["Heading1"]))
    anomaly_columns = {
        "IQR": ["Date", "Mine ID", "Mine Name", "Final Production", "Anomaly Type", "Lower Bound", "Upper Bound"],
        "Z-score": ["Date", "Mine ID", "Mine Name", "Final Production", "Z-score", "Anomaly Type"],
        "Moving Average": ["Date", "Mine ID", "Mine Name", "Final Production", "Moving Average", "Distance (%)", "Anomaly Type"],
        "Grubbs": ["Date", "Mine ID", "Mine Name", "Final Production", "Anomaly Type"],
    }
    numeric_anomaly_columns = {
        "Mine ID", "Final Production", "Lower Bound", "Upper Bound",
        "Z-score", "Moving Average", "Distance (%)",
    }
    for method, anomalies in anomaly_frames.items():
        story.append(Paragraph(
            f"{escape(method)} ({len(anomalies)} detected)",
            styles["Heading2"],
        ))
        add_table(anomalies, anomaly_columns[method], numeric_anomaly_columns)

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.set_page_config(page_title="Mine Dashboard", layout="wide")
st.title("Mine Production Dashboard")

SHEET_ID = "1WkZl92-C1rTQNsfYDvFA8tFEabfPjswo1CSgCPShZnk"

@st.cache_resource
def connect_to_sheets():
    credentials = dict(st.secrets["gcp_service_account"])
    return gspread.service_account_from_dict(credentials)

try:
    refresh_column, generate_column, download_column, _ = st.columns(
        [1.3, 1.8, 1.8, 8],
        gap="small",
    )
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
        worksheet = spreadsheet.worksheet("Data_copy")
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

    # Each
    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        data = mine_daily[mine_daily["Mine ID"] == mine_id].copy()
        values = data["Final Production"]

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr_value = q3 - q1
        lower = q1 - iqr_multiplier * iqr_value
        upper = q3 + iqr_multiplier * iqr_value

        anomalies = data[(values < lower) | (values > upper)].copy()

        if not anomalies.empty:
            anomalies["Mine ID"] = mine_id
            anomalies["Mine Name"] = mine_name
            anomalies["Anomaly Type"] = anomalies["Final Production"].apply(lambda x: "Low" if x < lower else "High")
            anomalies["Lower Bound"] = lower
            anomalies["Upper Bound"] = upper

            iqr_results.append(anomalies)

    # Total
    data = total_daily.copy()
    values = data["Final Production"]

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr_value = q3 - q1
    lower = q1 - iqr_multiplier * iqr_value
    upper = q3 + iqr_multiplier * iqr_value

    anomalies = data[(values < lower) | (values > upper)].copy()

    if not anomalies.empty:
        anomalies["Mine ID"] = "All"
        anomalies["Mine Name"] = "Total Output"
        anomalies["Anomaly Type"] = anomalies["Final Production"].apply( lambda x: "Low" if x < lower else "High")
        anomalies["Lower Bound"] = lower
        anomalies["Upper Bound"] = upper

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

    # Each
    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        data = mine_daily[mine_daily["Mine ID"] == mine_id].copy()
        values = data["Final Production"]

        scores = pd.Series(zscore(values, nan_policy="omit"),index=data.index)
        anomalies = data[abs(scores) > z_threshold].copy()

        if not anomalies.empty:
            anomalies["Mine ID"] = mine_id
            anomalies["Mine Name"] = mine_name
            anomalies["Z-score"] = scores.loc[anomalies.index]
            anomalies["Anomaly Type"] = anomalies["Z-score"].apply( lambda x: "Low" if x < 0 else "High")
            z_results.append(anomalies)

    # Total
    data = total_daily.copy()
    values = data["Final Production"]

    scores = pd.Series(zscore(values, nan_policy="omit"),index=data.index)
    anomalies = data[abs(scores) > z_threshold].copy()

    if not anomalies.empty:
        anomalies["Mine ID"] = "All"
        anomalies["Mine Name"] = "Total Output"
        anomalies["Z-score"] = scores.loc[anomalies.index]
        anomalies["Anomaly Type"] = anomalies["Z-score"].apply(lambda x: "Low" if x < 0 else "High")
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

    # Each
    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        data = mine_daily[mine_daily["Mine ID"] == mine_id].copy().sort_values("Date")

        data["Moving Average"] = (
            data.set_index("Date")["Final Production"]
            .rolling(f"{ma_window}D", closed="left", min_periods=1)
            .mean()
            .to_numpy()
        )
        data["Distance (%)"] = (
            (data["Final Production"] - data["Moving Average"])
            .div(data["Moving Average"].abs().where(data["Moving Average"].ne(0)))
            .mul(100)
        )

        anomalies = data[data["Distance (%)"].abs() > ma_threshold].copy()

        if not anomalies.empty:
            anomalies["Mine ID"] = mine_id
            anomalies["Mine Name"] = mine_name
            anomalies["Anomaly Type"] = anomalies["Distance (%)"].apply(lambda x: "Low" if x < 0 else "High")

            ma_results.append(anomalies)

    # Total
    data = total_daily.copy().sort_values("Date")

    data["Moving Average"] = (
        data.set_index("Date")["Final Production"]
        .rolling(f"{ma_window}D", closed="left", min_periods=1)
        .mean()
        .to_numpy()
    )

    data["Distance (%)"] = (
        (data["Final Production"] - data["Moving Average"])
        / data["Moving Average"].abs().where(data["Moving Average"].ne(0))
        * 100
    )

    anomalies = data[
        data["Distance (%)"].abs() > ma_threshold
    ].copy()

    if not anomalies.empty:
        anomalies["Mine ID"] = "All"
        anomalies["Mine Name"] = "Total Output"

        anomalies["Anomaly Type"] = anomalies["Distance (%)"].apply(
            lambda x: "Low" if x < 0 else "High"
        )

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

    # Each
    for mine_id, mine_name in mine_daily[["Mine ID", "Mine Name"]].drop_duplicates().itertuples(index=False, name=None):

        data = mine_daily[mine_daily["Mine ID"] == mine_id].copy()

        values = data["Final Production"].to_numpy()
        outlier_indices = (
            grubbs.two_sided_test_indices(values, alpha=alpha)
            if len(values) >= 3 and np.std(values) > 0
            else []
        )
        anomalies = data.iloc[outlier_indices].copy()

        if not anomalies.empty:
            anomalies["Mine ID"] = mine_id
            anomalies["Mine Name"] = mine_name
            anomalies["Anomaly Type"] = anomalies["Final Production"].apply(
                lambda value: "Low" if value < data["Final Production"].mean() else "High"
            )
            grubbs_results.append(anomalies)

    # Total
    data = total_daily.copy()
    values = data["Final Production"].to_numpy()
    outlier_indices = (
        grubbs.two_sided_test_indices(values, alpha=alpha)
        if len(values) >= 3 and np.std(values) > 0
        else []
    )
    anomalies = data.iloc[outlier_indices].copy()

    if not anomalies.empty:
        anomalies["Mine ID"] = "All"
        anomalies["Mine Name"] = "Total Output"
        anomalies["Anomaly Type"] = anomalies["Final Production"].apply(
            lambda value: "Low" if value < data["Final Production"].mean() else "High"
        )
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

    chart_data["Date"] = pd.to_datetime(chart_data["Date"])
    chart_data = chart_data.sort_values("Date")

    fig = go.Figure()

    if chart_scope == "All Individual Mines":
        trendline_added = False
        anomaly_legend_added = set()
        mine_options = chart_data[["Mine ID", "Mine Name"]].drop_duplicates()
        for mine_index, (mine_id, mine_name) in enumerate(
            mine_options.itertuples(index=False, name=None)
        ):
            mine_color = qualitative.Plotly[mine_index % len(qualitative.Plotly)]
            mine_data = chart_data[
                (chart_data["Mine ID"] == mine_id)
                & (chart_data["Mine Name"] == mine_name)
            ].sort_values("Date")

            if chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=mine_data["Date"],
                    y=mine_data["Final Production"],
                    mode="lines",
                    name=mine_name,
                    line=dict(color=mine_color),
                    legendrank=mine_index,
                ))
            else:
                fig.add_trace(go.Bar(
                    x=mine_data["Date"],
                    y=mine_data["Final Production"],
                    name=mine_name,
                    marker=dict(color=mine_color),
                    legendrank=mine_index,
                ))

            highlights = selected_anomalies
            if not highlights.empty:
                highlights = highlights[
                    (highlights["Mine ID"] == mine_id)
                    & (highlights["Mine Name"] == mine_name)
                ]
            for anomaly_type, color, symbol, label, legendrank in (
                ("High", "#C00000", "triangle-up", "Spike (detected)", 1001),
                ("Low", "#1F77B4", "triangle-down", "Drop (detected)", 1002),
            ):
                marked = highlights[highlights["Anomaly Type"] == anomaly_type]
                if not marked.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(marked["Date"]),
                        y=marked["Final Production"],
                        mode="markers",
                        name=label,
                        showlegend=anomaly_type not in anomaly_legend_added,
                        legendrank=legendrank,
                        legendgroup=label,
                        marker=dict(
                            color=color,
                            size=12,
                            symbol=symbol,
                            line=dict(color="white", width=1),
                        ),
                    ))
                    anomaly_legend_added.add(anomaly_type)

            valid = mine_data.dropna(subset=["Final Production"])
            if len(valid) > trend_degree:
                x = np.arange(len(valid))
                y = valid["Final Production"].to_numpy()
                coefficients = np.polyfit(x, y, trend_degree)
                trend = np.polyval(coefficients, x)
                fig.add_trace(go.Scatter(
                    x=valid["Date"],
                    y=trend,
                    mode="lines",
                    name="Polynomial trendline",
                    showlegend=not trendline_added,
                    legendrank=1000,
                    line=dict(color=mine_color, dash="dash")
                ))
                trendline_added = True

        if chart_type == "Stacked":
            fig.update_layout(barmode="stack")
    else:
        fig.add_trace(
            go.Scatter(
                x=chart_data["Date"],
                y=chart_data["Final Production"],
                mode="lines" if chart_type == "Line" else "markers",
                name="Production"
            ) if chart_type == "Line" else
            go.Bar(
                x=chart_data["Date"],
                y=chart_data["Final Production"],
                name="Production"
            )
        )

        if not selected_anomalies.empty:
            highlights = selected_anomalies.copy()

            if chart_scope == "Individual Mine":
                highlights = highlights[highlights["Mine ID"] == selected_id]
            else:
                highlights = highlights[highlights["Mine ID"] == "All"]

            if not highlights.empty:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(highlights["Date"]),
                    y=highlights["Final Production"],
                    mode="markers",
                    name=f"{anomaly_method} anomalies",
                    marker=dict(color="red", size=10, symbol="circle")
                ))

        valid = chart_data.dropna(subset=["Final Production"])
        if len(valid) > trend_degree:
            x = np.arange(len(valid))
            y = valid["Final Production"].to_numpy()
            coefficients = np.polyfit(x, y, trend_degree)
            trend = np.polyval(coefficients, x)

            fig.add_trace(go.Scatter(
                x=valid["Date"],
                y=trend,
                mode="lines",
                name=f"Trendline (degree {trend_degree})",
                line=dict(color="orange", dash="dash")
            ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Production",
        hovermode="x unified"
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
