from datetime import datetime
from html import escape
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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

def prepare_dataframe_for_display(data):
    display_data = data.copy()
    if "Mine ID" in display_data.columns:
        display_data["Mine ID"] = display_data["Mine ID"].astype("string")

    numeric_columns = display_data.select_dtypes(include="number").columns
    display_data[numeric_columns] = display_data[numeric_columns].round(2)
    return display_data

def add_table(frame, columns, styles, page_size, story, numeric_columns=()):
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
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

def add_trendline(axis, data, trend_degree, color="orange"):
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

def make_chart(
    data,
    title,
    series_column,
    all_events,
    trend_degree,
    story,
    series_name_column=None,
):
    figure, axis = plt.subplots(figsize=(10, 3))
    if series_name_column is None:
        axis.plot(data["Date"], data["Final Production"], label=series_column, color="blue")
        add_trendline(axis, data, trend_degree)
    else:
        for (mine_id, mine_name), mine_data in data.groupby(["Mine ID", series_name_column]):
            label = f"{mine_name} (ID {mine_id:g})"
            mine_line, = axis.plot(
                mine_data["Date"],
                mine_data["Final Production"],
                label=label,
            )
            add_trendline(axis, mine_data, trend_degree, color=mine_line.get_color())

    chart_events = all_events
    if not all_events.empty:
        if series_name_column is None:
            chart_events = all_events[
                (all_events["Mine ID"] == "All")
                & (all_events["Mine Name"] == "Total Output")
            ]
        else:
            chart_events = all_events[all_events["Mine ID"] != "All"]

    if not chart_events.empty:
        anomaly_styles = {
            "High": ("red", "^", "Spike"),
            "Low": ("blue", "v", "Drop"),
        }
        for anomaly_type, (color, marker, label) in anomaly_styles.items():
            marked = chart_events[chart_events["Anomaly Type"] == anomaly_type]
            if not marked.empty:
                axis.scatter(
                    marked["Date"],
                    marked["Final Production"],
                    color=color,
                    marker=marker,
                    s=30,
                    label=f"{label} (detected)",
                    zorder=4,
                )

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Production")
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    event_labels = {"Spike (detected)", "Drop (detected)"}
    mine_items, event_items, trendline_items = [], [], []
    for handle, label in zip(handles, labels):
        if label in event_labels:
            event_items.append((handle, label))
        elif label == "Polynomial trendline":
            if not trendline_items:
                trendline_items.append((handle, label))
        else:
            mine_items.append((handle, label))
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
        textColor=colors.darkcyan,
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
        fontName="Courier",
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
        styles,
        page_size,
        story,
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

    event_frames = [frame for frame in anomaly_frames.values() if not frame.empty]
    all_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    story.append(PageBreak())
    story.append(Paragraph("Production Charts", styles["Heading1"]))
    make_chart(
        total_daily, "Total Daily Production", "Total Output",
        all_events, trend_degree, story,
    )
    make_chart(
        mine_daily, "Daily Production by Mine", "Mine",
        all_events, trend_degree, story, "Mine Name",
    )

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
        add_table(
            anomalies,
            anomaly_columns[method],
            styles,
            page_size,
            story,
            numeric_anomaly_columns,
        )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()

