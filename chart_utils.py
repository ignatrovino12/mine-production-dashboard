import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def add_polynomial_trendline(
    figure,
    data,
    degree,
    name,
    color,
    showlegend=True,
    legendrank=None,
):
    valid = data.dropna(subset=["Final Production"])
    if len(valid) <= degree:
        return False

    x = np.arange(len(valid))
    production = valid["Final Production"].to_numpy()
    trend = np.polyval(np.polyfit(x, production, degree), x)
    line = dict(color=color, dash="dash")
    trace = go.Scatter(
        x=valid["Date"],
        y=trend,
        mode="lines",
        name=name,
        showlegend=showlegend,
        line=line,
    )
    if legendrank is not None:
        trace.legendrank = legendrank

    figure.add_trace(trace)
    return True


def build_production_chart(
    data,
    scope,
    chart_type,
    anomalies,
    anomaly_method,
    trend_degree,
    selected_mine_id=None,
):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    figure = go.Figure()

    if scope == "All Individual Mines":
        trendline_added = False
        anomaly_legend_added = set()
        mines = data[["Mine ID", "Mine Name"]].drop_duplicates()
        anomaly_styles = {
            "High": ("red", "triangle-up", "Spike (detected)", 1001),
            "Low": ("cyan", "triangle-down", "Drop (detected)", 1002),
        }

        for index, (mine_id, mine_name) in enumerate(
            mines.itertuples(index=False, name=None)
        ):
            color = qualitative.Plotly[index % len(qualitative.Plotly)]
            mine_data = data[
                (data["Mine ID"] == mine_id) & (data["Mine Name"] == mine_name)
            ]

            if chart_type == "Line":
                figure.add_trace(go.Scatter(
                    x=mine_data["Date"],
                    y=mine_data["Final Production"],
                    mode="lines",
                    name=mine_name,
                    line=dict(color=color),
                    legendrank=index,
                ))
            else:
                figure.add_trace(go.Bar(
                    x=mine_data["Date"],
                    y=mine_data["Final Production"],
                    name=mine_name,
                    marker=dict(color=color),
                    legendrank=index,
                ))

            mine_anomalies = anomalies
            if not anomalies.empty:
                mine_anomalies = anomalies[
                    (anomalies["Mine ID"] == mine_id)
                    & (anomalies["Mine Name"] == mine_name)
                ]
            for anomaly_type, (marker_color, symbol, label, legendrank) in anomaly_styles.items():
                marked = mine_anomalies[
                    mine_anomalies["Anomaly Type"] == anomaly_type
                ]
                if marked.empty:
                    continue
                figure.add_trace(go.Scatter(
                    x=pd.to_datetime(marked["Date"]),
                    y=marked["Final Production"],
                    mode="markers",
                    name=label,
                    showlegend=anomaly_type not in anomaly_legend_added,
                    legendrank=legendrank,
                    legendgroup=label,
                    marker=dict(
                        color=marker_color,
                        size=12,
                        symbol=symbol,
                        line=dict(color="white", width=1),
                    ),
                ))
                anomaly_legend_added.add(anomaly_type)

            trendline_added = add_polynomial_trendline(
                figure,
                mine_data,
                trend_degree,
                "Polynomial trendline",
                color,
                showlegend=not trendline_added,
                legendrank=1000,
            ) or trendline_added

        if chart_type == "Stacked":
            figure.update_layout(barmode="stack")
    else:
        if chart_type == "Line":
            figure.add_trace(go.Scatter(
                x=data["Date"],
                y=data["Final Production"],
                mode="lines",
                name="Production",
            ))
        else:
            figure.add_trace(go.Bar(
                x=data["Date"],
                y=data["Final Production"],
                name="Production",
            ))

        if not anomalies.empty:
            mine_anomalies = anomalies
            if scope == "Individual Mine":
                mine_anomalies = anomalies[anomalies["Mine ID"] == selected_mine_id]
            else:
                mine_anomalies = anomalies[anomalies["Mine ID"] == "All"]

            if not mine_anomalies.empty:
                figure.add_trace(go.Scatter(
                    x=pd.to_datetime(mine_anomalies["Date"]),
                    y=mine_anomalies["Final Production"],
                    mode="markers",
                    name=f"{anomaly_method} anomalies",
                    marker=dict(color="red", size=10, symbol="circle"),
                ))

        add_polynomial_trendline(
            figure,
            data,
            trend_degree,
            f"Trendline (degree {trend_degree})",
            "orange",
        )

    figure.update_layout(
        xaxis_title="Date",
        yaxis_title="Production",
        hovermode="x unified",
    )
    return figure
