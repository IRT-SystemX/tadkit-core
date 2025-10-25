from plotly.subplots import make_subplots
import plotly.graph_objs as go


def plot_raw_data(data):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines", name=col))
    return fig


def plot_double_data(data, scaled_anomalies):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for col in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[col], mode="lines", showlegend=False),
            row=1,
            col=1,
        )
    for col in scaled_anomalies.columns:
        fig.add_trace(
            go.Scatter(
                x=scaled_anomalies.index,
                y=scaled_anomalies[col],
                mode="lines",
                name=col,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        legend=dict(
            orientation="v",
            entrywidth=100,
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        width=1000,
        height=600,
    )

    return fig
