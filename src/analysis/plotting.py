import arcadia_pycolor as apc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from arcadia_pycolor.gradient import Gradient
from arcadia_pycolor.style_defaults import DEFAULT_FONT
from ete3 import NodeStyle, TextFace, TreeStyle
from plotly.subplots import make_subplots

# Set Plotly renderer for Quarto compatibility
pio.renderers.default = "plotly_mimetype+notebook_connected"


def tree_style_with_categorical_annotation(
    categories: dict[str, str],
    color_map: dict[str, str],
    highlight: list[str] | None = None,
) -> TreeStyle:
    if highlight is None:
        highlight = []

    def layout(node):
        node_style = NodeStyle()
        node_style["hz_line_width"] = 2
        node_style["vt_line_width"] = 2
        node_style["hz_line_color"] = "#666666"
        node_style["vt_line_color"] = "#666666"

        if node.is_leaf():
            is_highlighted = highlight and any(key in node.name for key in highlight)

            if is_highlighted:
                node_style["shape"] = "circle"
                node_style["size"] = 15

                matched_key = next((key for key in highlight if key in node.name), None)
                if matched_key:
                    text_face = TextFace(
                        f" {matched_key}", fsize=36, bold=False, ftype=DEFAULT_FONT
                    )
                    node.add_face(text_face, column=0, position="branch-right")
            else:
                node_style["shape"] = "circle"
                node_style["size"] = 15

            for key, category in categories.items():
                if key in node.name:
                    node_style["fgcolor"] = color_map.get(category, "#000000")
                    break
            else:
                node_style["fgcolor"] = "#000000"

        node.set_style(node_style)

    tree_style = TreeStyle()
    tree_style.layout_fn = layout
    tree_style.show_leaf_name = False
    tree_style.show_branch_length = False
    tree_style.show_scale = False
    tree_style.scale = 240
    tree_style.rotation = 90
    tree_style.margin_left = 20
    tree_style.margin_right = 20
    tree_style.margin_top = 20
    tree_style.margin_bottom = 20

    return tree_style


def tree_style_with_scalar_annotation(
    values: dict[str, float],
    gradient: Gradient,
    highlight: list[str] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> TreeStyle:
    all_values = list(values.values())
    if min_val is None:
        min_val = min(all_values)
    if max_val is None:
        max_val = max(all_values)

    color_mapping = {}
    for key, value in values.items():
        mapped_colors = gradient.map_values([value], min_value=min_val, max_value=max_val)
        color_mapping[key] = mapped_colors[0].hex_code

    if highlight is None:
        highlight = []

    def layout(node):
        node_style = NodeStyle()
        node_style["hz_line_width"] = 2
        node_style["vt_line_width"] = 2
        node_style["hz_line_color"] = "#333333"
        node_style["vt_line_color"] = "#333333"

        if node.is_leaf():
            is_highlighted = highlight and any(key in node.name for key in highlight)

            if is_highlighted:
                node_style["shape"] = "circle"
                node_style["size"] = 15

                matched_key = next((key for key in highlight if key in node.name), None)
                if matched_key:
                    text_face = TextFace(
                        f" {matched_key}", fsize=36, bold=False, ftype=DEFAULT_FONT
                    )
                    node.add_face(text_face, column=0, position="branch-right")
            else:
                node_style["shape"] = "circle"
                node_style["size"] = 15

            # Find matching key and set color based on scalar value
            for key in values.keys():
                if key in node.name:
                    node_style["fgcolor"] = color_mapping.get(key, "#000000")
                    break
            else:
                node_style["fgcolor"] = "#000000"

        node.set_style(node_style)

    tree_style = TreeStyle()
    tree_style.layout_fn = layout
    tree_style.show_leaf_name = False
    tree_style.show_branch_length = False
    tree_style.show_scale = False
    tree_style.scale = 240
    tree_style.rotation = 90
    tree_style.margin_left = 20
    tree_style.margin_right = 20
    tree_style.margin_top = 20
    tree_style.margin_bottom = 20

    return tree_style


def interactive_layer_weight_plot(
    df: pd.DataFrame,
    regression_df: pd.DataFrame,
    query_to_subfamily: dict[str, str],
    subfamily_colors: dict[str, str],
    num_layers: int = 22,
) -> go.Figure:
    """Create interactive plotly figure with dropdown to select layer weights."""

    queries = sorted(df["query"].unique().tolist())
    subfamilies = df["target_subfamily"].unique()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            (
                f'{query} (subfamily <span style="color:'
                f'{subfamily_colors[query_to_subfamily[query]]}">'
                f"{query_to_subfamily[query]}</span>)"
            )
            for query in queries
        ],
        horizontal_spacing=0.08,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="Tree distance (patristic)",
        y_title="Sequence weight",
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
    )

    traces_per_query = len(subfamilies) + 1
    num_traces_per_option = len(queries) * traces_per_query

    for option_idx in range(num_layers + 1):
        if option_idx == 0:
            weight_col = "median_weight"
            layer_label = "median"
            visible = True
        else:
            weight_col = f"layer_{option_idx - 1}_weight"
            layer_label = option_idx - 1
            visible = False

        for query_idx, query in enumerate(queries):
            query_data = df[df["query"] == query]

            for subfamily in subfamilies:
                subfamily_data = query_data[query_data["target_subfamily"] == subfamily]

                if not subfamily_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=subfamily_data["patristic_distance"],
                            y=subfamily_data[weight_col],
                            mode="markers",
                            marker=dict(color=subfamily_colors[subfamily], size=5, opacity=0.45),
                            name=subfamily,
                            legendgroup=subfamily,
                            showlegend=(query_idx == 0),
                            visible=visible,
                            customdata=list(
                                zip(
                                    subfamily_data["target"],
                                    [subfamily] * len(subfamily_data),
                                    strict=False,
                                )
                            ),
                            hovertemplate=(
                                "Target: %{customdata[0]}<br>Subfamily: %{customdata[1]}"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=query_idx + 1,
                    )

            reg_row = regression_df[
                (regression_df["query"] == query) & (regression_df["layer"] == layer_label)
            ]
            slope = reg_row["slope"].iloc[0]
            intercept = reg_row["intercept"].iloc[0]
            r_squared = reg_row["r_squared"].iloc[0]
            p_value = reg_row["p_value"].iloc[0]

            x_min = query_data["patristic_distance"].min()
            x_max = query_data["patristic_distance"].max()
            x_line = [x_min + i * (x_max - x_min) / 100 for i in range(101)]
            y_line = [slope * x + intercept for x in x_line]

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#555555", width=1.5),
                    name="Regression",
                    showlegend=False,
                    visible=visible,
                    customdata=[[r_squared, p_value, slope, intercept]] * 101,
                    hovertemplate=(
                        "R²: %{customdata[0]:.3f}<br>"
                        "p: %{customdata[1]:.3e}<br>"
                        "slope: %{customdata[2]:.3f}<br>"
                        "intercept: %{customdata[3]:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=query_idx + 1,
            )

    buttons = []
    for option_idx in range(num_layers + 1):
        visibility = [False] * ((num_layers + 1) * num_traces_per_option)

        start_idx = option_idx * num_traces_per_option
        end_idx = start_idx + num_traces_per_option
        for i in range(start_idx, end_idx):
            visibility[i] = True

        if option_idx == 0:
            label = "Median Weight"
            weight_col = "median_weight"
        else:
            label = f"Layer {option_idx - 1}"
            weight_col = f"layer_{option_idx - 1}_weight"

        y_min = df[weight_col].min()
        y_max = df[weight_col].max()
        y_range = [y_min * 0.90, y_max * 1.1]

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {"yaxis.range": y_range, "yaxis2.range": y_range, "yaxis3.range": y_range},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.2,
                yanchor="middle",
                pad=dict(t=0, b=0, l=2, r=2),
                bgcolor=apc.parchment.hex_code,
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=12),
            )
        ],
        width=800,
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.20,
            yanchor="middle",
        ),
        plot_bgcolor=apc.parchment.hex_code,
        paper_bgcolor=apc.parchment.hex_code,
    )

    x_min = df["patristic_distance"].min()
    x_max = df["patristic_distance"].max()
    x_diff = x_max - x_min
    fig.update_xaxes(range=[x_min - 0.05 * x_diff, x_max + 0.05 * x_diff], matches="x")

    median_y_min = df["median_weight"].min()
    median_y_max = df["median_weight"].max()
    median_y_diff = median_y_max - median_y_min
    median_y_range = [median_y_min - 0.05 * median_y_diff, median_y_max + 0.05 * median_y_diff]
    fig.update_yaxes(range=median_y_range)

    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=1)
    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=2)
    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=3)

    return fig
