from typing import Literal

import arcadia_pycolor as apc
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from arcadia_pycolor.gradient import Gradient
from arcadia_pycolor.style_defaults import DEFAULT_FONT
from ete3 import NodeStyle, TextFace, TreeStyle
from plotly.subplots import make_subplots

# Set Plotly renderer for Quarto compatibility
pio.renderers.default = "plotly_mimetype+notebook_connected"


def gradient_from_listed_colormap(
    cmap: mcolors.ListedColormap, name: str | None = None
) -> Gradient:
    """Converts a matplotlib ListedColormap to an arcadia_pycolor Gradient."""
    if name is None:
        name = cmap.name if hasattr(cmap, "name") else "gradient"

    colors = cmap.colors

    hex_codes = [
        apc.HexCode(name=f"{name}_{i}", hex_code=mcolors.to_hex(color))
        for i, color in enumerate(colors)
    ]

    return Gradient(name=name, colors=hex_codes)


def tree_style_with_highlights(
    highlight: list[str] | None = None,
    highlight_color: str = "#FF6B6B",
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
            is_highlighted = any(key in node.name for key in highlight)

            if is_highlighted:
                node_style["shape"] = "circle"
                node_style["size"] = 8
                node_style["fgcolor"] = highlight_color

                matched_key = next((key for key in highlight if key in node.name), None)
                if matched_key:
                    text_face = TextFace(
                        f" {matched_key}", fsize=36, bold=False, ftype=DEFAULT_FONT
                    )
                    node.add_face(text_face, column=0, position="branch-right")
            else:
                node_style["shape"] = "circle"
                node_style["size"] = 8
                node_style["fgcolor"] = "#CCCCCC"

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
        x_title="Sequence weight",
        y_title="Tree distance (patristic)",
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
                            x=subfamily_data[weight_col],
                            y=subfamily_data["patristic_distance"],
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

            x_min = query_data[weight_col].min()
            x_max = query_data[weight_col].max()
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

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
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
                bgcolor=apc.white.hex_code,
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
        plot_bgcolor=apc.white.hex_code,
        paper_bgcolor=apc.white.hex_code,
    )

    y_min = df["patristic_distance"].min()
    y_max = df["patristic_distance"].max()
    y_diff = y_max - y_min
    fig.update_yaxes(range=[y_min - 0.05 * y_diff, y_max + 0.05 * y_diff], matches="y")

    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=1)
    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=2)
    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=3)

    return fig


def ridgeline_r2_plot(
    df: pd.DataFrame,
    size_col: str = "Size",
    y_col: str = "Adjusted R2",
    gradient=None,
    bw_adjust: float = 0.11,
    gap: float = 0.7,
):
    from scipy.stats import gaussian_kde

    if gradient is None:
        gradient = apc.gradients.sunset.reverse()

    df_copy = df.copy()

    bin_edges = [200, 300, 400, 500, 600, 700, 800, 900, 1025]
    bin_labels = [
        "200-299",
        "300-399",
        "400-499",
        "500-599",
        "600-699",
        "700-799",
        "800-899",
        "900-1024",
    ]

    df_copy["Size Bin"] = pd.cut(df_copy[size_col], bins=bin_edges, labels=bin_labels, right=False)
    df_copy = df_copy.dropna(subset=["Size Bin"])

    actual_n_bins = len(df_copy["Size Bin"].unique())
    colors = [c.hex_code for c in gradient.resample_as_palette(actual_n_bins)]

    bins = sorted(df_copy["Size Bin"].unique())

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.25, 0.75],
        horizontal_spacing=0.25,
        shared_yaxes=True,
    )

    bin_counts = []
    bin_offsets = []

    x_min = 0.3
    x_max = 1.0
    x_range = np.linspace(x_min, x_max, 200)

    for i, bin_label in enumerate(bins):
        offset = i * gap
        bin_offsets.append(offset)
        bin_counts.append(len(df_copy[df_copy["Size Bin"] == bin_label]))
        bin_data = df_copy[df_copy["Size Bin"] == bin_label]
        values = bin_data[y_col].values

        mean_val = values.mean() if len(values) > 0 else 0
        std_val = values.std() if len(values) > 0 else 0

        if len(values) > 1:
            kde = gaussian_kde(values, bw_method=bw_adjust)
            density = kde(x_range)

            max_density = density.max()
            if max_density > 0:
                normalized_density = density / max_density
            else:
                normalized_density = density
        else:
            normalized_density = np.zeros_like(x_range)

        y_values = normalized_density + bin_offsets[i]

        hex_color = colors[i].lstrip("#")
        r, g_val, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        rgba_solid = f"rgba({r},{g_val},{b},0.9)"
        rgba_medium = f"rgba({r},{g_val},{b},0.2)"
        rgba_transparent = f"rgba({r},{g_val},{b},0.0)"

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=np.full_like(x_range, bin_offsets[i]),
                mode="lines",
                line=dict(color=colors[i], width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_values,
                mode="lines",
                line=dict(color="#444444", width=2),
                fill="tonexty",
                fillgradient=dict(
                    type="vertical",
                    colorscale=[
                        (0.0, rgba_transparent),
                        (0.1, rgba_medium),
                        (0.8, rgba_solid),
                        (1.0, rgba_solid),
                    ],
                ),
                showlegend=False,
                name=bin_label,
                hovertemplate=(
                    f"MSA depth: {bin_label}<br>Mean ± SD: "
                    f"{mean_val:.2f} ± {std_val:.2f}<extra></extra>",
                ),
            ),
            row=1,
            col=2,
        )

    for i in range(len(bins)):
        bar_height = gap * 0.8
        y_bottom = bin_offsets[i] - bar_height / 2
        y_top = bin_offsets[i] + bar_height / 2

        hex_color = colors[i].lstrip("#")
        r, g_val, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        rgba_opaque = f"rgba({r},{g_val},{b},1.0)"
        rgba_semi = f"rgba({r},{g_val},{b},0.5)"

        fig.add_trace(
            go.Scatter(
                x=[0, bin_counts[i], bin_counts[i], 0, 0],
                y=[y_bottom, y_bottom, y_top, y_top, y_bottom],
                fill="toself",
                fillgradient=dict(
                    type="horizontal",
                    colorscale=[(0, rgba_opaque), (1, rgba_semi)],
                ),
                mode="lines",
                line=dict(color="#444444", width=2),
                showlegend=False,
                name=str(bin_counts[i]),
                hoverinfo="name",
            ),
            row=1,
            col=1,
        )

    for i in range(len(bin_offsets)):
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=bin_offsets[i],
            y1=bin_offsets[i],
            xref="paper",
            yref="y",
            line=dict(color="rgba(150, 150, 150, 0.7)", width=0.75),
            layer="below",
        )

    fig.add_shape(
        type="rect",
        x0=0.23,
        x1=0.405,
        y0=0.03,
        y1=0.86,
        xref="paper",
        yref="paper",
        fillcolor=apc.parchment.hex_code,
        line=dict(color="black", width=1),
        layer="below",
    )

    for i, bin_label in enumerate(bins):
        label_color = "#888888" if (i == 0 and gradient.name == "verde_r") else colors[i]
        fig.add_annotation(
            x=0.317,
            y=bin_offsets[i],
            xref="paper",
            yref="y",
            text=bin_label,
            showarrow=False,
            font=dict(size=20, color=label_color, family="SuisseIntlMono"),
            xanchor="center",
            yanchor="middle",
        )

    fig.add_annotation(
        x=0.317,
        y=0.86,
        xref="paper",
        yref="paper",
        text="MSA depth",
        showarrow=False,
        font=dict(size=22, color="black", family="SuisseIntl-Medium"),
        xanchor="center",
        yanchor="bottom",
    )

    fig.update_xaxes(
        title=dict(text="Count", font=dict(family="SuisseIntl-Medium", size=20)),
        autorange="reversed",
        showline=True,
        linewidth=2.5,
        showgrid=False,
        ticks="outside",
        tickwidth=2.5,
        tickfont=dict(size=16),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title=dict(text=y_col.replace("R2", "R²"), font=dict(family="SuisseIntl-Medium", size=20)),
        range=[x_min, x_max],
        showline=True,
        linewidth=2.5,
        linecolor="black",
        ticks="outside",
        tickwidth=2.5,
        tickfont=dict(size=16),
        row=1,
        col=2,
    )

    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showline=False,
        showgrid=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showline=False,
        showgrid=False,
        row=1,
        col=2,
    )

    fig.update_layout(
        width=800,
        height=400,
        plot_bgcolor=apc.white.hex_code,
        paper_bgcolor=apc.white.hex_code,
        showlegend=False,
        margin=dict(l=80, r=40, t=40, b=80),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
    )

    apc.plotly.style_plot(fig, monospaced_axes="x", row=1, col=1)
    apc.plotly.style_plot(fig, monospaced_axes="x", row=1, col=2)

    return fig


def stacked_feature_importance_plot(
    df: pd.DataFrame,
    size_col: str = "Size",
    feature_cols: list[int] | None = None,
    gap: float = -0.78,
    gradient=None,
):
    if feature_cols is None:
        feature_cols = list(range(22))

    if gradient is None:
        gradient = apc.gradients.sunset.reverse()

    df_copy = df.copy()

    bin_edges = [200, 300, 400, 500, 600, 700, 800, 900, 1025]
    bin_labels = [
        "200-299",
        "300-399",
        "400-499",
        "500-599",
        "600-699",
        "700-799",
        "800-899",
        "900-1024",
    ]

    df_copy["Size Bin"] = pd.cut(df_copy[size_col], bins=bin_edges, labels=bin_labels, right=False)
    df_copy = df_copy.dropna(subset=["Size Bin"])

    actual_n_bins = len(df_copy["Size Bin"].unique())
    colors = [c.hex_code for c in gradient.resample_as_palette(actual_n_bins)]

    bins = sorted(df_copy["Size Bin"].unique())
    feature_indices = np.arange(len(feature_cols))

    all_mean_values = []
    for bin_label in bins:
        bin_data = df_copy[df_copy["Size Bin"] == bin_label]
        mean_values = bin_data[feature_cols].mean().values
        all_mean_values.append(mean_values)

    all_mean_values_array = np.array(all_mean_values)
    global_min = all_mean_values_array.min()
    global_max = all_mean_values_array.max()

    fig = go.Figure()

    for i in reversed(range(len(bins))):
        bin_label = bins[i]
        bin_data = df_copy[df_copy["Size Bin"] == bin_label]

        mean_values = bin_data[feature_cols].mean().values
        std_values = bin_data[feature_cols].std().values
        n_samples = len(bin_data)
        se_values = std_values / np.sqrt(n_samples)

        percentage_values = (mean_values / mean_values.sum()) * 100
        se_percentage_values = (se_values / mean_values.sum()) * 100

        if global_max > global_min:
            scaled_values = (mean_values - global_min) / (global_max - global_min)
        else:
            scaled_values = np.zeros_like(mean_values)

        offset = i * (1.0 + gap)
        y_values = scaled_values + offset

        hover_data = np.column_stack([percentage_values, se_percentage_values])

        hex_color = colors[i].lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        rgba_solid = f"rgba({r},{g},{b},0.9)"
        rgba_medium = f"rgba({r},{g},{b},0.5)"
        rgba_transparent = f"rgba({r},{g},{b},0.0)"

        line_gray_value = int(102 * (1 - i / (len(bins) - 1)))
        line_color = f"#{line_gray_value:02x}{line_gray_value:02x}{line_gray_value:02x}"

        fig.add_trace(
            go.Scatter(
                x=feature_indices,
                y=np.full_like(feature_indices, offset, dtype=float),
                mode="lines",
                line=dict(color=colors[i], width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=feature_indices,
                y=y_values,
                mode="lines",
                line=dict(color=line_color, width=2.0),
                fill="tonexty",
                fillgradient=dict(
                    type="vertical",
                    colorscale=[
                        (0.0, rgba_transparent),
                        (0.1, rgba_medium),
                        (0.5, rgba_solid),
                        (1.0, rgba_solid),
                    ],
                ),
                showlegend=False,
                hovertemplate=(
                    f"MSA depth: {bin_label}<br>Layer: "
                    f"%{{x}}<br>Importance: %{{customdata[0]:.1f}} "
                    f"± %{{customdata[1]:.1f}}%<extra></extra>"
                ),
                customdata=hover_data,
            )
        )

    bin_label_annotations = []
    label_offset = 0.1
    for i in range(len(bins)):
        bin_label = bins[i]
        offset = i * (1.0 + gap) + label_offset
        hex_color = colors[i].lstrip("#")
        label_color = "#888888" if (i == 0 and gradient.name == "verde_r") else f"#{hex_color}"

        bin_label_annotations.append(
            dict(
                text=bin_label,
                x=1.01,
                y=offset,
                xref="paper",
                yref="y",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntlMono", size=20, color=label_color),
            )
        )

    y_max = (len(bins) - 1) * (1.0 + gap) + 1.0
    num_segments = 50

    shapes = []
    for x_val in range(len(feature_cols)):
        y_positions = np.linspace(0, y_max, num_segments + 1)
        for j in range(num_segments):
            y_start = y_positions[j]
            y_end = y_positions[j + 1]
            y_mid = (y_start + y_end) / 2
            y_fraction = y_mid / y_max

            if y_fraction <= 0.6:
                opacity = 0.25
            elif y_fraction <= 0.8:
                opacity = 0.25 * (1 - (y_fraction - 0.6) / 0.2)
            else:
                opacity = 0.0

            if opacity > 0:
                shapes.append(
                    dict(
                        type="line",
                        x0=x_val,
                        x1=x_val,
                        y0=y_start,
                        y1=y_end,
                        line=dict(color=f"rgba(128, 128, 128, {opacity})", width=1),
                        layer="below",
                    )
                )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Layer index", font=dict(size=24)),
            tickmode="linear",
            tick0=0,
            dtick=1,
            ticks="",
            showline=False,
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            title="",
            ticks="",
            showline=False,
            showgrid=False,
            range=[0, y_max],
        ),
        shapes=shapes,
        width=800,
        height=450,
        plot_bgcolor=apc.white.hex_code,
        paper_bgcolor=apc.white.hex_code,
        showlegend=False,
        margin=dict(l=80, r=120, t=0, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
        annotations=[
            dict(
                text="Feature importance",
                x=-0.04,
                y=0.36,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntl-Medium", size=24),
                textangle=-90,
            ),
            dict(
                text="MSA depth",
                x=1.01,
                y=0.73,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntl-Medium", size=18),
                textangle=0,
            ),
        ]
        + bin_label_annotations,
    )

    apc.plotly.style_plot(fig, monospaced_axes="x")

    return fig


def stacked_feature_importance_by_r2_plot(
    df: pd.DataFrame,
    r2_col: str = "Adjusted R2",
    feature_cols: list[int] | None = None,
    gap: float = -0.78,
    gradient=None,
):
    if feature_cols is None:
        feature_cols = list(range(22))

    if gradient is None:
        gradient = apc.gradients.sunset.reverse()

    df_copy = df.copy()

    bin_edges = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = [
        "0.3-0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        "0.9-1.0",
    ]

    df_copy["R2 Bin"] = pd.cut(df_copy[r2_col], bins=bin_edges, labels=bin_labels, right=False)
    df_copy = df_copy.dropna(subset=["R2 Bin"])

    actual_n_bins = len(df_copy["R2 Bin"].unique())
    colors = [c.hex_code for c in gradient.resample_as_palette(actual_n_bins)]

    bins = sorted(df_copy["R2 Bin"].unique())
    feature_indices = np.arange(len(feature_cols))

    all_mean_values = []
    for bin_label in bins:
        bin_data = df_copy[df_copy["R2 Bin"] == bin_label]
        mean_values = bin_data[feature_cols].mean().values
        all_mean_values.append(mean_values)

    all_mean_values_array = np.array(all_mean_values)
    global_min = all_mean_values_array.min()
    global_max = all_mean_values_array.max()

    fig = go.Figure()

    for i in reversed(range(len(bins))):
        bin_label = bins[i]
        bin_data = df_copy[df_copy["R2 Bin"] == bin_label]

        mean_values = bin_data[feature_cols].mean().values
        std_values = bin_data[feature_cols].std().values
        n_samples = len(bin_data)
        se_values = std_values / np.sqrt(n_samples)

        percentage_values = (mean_values / mean_values.sum()) * 100
        se_percentage_values = (se_values / mean_values.sum()) * 100

        if global_max > global_min:
            scaled_values = (mean_values - global_min) / (global_max - global_min)
        else:
            scaled_values = np.zeros_like(mean_values)

        offset = i * (1.0 + gap)
        y_values = scaled_values + offset

        hover_data = np.column_stack([percentage_values, se_percentage_values])

        hex_color = colors[i].lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        rgba_solid = f"rgba({r},{g},{b},0.9)"
        rgba_medium = f"rgba({r},{g},{b},0.5)"
        rgba_transparent = f"rgba({r},{g},{b},0.0)"

        line_gray_value = int(102 * (1 - i / (len(bins) - 1)))
        line_color = f"#{line_gray_value:02x}{line_gray_value:02x}{line_gray_value:02x}"

        fig.add_trace(
            go.Scatter(
                x=feature_indices,
                y=np.full_like(feature_indices, offset, dtype=float),
                mode="lines",
                line=dict(color=colors[i], width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=feature_indices,
                y=y_values,
                mode="lines",
                line=dict(color=line_color, width=2.0),
                fill="tonexty",
                fillgradient=dict(
                    type="vertical",
                    colorscale=[
                        (0.0, rgba_transparent),
                        (0.1, rgba_medium),
                        (0.5, rgba_solid),
                        (1.0, rgba_solid),
                    ],
                ),
                showlegend=False,
                hovertemplate=(
                    f"Adjusted R²: {bin_label}<br>Layer: "
                    f"%{{x}}<br>Importance: %{{customdata[0]:.1f}} "
                    f"± %{{customdata[1]:.1f}}%<extra></extra>"
                ),
                customdata=hover_data,
            )
        )

    bin_label_annotations = []
    label_offset = 0.1
    for i in range(len(bins)):
        bin_label = bins[i]
        offset = i * (1.0 + gap) + label_offset
        hex_color = colors[i].lstrip("#")
        label_color = "#888888" if (i == 0 and gradient.name == "verde_r") else f"#{hex_color}"

        bin_label_annotations.append(
            dict(
                text=bin_label,
                x=1.01,
                y=offset,
                xref="paper",
                yref="y",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntlMono", size=20, color=label_color),
            )
        )

    y_max = (len(bins) - 1) * (1.0 + gap) + 1.0
    num_segments = 50

    shapes = []
    for x_val in range(len(feature_cols)):
        y_positions = np.linspace(0, y_max, num_segments + 1)
        for j in range(num_segments):
            y_start = y_positions[j]
            y_end = y_positions[j + 1]
            y_mid = (y_start + y_end) / 2
            y_fraction = y_mid / y_max

            if y_fraction <= 0.6:
                opacity = 0.25
            elif y_fraction <= 0.8:
                opacity = 0.25 * (1 - (y_fraction - 0.6) / 0.2)
            else:
                opacity = 0.0

            if opacity > 0:
                shapes.append(
                    dict(
                        type="line",
                        x0=x_val,
                        x1=x_val,
                        y0=y_start,
                        y1=y_end,
                        line=dict(color=f"rgba(128, 128, 128, {opacity})", width=1),
                        layer="below",
                    )
                )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Layer index", font=dict(size=24)),
            tickmode="linear",
            tick0=0,
            dtick=1,
            ticks="",
            showline=False,
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=False,
            title="",
            ticks="",
            showline=False,
            showgrid=False,
            range=[0, y_max],
        ),
        shapes=shapes,
        width=800,
        height=450,
        plot_bgcolor=apc.white.hex_code,
        paper_bgcolor=apc.white.hex_code,
        showlegend=False,
        margin=dict(l=80, r=120, t=0, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
        annotations=[
            dict(
                text="Feature importance",
                x=-0.04,
                y=0.36,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntl-Medium", size=24),
                textangle=-90,
            ),
            dict(
                text="Adjusted R²",
                x=1.01,
                y=0.70,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntl-Medium", size=18),
                textangle=0,
            ),
        ]
        + bin_label_annotations,
    )

    apc.plotly.style_plot(fig, monospaced_axes="x")

    return fig
