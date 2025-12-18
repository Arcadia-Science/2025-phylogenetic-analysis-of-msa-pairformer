import arcadia_pycolor as apc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from arcadia_pycolor.gradient import Gradient
from matplotlib.colors import PowerNorm
from arcadia_pycolor.style_defaults import DEFAULT_FONT, MONOSPACE_FONT
from ete3 import NodeStyle, TextFace, Tree, TreeStyle
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, zscore
from statsmodels.stats.anova import anova_lm

from analysis.regression import regress_and_analyze_features
from analysis.tree import get_patristic_distance, read_newick
from MSA_Pairformer.dataset import MSA

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
    line_width: int = 2,
) -> TreeStyle:
    if highlight is None:
        highlight = []

    def layout(node):
        node_style = NodeStyle()
        node_style["hz_line_width"] = line_width
        node_style["vt_line_width"] = line_width
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
                node_style["fgcolor"] = "#666666"

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
        else:
            node_style["size"] = 0

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
    bin_col: str = "MSA Depth",
    y_col: str = "Adjusted R2",
    gradient=None,
    bw_adjust: float = 0.11,
    gap: float = 0.7,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    n_bins: int = 8,
):
    from scipy.stats import gaussian_kde

    if gradient is None:
        gradient = apc.gradients.sunset.reverse()

    df_copy = df.copy()

    if bin_edges is None:
        if bin_col == "MSA Depth":
            bin_edges = [200, 300, 400, 500, 600, 700, 800, 900, 1025]
        else:
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = df_copy[bin_col].quantile(quantiles).tolist()

    if bin_labels is None:
        if bin_col == "MSA Depth":
            bin_labels = [
                f"{bin_edges[i]:g}-{bin_edges[i + 1] - 1:g}" for i in range(len(bin_edges) - 1)
            ]
        else:
            bin_labels = [
                f"{bin_edges[i]:g}-{bin_edges[i + 1]:g}" for i in range(len(bin_edges) - 1)
            ]

    df_copy["Bin"] = pd.cut(df_copy[bin_col], bins=bin_edges, labels=bin_labels, right=False)
    df_copy = df_copy.dropna(subset=["Bin"])

    actual_n_bins = len(df_copy["Bin"].unique())
    colors = [c.hex_code for c in gradient.resample_as_palette(actual_n_bins)]

    bins = sorted(df_copy["Bin"].unique())

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
        bin_counts.append(len(df_copy[df_copy["Bin"] == bin_label]))
        bin_data = df_copy[df_copy["Bin"] == bin_label]
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
                hoverinfo="text",
                text=f"{bin_col}: {bin_label}<br>Mean ± SD: {mean_val:.2f} ± {std_val:.2f}",
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
        label_color = "black"
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
        text=bin_col,
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
    bin_col: str,
    bin_edges: list[float],
    bin_display_name: str,
    bin_labels: list[str] | None = None,
    feature_cols: list[int] | None = None,
    gap: float = -0.78,
    gradient=None,
    annotation_y_position: float = 0.73,
):
    if feature_cols is None:
        feature_cols = list(range(22))

    if gradient is None:
        gradient = apc.gradients.sunset.reverse()

    df_copy = df.copy()

    if bin_labels is None:
        bin_labels = [f"{bin_edges[i]:g}-{bin_edges[i + 1]:g}" for i in range(len(bin_edges) - 1)]

    bin_column_name = f"{bin_col} Bin"
    df_copy[bin_column_name] = pd.cut(
        df_copy[bin_col], bins=bin_edges, labels=bin_labels, right=False
    )
    df_copy = df_copy.dropna(subset=[bin_column_name])

    actual_n_bins = len(df_copy[bin_column_name].unique())
    colors = [c.hex_code for c in gradient.resample_as_palette(actual_n_bins)]

    bins = sorted(df_copy[bin_column_name].unique())
    feature_indices = np.arange(len(feature_cols))

    all_mean_values = []
    for bin_label in bins:
        bin_data = df_copy[df_copy[bin_column_name] == bin_label]
        mean_values = bin_data[feature_cols].mean().values
        all_mean_values.append(mean_values)

    all_mean_values_array = np.array(all_mean_values)
    global_min = all_mean_values_array.min()
    global_max = all_mean_values_array.max()

    fig = go.Figure()

    for i in reversed(range(len(bins))):
        bin_label = bins[i]
        bin_data = df_copy[df_copy[bin_column_name] == bin_label]

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
                    f"{bin_display_name}: {bin_label}<br>Layer: "
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
                font=dict(family="SuisseIntlMono", size=20, color="black"),
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
                text=bin_display_name,
                x=1.01,
                y=annotation_y_position,
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


def multivariate_regression_figures(df: pd.DataFrame) -> None:
    features = [
        "Phylogenetic diversity",
        "Colless",
        "Cherry count",
        "Ultrametricity CV",
        "Patristic mean",
        "Patristic std",
        "Query centrality",
    ]

    df_original = df.copy()
    df_zscore = df.copy()
    for feature in features:
        df_zscore[feature] = zscore(df_zscore[feature])

    formula = 'Q("Adjusted R2") ~ ' + " + ".join(f'Q("{f}")' for f in features)
    lm = smf.ols(formula, data=df_zscore).fit()

    anova_results = anova_lm(lm, typ=3)
    anova_feature_names = [f'Q("{f}")' for f in features]
    feature_ss = anova_results.loc[anova_feature_names, "sum_sq"]
    total_feature_ss = feature_ss.sum()
    feature_proportions = feature_ss / total_feature_ss
    feature_importance_pct = feature_proportions * lm.rsquared_adj * 100

    importance_df = pd.DataFrame(
        {"Feature": features, "Importance (%)": feature_importance_pct.values}
    )

    feature_colors = {
        "Phylogenetic diversity": apc.lapis.hex_code,
        "Patristic std": apc.fern.hex_code,
    }
    bar_colors = [feature_colors.get(f, "#E2E2E2") for f in features]

    fig1, ax1 = plt.subplots()
    ax1.bar(
        importance_df["Feature"],
        importance_df["Importance (%)"],
        color=bar_colors,
        edgecolor="black",
        linewidth=2,
        alpha=0.85,
    )
    for i, (_, row) in enumerate(importance_df.iterrows()):
        is_highlighted = row["Feature"] in feature_colors
        ax1.text(
            i,
            row["Importance (%)"] * 1.15,
            f"{row['Importance (%)']:.1f}%",
            ha="center",
            va="bottom",
            fontfamily=MONOSPACE_FONT,
            fontweight="bold" if is_highlighted else "normal",
        )
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(importance_df)))
    ax1.set_xticklabels(importance_df["Feature"], rotation=45, ha="right")
    apc.mpl.style_plot(ax1, monospaced_axes="y")
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.hexbin(
        df_original["Phylogenetic diversity"],
        df_original["Adjusted R2"],
        gridsize=25,
        cmap=apc.gradients.blues.reverse().to_mpl_cmap(),
        mincnt=1,
        norm=PowerNorm(gamma=0.85),
    )
    ax2.set_xlabel("Phylogenetic diversity")
    ax2.set_ylabel("Adjusted R²")
    apc.mpl.style_plot(ax2, monospaced_axes="both")
    fig2.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.hexbin(
        df_original["Patristic std"],
        df_original["Adjusted R2"],
        gridsize=25,
        cmap=apc.gradients.greens.reverse().to_mpl_cmap(),
        mincnt=1,
        norm=PowerNorm(gamma=0.85),
        xscale="log",
    )
    ax3.set_xlabel("Patristic std")
    ax3.set_ylabel("Adjusted R²")
    apc.mpl.style_plot(ax3, monospaced_axes="both")
    fig3.tight_layout()
    plt.show()


def visualize_expected_vs_actual(
    tree: Tree,
    msa: MSA,
    weights: torch.Tensor,
    query: str,
    color_map: dict[str, str] | None = None,
):
    tree = tree.copy()
    dist_to_query = get_patristic_distance(tree, query)[msa.ids_l].values
    dist_to_query = dist_to_query[dist_to_query > 0]
    weights = weights.clone()[1:, :]
    model, _ = regress_and_analyze_features(weights, dist_to_query)

    y_pred = model.fittedvalues
    y_actual = model.model.endog
    rho, _ = spearmanr(y_pred, y_actual)

    fig, ax = plt.subplots()

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k-", lw=1)

    if color_map is not None:
        ids = [id for id in msa.ids_l if id in color_map]
        colors = [color_map[id] for id in ids]
        ax.scatter(y_actual, y_pred, c=colors, alpha=0.7, s=50)
    else:
        ax.scatter(y_actual, y_pred, alpha=0.7, s=50)

    ax.set_xlabel("Actual (standard normalized)")
    ax.set_ylabel("Predicted (standard normalized)")
    ax.set_aspect("equal", adjustable="box")

    apc.mpl.style_plot(ax, monospaced_axes="both")

    annotation_text = (
        f"R²: {model.rsquared:.3f}\nR² Adjusted: {model.rsquared_adj:.3f}\nSpearman: {rho:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily=MONOSPACE_FONT,
        verticalalignment="top",
        bbox=dict(
            facecolor="#FFFFFF",
            edgecolor="#444444",
            linewidth=1.0,
            alpha=0.8,
        ),
    )

    return fig
