import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
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
    n_bins: int = 10,
    gradient=None,
    bw_adjust: float = 0.5,
    aspect: int = 15,
    height: float = 0.5,
    hspace: float = -0.25,
):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "figure.autolayout": False})

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

    bins = sorted(df_copy["Size Bin"].unique(), reverse=True)
    df_copy["Size Bin Label"] = df_copy["Size Bin"]

    df_copy["Size Bin Label"] = pd.Categorical(
        df_copy["Size Bin Label"],
        categories=list(reversed(bin_labels)),
        ordered=True,
    )

    palette = {
        label: colors[bin_labels.index(label)]
        for label in bin_labels
        if label in df_copy["Size Bin Label"].values
    }

    g = sns.FacetGrid(
        df_copy,
        row="Size Bin Label",
        hue="Size Bin Label",
        aspect=aspect,
        height=height,
        palette=palette,
    )

    g.map(
        sns.kdeplot,
        y_col,
        bw_adjust=bw_adjust,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
        clip=(0.3, 1.0),
    )
    g.map(sns.kdeplot, y_col, clip_on=False, color="w", lw=2, bw_adjust=bw_adjust, clip=(0.3, 1.0))
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, y_col)
    g.figure.subplots_adjust(hspace=hspace)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    for ax in g.axes.flat:
        ax.set_xlim(left=0.3, right=1)
        apc.mpl.style_plot(ax, monospaced_axes="x")

    x_label = y_col.replace("R2", "R$^2$")
    g.set_axis_labels(x_var=x_label, fontweight="bold", fontsize=16)
    g.figure.text(
        0.084,
        0.47,
        "MSA depth range",
        va="center",
        rotation=90,
        fontweight="bold",
        fontsize=16,
    )

    g.axes[-1, 0].tick_params(axis="x", bottom=True, length=4, width=1)

    return g


def stacked_feature_importance_plot(
    df: pd.DataFrame,
    size_col: str = "Size",
    feature_cols: list[int] | None = None,
    n_bins: int = 10,
    gap: float = 0.0,
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

    plotly_colorscale = [
        [i / 99, gradient.map_values([i], min_value=0, max_value=99)[0].hex_code]
        for i in range(100)
    ]

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
                hovertemplate=f"MSA depth: {bin_label}<br>Layer: %{{x}}<br>Importance: %{{customdata[0]:.1f}} ± %{{customdata[1]:.1f}}%<extra></extra>",
                customdata=hover_data,
            )
        )

    colorbar_x = 1.02
    colorbar_y_center = 0.37
    colorbar_len = 0.5
    colorbar_y_top = colorbar_y_center + colorbar_len / 2
    colorbar_y_bottom = colorbar_y_center - colorbar_len / 2

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=plotly_colorscale,
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(
                        text="MSA depth",
                        side="right",
                        font=dict(family="SuisseIntl-Medium", size=20),
                    ),
                    thickness=20,
                    len=colorbar_len,
                    x=colorbar_x,
                    xanchor="left",
                    y=colorbar_y_center,
                    yanchor="middle",
                    tickvals=[],
                    ticktext=[],
                    outlinewidth=0,
                ),
            ),
            hoverinfo="none",
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
        margin=dict(l=80, r=120, t=40, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
        annotations=[
            dict(
                text="Feature importance",
                x=-0.04,
                y=0.37,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(family="SuisseIntl-Medium", size=24),
                textangle=-90,
            ),
            dict(
                text="Large",
                x=colorbar_x + 0.01,
                y=colorbar_y_top - 0.03,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=18),
                textangle=0,
            ),
            dict(
                text="Small",
                x=colorbar_x + 0.01,
                y=colorbar_y_bottom + 0.024,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=dict(size=18),
                textangle=0,
            ),
        ],
    )

    apc.plotly.style_plot(fig, monospaced_axes="x")

    return fig


if __name__ == "__main__":
    from pathlib import Path

    apc.plotly.setup()
    apc.mpl.setup()

    df = pd.read_csv(Path("test2.tsv"), sep="\t")

    number_cols = [str(i) for i in range(22)]
    df = df.rename(columns={col: int(col) for col in number_cols if col in df.columns})

    g_ridge = ridgeline_r2_plot(df, gradient=apc.gradients.verde.reverse())
    g_ridge.savefig("seaborn_ridgeline_plot.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Plot saved to seaborn_ridgeline_plot.png")

    fig = stacked_feature_importance_plot(df, gap=-0.78, gradient=apc.gradients.verde.reverse())
    fig.write_html("feature_importance.html")
    print("Plot saved to feature_importance.html")
