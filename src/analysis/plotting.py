from pathlib import Path

import arcadia_pycolor as apc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from arcadia_pycolor.gradient import Gradient
from ete3 import NodeStyle, TextFace, TreeStyle
from plotly.subplots import make_subplots

# Set Plotly renderer for Quarto compatibility
pio.renderers.default = "plotly_mimetype+notebook_connected"


def tree_style_with_categorical_annotation(
    categories: dict[str, str],
    highlight: list[str] | None = None,
) -> TreeStyle:
    unique_categories = set(categories.values())
    colors = apc.palettes.primary.colors
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

    if highlight is None:
        highlight = []

    def layout(node):
        if node.is_leaf():
            node_style = NodeStyle()

            is_highlighted = highlight and any(key in node.name for key in highlight)

            if is_highlighted:
                node_style["shape"] = "sphere"
                node_style["size"] = 15

                matched_key = next((key for key in highlight if key in node.name), None)
                if matched_key:
                    text_face = TextFace(matched_key, fsize=32, bold=False)
                    node.add_face(text_face, column=0, position="branch-right")
            else:
                node_style["shape"] = "sphere"
                node_style["size"] = 15

            for key, category in categories.items():
                if key in node.name:
                    node_style["fgcolor"] = color_map.get(category, "#CCCCCC")
                    break
            else:
                node_style["fgcolor"] = "#CCCCCC"

            node.set_style(node_style)

    tree_style = TreeStyle()
    tree_style.layout_fn = layout
    tree_style.show_leaf_name = False
    tree_style.show_branch_length = False
    tree_style.scale = 240
    tree_style.rotation = 90

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
        if node.is_leaf():
            node_style = NodeStyle()

            is_highlighted = highlight and any(key in node.name for key in highlight)

            if is_highlighted:
                node_style["shape"] = "sphere"
                node_style["size"] = 15

                matched_key = next((key for key in highlight if key in node.name), None)
                if matched_key:
                    text_face = TextFace(matched_key, fsize=32, bold=False)
                    node.add_face(text_face, column=0, position="branch-right")
            else:
                node_style["shape"] = "sphere"
                node_style["size"] = 15

            # Find matching key and set color based on scalar value
            for key in values.keys():
                if key in node.name:
                    node_style["fgcolor"] = color_mapping.get(key, "#CCCCCC")
                    break
            else:
                node_style["fgcolor"] = "#CCCCCC"

            node.set_style(node_style)

    tree_style = TreeStyle()
    tree_style.layout_fn = layout
    tree_style.show_leaf_name = False
    tree_style.show_branch_length = False
    tree_style.scale = 240
    tree_style.rotation = 90

    return tree_style


def interactive_layer_weight_plot(df: pd.DataFrame, num_layers: int = 22) -> go.Figure:
    """Create interactive plotly figure with dropdown to select layer weights."""

    queries = df["query"].unique()
    subfamilies = df["subfamily"].unique()
    colors = [color.hex_code for color in apc.palettes.primary.colors[: len(subfamilies)]]
    subfamily_colors = {subfamily: colors[i] for i, subfamily in enumerate(subfamilies)}

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{query} (subfamily {subfamily})"
            for query, subfamily in zip(queries, subfamilies, strict=False)
        ],
        horizontal_spacing=0.08,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="Tree distance (patristic)",
        y_title="Sequence weight",
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
    )

    num_traces_per_option = len(queries) * len(subfamilies)

    for option_idx in range(num_layers + 1):
        if option_idx == 0:
            weight_col = "median_weight"
            visible = True
        else:
            weight_col = f"layer_{option_idx - 1}_weight"
            visible = False

        for query_idx, query in enumerate(queries):
            query_data = df[df["query"] == query]

            for subfamily in subfamilies:
                subfamily_data = query_data[query_data["subfamily"] == subfamily]

                if not subfamily_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=subfamily_data["patristic_distance"],
                            y=subfamily_data[weight_col],
                            mode="markers",
                            marker=dict(color=subfamily_colors[subfamily], size=8, opacity=0.6),
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
                            # textfont=dict(
                            #    family="Suisse Int'l",
                            #    size=13,
                            # ),
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
        else:
            label = f"Layer {option_idx - 1}"

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": visibility}],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=-0.15,
                xanchor="left",
                y=1.05,
                yanchor="bottom",
                pad=dict(t=0, b=0, l=2, r=2),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=12),
            )
        ],
        width=800,
        height=400,
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.20, yanchor="top"),
    )

    x_min = df["patristic_distance"].min()
    x_max = df["patristic_distance"].max()
    x_diff = x_max - x_min
    fig.update_xaxes(range=[x_min - 0.02 * x_diff, x_max + 0.02 * x_diff], matches="x")
    fig.update_yaxes(rangemode="tozero")

    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=1)
    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=2)
    apc.plotly.style_plot(fig, monospaced_axes="all", row=1, col=3)
    # apc.plotly.style_legend(fig)

    return fig


if __name__ == "__main__":
    import pandas as pd

    from analysis.tree import read_newick, subset_tree

    tree = read_newick("data/response_regulators/PF00072.final.fasttree.newick")
    tree = subset_tree(tree, 200, force_include=["1NXS", "4CBV", "4E7P"])
    membership_path = "data/response_regulators/membership.txt"
    categories = (
        pd.read_csv(membership_path, sep="\t").set_index("record_id")["subfamily"].to_dict()
    )

    tree_style = tree_style_with_categorical_annotation(
        tree, categories, ["1NXS", "4CBV", "4E7P"], Path("tree_plot.png")
    )
    tree.render("tree_plot.png", tree_style=tree_style)
