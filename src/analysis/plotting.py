from pathlib import Path

import arcadia_pycolor as apc
import ipywidgets as widgets
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
    """Create interactive plotly figure with dropdown to select layer weights"""

    categories = df["query"].unique()
    subfamilies = df["subfamily"].unique()
    colors = [color.hex_code for color in apc.palettes.primary.colors[: len(subfamilies)]]
    subfamily_colors = {subfamily: colors[i] for i, subfamily in enumerate(subfamilies)}

    # Create subplot with 3 columns for the 3 queries
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"Query: {cat}" for cat in categories],
        horizontal_spacing=0.08,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="Patristic Distance",
        y_title="Weight",
    )

    # Create traces for each layer
    traces_by_layer = {}

    for layer_idx in range(num_layers):
        weight_col = f"layer_{layer_idx}_weight"
        layer_traces = []

        for i, category in enumerate(categories):
            category_data = df[df["query"] == category]

            for subfamily in subfamilies:
                subfamily_data = category_data[category_data["subfamily"] == subfamily]
                if not subfamily_data.empty:
                    trace = go.Scatter(
                        x=subfamily_data["patristic_distance"],
                        y=subfamily_data[weight_col],
                        mode="markers",
                        marker=dict(color=subfamily_colors[subfamily], size=8, opacity=0.6),
                        name=f"{subfamily}",
                        legendgroup=subfamily,
                        showlegend=(i == 0),  # Only show legend for first subplot
                        visible=False,  # Initially hide all layer traces
                    )
                    layer_traces.append((trace, i + 1))  # Store trace and column index

        traces_by_layer[layer_idx] = layer_traces

    # Add all traces to the figure
    for layer_traces in traces_by_layer.values():
        for trace, col_idx in layer_traces:
            fig.add_trace(trace, row=1, col=col_idx)

    # Create dropdown menu
    dropdown_buttons = []
    for layer_idx in range(num_layers):
        # Create visibility array - True for traces of this layer, False for others
        visibility = []
        showlegend = []
        for layer_idx in range(num_layers):
            if layer_idx == layer_idx:
                visibility.extend([True] * len(traces_by_layer[layer_idx]))
                # Show legend only for first subplot traces of the selected layer
                for _, (_, col_idx) in enumerate(traces_by_layer[layer_idx]):
                    showlegend.append(col_idx == 1)  # Only first column shows legend
            else:
                visibility.extend([False] * len(traces_by_layer[layer_idx]))
                showlegend.extend([False] * len(traces_by_layer[layer_idx]))

        dropdown_buttons.append(
            dict(
                label=f"Layer {layer_idx}",
                method="update",
                args=[{"visible": visibility, "showlegend": showlegend}],
            )
        )

    # Add median weight option
    median_traces = []
    for i, category in enumerate(categories):
        category_data = df[df["query"] == category]

        for subfamily in subfamilies:
            subfamily_data = category_data[category_data["subfamily"] == subfamily]
            if not subfamily_data.empty:
                trace = go.Scatter(
                    x=subfamily_data["patristic_distance"],
                    y=subfamily_data["median_weight"],
                    mode="markers",
                    marker=dict(color=subfamily_colors[subfamily], size=8, opacity=0.6),
                    name=f"{subfamily}",
                    legendgroup=subfamily,
                    showlegend=(i == 0),
                    visible=True,  # Show median traces by default
                )
                median_traces.append((trace, i + 1))

    # Add median traces
    for trace, col_idx in median_traces:
        fig.add_trace(trace, row=1, col=col_idx)

    # Add median option to dropdown (put it first since it's default)
    median_visibility = [False] * len(
        [t for traces in traces_by_layer.values() for t in traces]
    ) + [True] * len(median_traces)
    median_showlegend = [False] * len(
        [t for traces in traces_by_layer.values() for t in traces]
    ) + [trace[1] == 1 for trace in median_traces]  # Only first column shows legend
    dropdown_buttons.insert(
        0,
        dict(
            label="Median Weight",
            method="update",
            args=[{"visible": median_visibility, "showlegend": median_showlegend}],
        ),
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Patristic Distance vs Layer Weights (Interactive)", x=0.5, xanchor="center"
        ),
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=-0.1,
                xanchor="left",
                y=1.05,
                yanchor="bottom",
                pad=dict(t=0, b=0, l=2, r=2),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=10),
            )
        ],
        width=800,
        height=450,
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"),
    )

    # Remove individual axis labels since we have shared titles
    # The shared axis titles are set in make_subplots

    return fig


def interactive_layer_weight_plot_widget(df: pd.DataFrame, num_layers: int = 22):
    """Create interactive plotly figure with ipywidgets dropdown for layer selection

    Returns a VBox widget that contains both the dropdown and plot, suitable for Quarto rendering.
    """

    # Create dropdown widget
    layer_options = ["Median Weight"] + [f"Layer {i}" for i in range(num_layers)]
    layer_dropdown = widgets.Dropdown(
        options=layer_options,
        value="Median Weight",
        description="Select Layer:",
        style={"description_width": "100px"},
        layout={"width": "200px"},
    )

    categories = df["query"].unique()
    subfamilies = df["subfamily"].unique()
    colors = [color.hex_code for color in apc.palettes.primary.colors[: len(subfamilies)]]
    subfamily_colors = {subfamily: colors[i] for i, subfamily in enumerate(subfamilies)}

    # Create output widget to hold the plot
    output = widgets.Output()

    def create_and_show_plot(selected_layer):
        """Create and display plot for the selected layer"""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[f"Query: {cat}" for cat in categories],
            horizontal_spacing=0.08,
            shared_xaxes=True,
            shared_yaxes=True,
            x_title="Patristic Distance",
            y_title="Weight",
        )

        # Determine which column to use
        if selected_layer == "Median Weight":
            weight_col = "median_weight"
        else:
            layer_num = int(selected_layer.split()[-1])
            weight_col = f"layer_{layer_num}_weight"

        # Add traces for each query and subfamily
        for i, category in enumerate(categories):
            category_data = df[df["query"] == category]

            for subfamily in subfamilies:
                subfamily_data = category_data[category_data["subfamily"] == subfamily]
                if not subfamily_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=subfamily_data["patristic_distance"],
                            y=subfamily_data[weight_col],
                            mode="markers",
                            marker=dict(color=subfamily_colors[subfamily], size=8, opacity=0.6),
                            name=f"{subfamily}",
                            legendgroup=subfamily,
                            showlegend=(i == 0),  # Only show legend for first subplot
                        ),
                        row=1,
                        col=i + 1,
                    )

        fig.update_layout(
            title=dict(text=f"Patristic Distance vs {selected_layer}", x=0.5, xanchor="center"),
            width=800,
            height=450,
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"),
        )

        # Clear output and show new plot
        with output:
            output.clear_output(wait=True)
            fig.show()

    def update_plot(change):
        """Update plot when dropdown selection changes"""
        selected = change["new"]
        create_and_show_plot(selected)

    layer_dropdown.observe(update_plot, names="value")

    # Show initial plot
    create_and_show_plot("Median Weight")

    # Return a VBox containing both widgets for Quarto compatibility
    return widgets.VBox([layer_dropdown, output])


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
