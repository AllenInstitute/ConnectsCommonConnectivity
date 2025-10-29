from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_array
from scipy.stats import rankdata


def clear_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])


def get_relative_measurement(ax, main_ax, measurement="height"):
    """
    Get the height of an axis's bounding box, as a fraction of main_ax's height.
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure renderer is up-to-date
    renderer = fig.canvas.get_renderer()

    # Get height in inches for the label axis and the main axis
    label_bbox = ax.get_tightbbox(renderer=renderer)
    main_bbox = main_ax.get_tightbbox(renderer=renderer)
    if measurement == "height":
        label_height_inches = label_bbox.height
        main_height_inches = main_bbox.height
    elif measurement == "width":
        label_height_inches = label_bbox.width
        main_height_inches = main_bbox.width

    return label_height_inches / main_height_inches


def draw_label_arc(ax, row_label, col_label):
    # Parameters
    xshift = -0.04
    yshift = 0.02
    center = (0 + xshift, 1 + yshift)
    radius = 0.05
    theta1 = 110  # Start angle in degrees (positive y-axis)
    theta2 = 160  # End angle in degrees (negative x-axis)

    # Draw the arc (no arrowhead)
    arc = Arc(
        center,
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        linewidth=2,
        color="black",
        clip_on=False,
        transform=ax.transAxes,
    )
    ax.add_patch(arc)

    # Add an arrowhead at the end of the arc
    # Compute end point of the arc
    start_angle_rad = np.radians(theta2)
    start_point = (
        center[0] + radius * np.cos(start_angle_rad),
        center[1] + radius * np.sin(start_angle_rad),
    )

    end_angle_rad = np.radians(theta1)
    end_point = (
        center[0] + radius * np.cos(end_angle_rad),
        center[1] + radius * np.sin(end_angle_rad),
    )
    dx = -np.sin(end_angle_rad)
    dy = np.cos(end_angle_rad)

    arrow_tip = end_point
    tip_pad = 0.02
    arrow = FancyArrowPatch(
        posA=arrow_tip,
        posB=(arrow_tip[0] - tip_pad * dx, arrow_tip[1] - tip_pad * dy),
        arrowstyle="->",
        mutation_scale=15,
        color="black",
        linewidth=2,
        clip_on=False,
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)

    ax.text(
        start_point[0],
        start_point[1],
        row_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize="medium",
    )
    ax.text(
        end_point[0] + tip_pad,
        end_point[1],
        col_label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize="medium",
    )


class AxisGrid:
    def __init__(
        self,
        ax,
        spines=True,
    ):
        fig = ax.figure
        divider = make_axes_locatable(ax)
        self.spines = spines
        self.fig = fig
        self.ax = ax
        self.divider = divider
        self.top_axs = []
        self.left_axs = []
        self.bottom_axs = []
        self.right_axs = []
        self.side_axs = {
            "top": self.top_axs,
            "bottom": self.bottom_axs,
            "left": self.left_axs,
            "right": self.right_axs,
        }

    @property
    def all_top_axs(self):
        return [self.ax] + self.top_axs

    @property
    def all_bottom_axs(self):
        return [self.ax] + self.bottom_axs

    @property
    def all_left_axs(self):
        return [self.ax] + self.left_axs

    @property
    def all_right_axs(self):
        return [self.ax] + self.right_axs

    def append_axes(self, side, size="10%", pad="auto", **kwargs) -> plt.Axes:
        # NOTE: old way was using shared axes, but labels kept getting annoying
        # kws = {}
        # if side in ["top", "bottom"]:
        #     kws["sharex"] = self.ax
        # elif side in ["left", "right"]:
        #     kws["sharey"] = self.ax

        if pad == "auto":
            if len(self.side_axs[side]) > 0:
                last_ax = self.side_axs[side][-1]
                measurement = "height" if side in ["top", "bottom"] else "width"
                pad = get_relative_measurement(last_ax, self.ax, measurement)
            else:
                pad = 0.0

        # NOTE: this was VERY fragile, could not figure out how to do it in right in
        # float or in manual axes_size like pad = axes_size.from_any(
        #   pad, fraction_ref=axes_size.AxesX(self.ax)
        # )
        pad = f"{pad * 110}%"
        ax = self.divider.append_axes(side, size=size, pad=pad, **kwargs)

        clear_axis(ax)
        ax.tick_params(
            which="both",
            length=0,
        )

        if side in ["top", "bottom"]:
            ax.set_xlim(self.ax.get_xlim())
        elif side in ["left", "right"]:
            ax.set_ylim(self.ax.get_ylim())

        self.side_axs[side].append(ax)
        return ax

    def set_title(self, title, **kwargs):
        for ax in self.all_top_axs:
            ax.set_title("", **kwargs)
        text = self.all_top_axs[-1].set_title(title, **kwargs)
        return text

    def set_xlabel(self, xlabel, **kwargs):
        for ax in self.all_bottom_axs:
            ax.set_xlabel("", **kwargs)
        # NOTE a bit of an abuse of notation here but putting xlabel on the top
        text = self.all_top_axs[-1].set_title(xlabel, **kwargs)
        return text

    def set_ylabel(self, ylabel, **kwargs):
        for ax in self.all_left_axs:
            ax.set_ylabel("", **kwargs)
        text = self.all_left_axs[-1].set_ylabel(ylabel, **kwargs)
        return text

    def set_corner_title(self, title, **kwargs):
        """
        Set a title in the top left corner of the grid.
        """
        # ax = self.all_top_axs[-1]

        text = self.ax.text(0, 1, title, ha="right", rotation=0, **kwargs)
        return text


def draw_bracket(ax, start, end, axis="x", color="black"):
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))
    x = np.linspace(start, end, 1000)
    if axis == "x":
        ax.plot(x, -curve, color=color)
    elif axis == "y":
        ax.plot(curve, x, color=color)


def draw_box(ax, start, end, axis="x", color="black", alpha=0.5, lw=0.5):
    if axis == "x":
        rect = plt.Rectangle(
            (start, 0),
            end - start + 1,
            1,
            color=color,
        )
        ax.axvline(start - 0.5, lw=0.5, alpha=1, color="black", zorder=2)
    elif axis == "y":
        rect = plt.Rectangle(
            (0, start),
            1,
            end - start + 1,
            color=color,
        )
        ax.axhline(start - 0.5, lw=0.5, alpha=1, color="black", zorder=2)
    ax.add_patch(rect)


def add_position_column(nodes, pos_key="position"):
    if pos_key in nodes.columns:
        pos_key = "_" + pos_key
        pos_key = add_position_column(nodes, pos_key)
    else:
        nodes[pos_key] = np.arange(len(nodes))
    return pos_key


def adjacencyplot(
    adjacency: Union[np.ndarray, csr_array, pd.DataFrame],
    nodes: pd.DataFrame = None,
    plot_type: Literal["heatmap", "scattermap"] = "heatmap",
    groupby: Optional[list[str]] = None,
    sortby: Optional[list[str]] = None,
    group_element: Literal["box", "bracket"] = "box",
    group_axis_size: str = "1%",
    node_palette: Optional[dict] = None,
    edge_palette: Optional[Union[str, dict, Callable]] = "Greys",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 8),
    edge_size: bool = True,
    edge_hue: bool = True,
    hue_norm: Optional[tuple] = None,
    sizes: tuple = (1, 10),
    size_norm: Optional[tuple] = None,
    edge_linewidth: float = 0.05,
    label_fontsize: Union[float, int, str] = "medium",
    title_fontsize: Union[float, int, str] = "large",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    arc_labels: Optional[tuple] = ("Pre", "Post"),
    # NEW PARAMETERS FOR LONG-FORM / RECTANGULAR SUPPORT
    long_form: bool = False,
    pre_id_col: str = "pre_pt_root_id",
    post_id_col: str = "post_pt_root_id",
    weight_col: str = "sum_size",
    node_id_col: str = "pt_root_id",
    row_groupby: Optional[Union[str, list[str]]] = None,
    col_groupby: Optional[Union[str, list[str]]] = None,
    row_sortby: Optional[Union[str, list[str]]] = None,
    col_sortby: Optional[Union[str, list[str]]] = None,
    # NEW VALUE BAR (SIDE HEATMAP) OPTIONS
    row_value_bars: Optional[list[str]] = None,
    col_value_bars: Optional[list[str]] = None,
    value_bar_size: str = "2%",
    value_bar_cmaps: Optional[dict] = None,
    value_bar_default_cmap: str = "viridis",
    **kwargs,
):
    """Plot connectivity / adjacency data with rich grouping, ordering and two input modes.

    Two Input Modes
    ---------------
    1. Matrix mode (legacy, usually square): ``adjacency`` is a dense/sparse 2D structure
       (``np.ndarray``, ``csr_array`` or ``pd.DataFrame``). Optional ``nodes`` metadata
       refers to the same set of nodes for both rows and columns.
    2. Long-form edge list mode (rectangular allowed): ``adjacency`` is a DataFrame of
       non‑zero edges with columns identifying a *pre* (row) id, a *post* (column) id,
       and a weight/metric. Separate row and column orderings & groupings are supported.
       This mode is activated by either setting ``long_form=True`` or by passing a
       DataFrame that contains all of ``pre_id_col``, ``post_id_col`` and ``weight_col``.

    Parameters (Common)
    -------------------
    adjacency : array-like | DataFrame
        Matrix or long-form edge table (see modes above).
    nodes : DataFrame, optional
        Node metadata. Required in long-form mode. In matrix mode may be omitted.
    plot_type : {"heatmap", "scattermap"}
        Visualization style: dense heatmap (aggregated) or scatter of non-zero edges.
    groupby : list[str] | str, optional
        (Matrix mode) Columns in ``nodes`` used to group AND sort simultaneously.
        Acts as a fallback for row/col specific grouping in long-form mode.
    sortby : list[str] | str, optional
        (Matrix mode) Additional sort keys (after groups). Fallback for row/col
        specific sorts in long-form mode.
    group_element : {"box", "bracket"}
        Visual indicator style for groups along margins.
    group_axis_size : str
        Size of each appended group axis (e.g. "1%").
    node_palette : dict, optional
        Mapping from group/category labels to colors (used for group indicators & labels).
    edge_palette : str | dict | callable
        Palette for edge color mapping when ``edge_hue`` is True or for heatmap.
    ax : matplotlib.Axes, optional
        Existing axes to draw on. Created if None.
    figsize : tuple
        Figure size if a new figure is created.
    edge_size : bool
        If True, scale scatter marker size by edge weight / data value.
    edge_hue : bool
        If True, color edges by weight ranking/value. If False, single color.
    hue_norm : (min, max), optional
        Explicit normalization range for hue.
    sizes : (min, max)
        Min/max scatter marker size when ``edge_size`` is True.
    edge_linewidth : float
        Outline width for scatter markers.
    label_fontsize : number | str
        Font size for axis / group labels.
    title_fontsize : number | str
        Font size for plot title.
    title, xlabel, ylabel : str, optional
        Text labels; applied only if provided.
    arc_labels : tuple(str, str) | None
        If tuple, draw the decorative arc with these labels (row_label, col_label).

    Long-form Specific Parameters
    -----------------------------
    long_form : bool
        Force long-form interpretation (auto-detected otherwise).
    pre_id_col / post_id_col : str
        Column names in the edge DataFrame identifying source (row) and target (col) ids.
    weight_col : str
        Edge weight / metric column to visualize.
    node_id_col : str
        Column in ``nodes`` used to match both pre and post ids.
    row_groupby / col_groupby : list[str] | str, optional
        Independent grouping columns for rows and columns. If omitted, falls back to
        ``groupby``.
    row_sortby / col_sortby : list[str] | str, optional
        Independent within-group additional ordering. If omitted, falls back to
        ``sortby``.

    Value Bar (Annotation Strip) Parameters
    --------------------------------------
    row_value_bars / col_value_bars : list[str], optional
        One or more column names in ``nodes`` to visualize as thin color-mapped
        bars aligned to the ordered rows or columns respectively. Values should
        be numeric; non-numeric columns are converted to categorical codes.
    value_bar_size : str
        Size (thickness) of each value bar axis (e.g. "2%"). Applied per bar.
    value_bar_cmaps : dict, optional
        Mapping from column name -> matplotlib colormap name. Falls back to
        ``value_bar_default_cmap`` when a column not specified.
    value_bar_default_cmap : str
        Default colormap for value bars when not specified in ``value_bar_cmaps``.

    **kwargs :
        Forwarded to seaborn (``heatmap`` or ``scatterplot``) depending on ``plot_type``.

    Returns
    -------
    ax : matplotlib.Axes
        The main plotting axes.
    grid : AxisGrid
        Container managing appended group axes for further customization.

    Raises
    ------
    ValueError
        If required columns are missing, nodes are absent in long-form mode, or invalid
        options are supplied.

    Examples
    --------
    Matrix (legacy):
    >>> ax, grid = adjacencyplot(adj_matrix, nodes=meta, groupby=['class'])

    Long-form (rectangular):
    >>> ax, grid = adjacencyplot(
    ...     edges_df, nodes=node_df, long_form=True,
    ...     pre_id_col='pre_pt_root_id', post_id_col='post_pt_root_id',
    ...     weight_col='sum_size', row_groupby='pre_type', col_groupby='post_type',
    ...     plot_type='scattermap')
    """
    # ---------------------- Detect long-form invocation ----------------------
    if not long_form and isinstance(adjacency, pd.DataFrame):
        required = {pre_id_col, post_id_col, weight_col}
        if required.issubset(adjacency.columns):
            long_form = True  # implicit enable

    if long_form:
        # Validate edges DataFrame
        if not isinstance(adjacency, pd.DataFrame):  # pragma: no cover (defensive)
            raise ValueError("In long_form mode, 'adjacency' must be a pandas DataFrame of edges.")
        edges = adjacency
        missing = [c for c in [pre_id_col, post_id_col, weight_col] if c not in edges.columns]
        if missing:
            raise ValueError(f"Edge DataFrame missing required columns: {missing}")
        if nodes is None:
            raise ValueError("'nodes' DataFrame is required in long_form mode.")
        if node_id_col not in nodes.columns:
            raise ValueError(
                f"nodes DataFrame missing node_id_col '{node_id_col}'. Available: {list(nodes.columns)}"
            )

        # Determine distinct row/column id sets
        row_ids = edges[pre_id_col].unique()
        col_ids = edges[post_id_col].unique()

        # Subset nodes for rows / columns (retain original metadata)
        row_nodes = nodes[nodes[node_id_col].isin(row_ids)].copy()
        col_nodes = nodes[nodes[node_id_col].isin(col_ids)].copy()
        if row_nodes.empty:
            raise ValueError("No matching row nodes found in 'nodes' for provided pre ids.")
        if col_nodes.empty:
            raise ValueError("No matching column nodes found in 'nodes' for provided post ids.")

        # Resolve grouping/sorting specs
        def _ensure_list(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            return list(x)

        # Fallback to legacy groupby/sortby if row/col specific not provided
        row_groupby_list = _ensure_list(row_groupby if row_groupby is not None else groupby)
        col_groupby_list = _ensure_list(col_groupby if col_groupby is not None else groupby)
        row_sortby_list = _ensure_list(row_sortby if row_sortby is not None else sortby)
        col_sortby_list = _ensure_list(col_sortby if col_sortby is not None else sortby)

        # Compose ordering keys
        row_order_keys = row_groupby_list + row_sortby_list
        col_order_keys = col_groupby_list + col_sortby_list

        # Validate grouping / sorting columns exist
        for cols, df_name, df in [
            (row_order_keys, "row_nodes", row_nodes),
            (col_order_keys, "col_nodes", col_nodes),
        ]:
            missing_cols = [c for c in cols if c and c not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in {df_name} DataFrame.")

        if row_order_keys:
            row_nodes = row_nodes.sort_values(row_order_keys)
        if col_order_keys:
            col_nodes = col_nodes.sort_values(col_order_keys)

        # Assign positional indices
        row_pos_key = add_position_column(row_nodes, pos_key="row_position")
        col_pos_key = add_position_column(col_nodes, pos_key="col_position")

        # Build mapping from id -> position
        row_pos_map = dict(zip(row_nodes[node_id_col], row_nodes[row_pos_key]))
        col_pos_map = dict(zip(col_nodes[node_id_col], col_nodes[col_pos_key]))

        # Map edge endpoints; drop edges with unmapped ids (shouldn't happen if validated)
        mapped = edges[[pre_id_col, post_id_col, weight_col]].copy()
        mapped["_row_pos"] = mapped[pre_id_col].map(row_pos_map)
        mapped["_col_pos"] = mapped[post_id_col].map(col_pos_map)
        mapped = mapped.dropna(subset=["_row_pos", "_col_pos"])  # safety

        sources = mapped["_row_pos"].astype(int).to_numpy()
        targets = mapped["_col_pos"].astype(int).to_numpy()
        data_vals = mapped[weight_col].to_numpy()

        if edge_size:
            size = data_vals
        else:
            size = None
        if edge_hue:
            hue = data_vals
        else:
            hue = None
            edge_palette = None

        # Create axis if needed
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if plot_type == "heatmap":
            # Construct dense matrix (rows x cols)
            n_rows = len(row_nodes)
            n_cols = len(col_nodes)
            dense = np.zeros((n_rows, n_cols), dtype=float)
            dense[sources, targets] = data_vals
            sns.heatmap(
                dense,
                xticklabels=False,
                yticklabels=False,
                vmin=hue_norm[0] if hue_norm else None,
                vmax=hue_norm[1] if hue_norm else None,
                ax=ax,
                square=False,
                cbar_kws={"label": weight_col, "shrink": 0.5},
                cmap=edge_palette,
                **kwargs,
            )
            line_zorder = 2
        elif plot_type == "scattermap":
            sns.scatterplot(
                y=sources,
                x=targets,
                size=size,
                hue=hue,
                hue_norm=hue_norm,
                ax=ax,
                sizes=sizes,
                size_norm=size_norm,
                palette=edge_palette,
                linewidth=edge_linewidth,
                color="black",
                **kwargs,
            )
            line_zorder = -1
            if hue is not None or size is not None:
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(1, 1),
                    title=weight_col,
                    fontsize=label_fontsize,
                )
        else:
            raise ValueError("plot_type must be 'heatmap' or 'scattermap'")

        # Axis limits / orientation
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, len(col_nodes) - 0.5)
        ax.set_ylim(len(row_nodes) - 0.5, -0.5)  # invert to match original style

        grid = AxisGrid(ax)

        # Helper to draw grouping bars for one axis
        def _draw_groups(axis_side: str, group_cols: list[str], df: pd.DataFrame, pos_key: str):
            if not group_cols:
                return
            # Iterate levels from innermost -> outermost (reverse) for nesting
            for i, level in enumerate(group_cols[::-1]):
                cax = grid.append_axes(
                    "left" if axis_side == "rows" else "top",
                    size=group_axis_size,
                    pad="auto",
                    zorder=len(group_cols) - i,
                )
                if group_element == "bracket":
                    cax.spines[["top", "bottom", "left", "right"]].set_visible(False)

                grouped = df.groupby(group_cols[: len(group_cols) - i])[pos_key]
                # Compute segment boundaries & centers. We only drop upper levels
                # when there is at least one to drop. Previous version passed None
                # to droplevel() when no levels remained, triggering KeyError.
                starts = grouped.min().rename("start")
                ends = grouped.max().rename("end")
                means = grouped.mean().rename("mean")
                drop_levels = group_cols[: len(group_cols) - i - 1]
                if drop_levels:  # only drop if there are levels to remove
                    starts = starts.droplevel(drop_levels)
                    ends = ends.droplevel(drop_levels)
                    means = means.droplevel(drop_levels)
                info = pd.concat([starts, ends], axis=1)

                for group_name, (start, end) in info.iterrows():
                    color = node_palette[group_name] if node_palette and group_name in node_palette else "black"
                    if group_element == "box":
                        if axis_side == "rows":
                            draw_box(cax, start + 0.5, end, axis="y", color=color)
                        else:
                            draw_box(cax, start + 0.5, end, axis="x", color=color)
                    else:
                        if axis_side == "rows":
                            draw_bracket(cax, start, end, axis="y", color=color)
                        else:
                            draw_bracket(cax, start, end, axis="x", color=color)

                if axis_side == "rows":
                    cax.set_yticks(means.values)
                    ticklabels = cax.set_yticklabels(means.index, rotation=0, fontsize=8)
                else:
                    cax.set_xticks(means.values)
                    cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                    ticklabels = cax.set_xticklabels(
                        means.index, rotation=45, fontsize=label_fontsize, ha="left"
                    )
                if node_palette:
                    for label, name in zip(ticklabels, means.index):
                        if name in node_palette:
                            label.set_color(node_palette[name])

        _draw_groups("rows", row_groupby_list, row_nodes, row_pos_key)
        _draw_groups("cols", col_groupby_list, col_nodes, col_pos_key)

        # -------------- Value bar helper (after groups so bars are outermost) --------------
        def _ensure_numeric(arr: pd.Series) -> pd.Series:
            if np.issubdtype(arr.dtype, np.number):
                return arr
            # convert categorical/string to codes so they can be colormapped
            return pd.Series(pd.Categorical(arr).codes, index=arr.index, name=arr.name)

        def _add_value_bars(axis_side: str, value_cols: list[str], df: pd.DataFrame):
            if not value_cols:
                return
            for col in value_cols:
                if col not in df.columns:
                    raise ValueError(f"Value bar column '{col}' not found in nodes DataFrame.")
                vals = _ensure_numeric(df[col])
                cmap = (
                    value_bar_cmaps[col]
                    if (value_bar_cmaps and col in value_bar_cmaps)
                    else value_bar_default_cmap
                )
                cax = grid.append_axes(
                    "left" if axis_side == "rows" else "top",
                    size=value_bar_size,
                    pad="auto",
                    zorder=0,
                )
                clear_axis(cax)
                if axis_side == "rows":
                    # vertical strip (rows x 1)
                    arr = vals.to_numpy().reshape(-1, 1)
                    cax.imshow(
                        arr,
                        aspect="auto",
                        origin="upper",
                        cmap=cmap,
                        interpolation="nearest",
                    )
                    cax.set_title(col, fontsize=label_fontsize, rotation=0)
                else:
                    # horizontal strip (1 x cols)
                    arr = vals.to_numpy().reshape(1, -1)
                    cax.imshow(
                        arr,
                        aspect="auto",
                        origin="lower",
                        cmap=cmap,
                        interpolation="nearest",
                    )
                    cax.set_ylabel(col, fontsize=label_fontsize, rotation=0)

        _add_value_bars("rows", row_value_bars or [], row_nodes)
        _add_value_bars("cols", col_value_bars or [], col_nodes)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)

        if isinstance(arc_labels, tuple):
            draw_label_arc(ax, *arc_labels)

        return ax, grid

    # ---------------------- Legacy square matrix path (unchanged logic) ----------------------
    if nodes is None:
        nodes = pd.DataFrame(index=np.arange(adjacency.shape[0]))
    nodes = nodes.reset_index().copy()
    if sortby is not None:
        if isinstance(sortby, str):
            sortby = [sortby]
    else:
        sortby = []
    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
    else:
        groupby = []
    sort_by = groupby + sortby

    nodes = nodes.sort_values(sort_by)

    if isinstance(adjacency, pd.DataFrame):
        adjacency = adjacency.values

    sources, targets = np.nonzero(adjacency)
    data = adjacency[sources, targets]

    pos_key = add_position_column(nodes)
    sources = nodes.loc[sources][pos_key].values
    targets = nodes.loc[targets][pos_key].values

    ranked_data = rankdata(data, method="average") / len(data)

    if edge_size:
        size = data
    else:
        size = None
    if edge_hue:
        hue = data
    else:
        hue = None
        edge_palette = None

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if plot_type == "heatmap":
        if isinstance(adjacency, csr_array):
            adjacency = adjacency.todense()
        plot_adjacency = np.zeros_like(adjacency)
        plot_adjacency[sources, targets] = data
        sns.heatmap(
            plot_adjacency,
            xticklabels=False,
            yticklabels=False,
            vmin=hue_norm[0] if hue_norm else None,
            vmax=hue_norm[1] if hue_norm else None,
            ax=ax,
            square=True,
            cbar_kws={"label": "Synapse count", "shrink": 0.5},
            cmap=edge_palette,
            **kwargs,
        )
        line_zorder = 2
    elif plot_type == "scattermap":
        sns.scatterplot(
            y=sources,
            x=targets,
            size=size,
            hue=hue,
            hue_norm=hue_norm,
            ax=ax,
            sizes=sizes,
            palette=edge_palette,
            linewidth=edge_linewidth,
            color="black",
            **kwargs,
        )
        line_zorder = -1
        if hue is not None or size is not None:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(1, 1),
                title="Edge weight",
                fontsize=label_fontsize,
            )

    if adjacency.shape[0] == adjacency.shape[1]:
        ax.axis("square")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, adjacency.shape[0] - 0.5)
    ax.set_ylim(adjacency.shape[1] - 0.5, -0.5)

    grid = AxisGrid(ax)

    # add groupby indicators starting from last to first
    for i, level in enumerate(groupby[::-1]):
        cax_left = grid.append_axes(
            "left", size=group_axis_size, pad="auto", zorder=len(sort_by) - i
        )
        cax_top = grid.append_axes(
            "top", size=group_axis_size, pad="auto", zorder=len(sort_by) - i
        )
        if group_element == "bracket":
            cax_left.spines[["top", "bottom", "left", "right"]].set_visible(False)
            cax_top.spines[["top", "bottom", "left", "right"]].set_visible(False)

        means = (
            nodes.groupby(groupby[: len(groupby) - i])[pos_key]
            .mean()
            .rename("mean")
            .droplevel(groupby[: len(groupby) - i - 1])
        )
        starts = (
            nodes.groupby(groupby[: len(groupby) - i])[pos_key]
            .min()
            .rename("start")
            .droplevel(groupby[: len(groupby) - i - 1])
        )
        ends = (
            nodes.groupby(groupby[: len(groupby) - i])[pos_key]
            .max()
            .rename("end")
            .droplevel(groupby[: len(groupby) - i - 1])
        )
        info = pd.concat([starts, ends], axis=1)

        for group_name, (start, end) in info.iterrows():
            if group_element == "box":
                draw_box(cax_left, start + 0.5, end, axis="y", color=node_palette[group_name])
                draw_box(cax_top, start + 0.5, end, axis="x", color=node_palette[group_name])
            elif group_element == "bracket":
                draw_bracket(
                    cax_left, start, end, axis="y", color=node_palette[group_name]
                )
                draw_bracket(
                    cax_top, start, end, axis="x", color=node_palette[group_name]
                )

            ax.axhline(start, lw=0.5, alpha=0.5, color="black", zorder=line_zorder)
            ax.axvline(start, lw=0.5, alpha=0.5, color="black", zorder=line_zorder)

            if end == (len(nodes) - 1):
                ax.axhline(
                    len(nodes),
                    lw=0.5,
                    alpha=0.5,
                    color="black",
                    clip_on=False,
                    zorder=line_zorder,
                )
                ax.axvline(
                    len(nodes),
                    lw=0.5,
                    alpha=0.5,
                    color="black",
                    clip_on=False,
                    zorder=line_zorder,
                )

        cax_left.set_yticks(means.values)
        ticklabels = cax_left.set_yticklabels(means.index, rotation=0, fontsize=8)
        for label, color in zip(ticklabels, means.index.map(node_palette)):
            label.set_color(color)

        cax_top.set_xticks(means.values)
        cax_top.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ticklabels = cax_top.set_xticklabels(
            means.index, rotation=45, fontsize=label_fontsize, ha="left"
        )
        for label, color in zip(ticklabels, means.index.map(node_palette)):
            label.set_color(color)
            label.set_in_layout(True)

    # ---------------- Value bars (legacy square path) -----------------
    if row_value_bars or col_value_bars:
        # Create ordered copy of nodes aligned with existing positions
        ordered_nodes = nodes.sort_values(sort_by).reset_index(drop=True)

        def _ensure_numeric(arr: pd.Series) -> pd.Series:
            if np.issubdtype(arr.dtype, np.number):
                return arr
            return pd.Series(pd.Categorical(arr).codes, index=arr.index, name=arr.name)

        def _add_value_bars(axis_side: str, value_cols: list[str], df: pd.DataFrame):
            if not value_cols:
                return
            for col in value_cols:
                if col not in df.columns:
                    raise ValueError(f"Value bar column '{col}' not found in nodes DataFrame.")
                vals = _ensure_numeric(df[col])
                cmap = (
                    value_bar_cmaps[col]
                    if (value_bar_cmaps and col in value_bar_cmaps)
                    else value_bar_default_cmap
                )
                cax = grid.append_axes(
                    "left" if axis_side == "rows" else "top",
                    size=value_bar_size,
                    pad="auto",
                    zorder=0,
                )
                clear_axis(cax)
                if axis_side == "rows":
                    arr = vals.to_numpy().reshape(-1, 1)
                    cax.imshow(arr, aspect="auto", origin="upper", cmap=cmap, interpolation="nearest")
                    cax.set_title(col, fontsize=label_fontsize, rotation=0)
                else:
                    arr = vals.to_numpy().reshape(1, -1)
                    cax.imshow(arr, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
                    cax.set_ylabel(col, fontsize=label_fontsize, rotation=0)

        _add_value_bars("rows", row_value_bars or [], ordered_nodes)
        _add_value_bars("cols", col_value_bars or [], ordered_nodes)

    if xlabel is not None:
        grid.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        grid.set_ylabel(ylabel, fontsize=label_fontsize)
    if title is not None:
        grid.set_title(title, fontsize=title_fontsize)

    if isinstance(arc_labels, tuple):
        draw_label_arc(ax, *arc_labels)

    return ax, grid


# NOTE: this was the code for generating the palette in Workshop1.ipynb
# node_hue = "cell_type"
# n_e_classes = len(proof_cell_df.query("cell_type_coarse == 'E'")[node_hue].unique())
# n_i_classes = len(proof_cell_df.query("cell_type_coarse == 'I'")[node_hue].unique())
# e_colors = sns.cubehelix_palette(
#     start=0.4, rot=0.3, light=0.85, hue=1.0, dark=0.4, gamma=1.3, n_colors=n_e_classes
# )
# i_colors = sns.cubehelix_palette(
#     start=0.3, rot=-0.4, light=0.75, dark=0.2, hue=1.0, gamma=1.3, n_colors=n_i_classes
# )
# cell_type_palette = dict(
#     zip(
#         proof_cell_df.sort_values(["cell_type_coarse", node_hue])[node_hue].unique(),
#         e_colors + i_colors,
#     )
# )
# cell_type_palette["E"] = np.array(list(e_colors)).mean(axis=0)
# cell_type_palette["I"] = np.array(list(i_colors)).mean(axis=0)
cell_type_palette = {
    "L2-IT": [0.9075666881074735, 0.7831311441799196, 0.6949858653492867],
    "L3-IT": [0.8708834148859907, 0.6888410680873137, 0.6022311567741945],
    "L4-IT": [0.8274067443395999, 0.5906550752430283, 0.5233409192850811],
    "L5-ET": [0.7795806757764551, 0.5022849626842287, 0.4643377130216367],
    "L5-IT": [0.719621740596695, 0.41459615740890743, 0.41383679416754976],
    "L5-NP": [0.6539490167887168, 0.3392197573102502, 0.37390086854124416],
    "L6-CT": [0.5747615524284997, 0.26764431745066686, 0.3354259029195752],
    "L6-IT": [0.4927058890394963, 0.20865645456553394, 0.299493097642672],
    "DTC": [0.5054276262176057, 0.7742578554634272, 0.7550447895194092],
    "ITC": [0.307398668309227, 0.535864603420623, 0.6514437955029915],
    "PTC": [0.21347879954025112, 0.2909729895091856, 0.47951242239001035],
    "STC": [0.12751763609942932, 0.10157769757635066, 0.2292792004813278],
    "excitatory": [0.72830947, 0.47437862, 0.46344404],
    "inhibitory": [0.28845568, 0.42566829, 0.52882005],
}


def check_index(
    index: Union[pd.Index, pd.DataFrame, pd.Series, np.ndarray, list],
) -> pd.Index:
    if isinstance(index, (pd.DataFrame, pd.Series)):
        index = index.index
    elif isinstance(index, (np.ndarray, list)):
        index = pd.Index(index)
    else:
        raise TypeError(
            f"Index has to be of type pd.DataFrame, pd.Series, np.ndarray or list; got {type(index)}"
        )
    return index


def filter_synapse_table(
    synapse_table: pd.DataFrame, pre_root_ids=None, post_root_ids=None
):
    """Filter synapse table by pre and post root ids.

    Args:
        synapse_table: synapse table with pre_pt_root_ids and post_pt_root_ids as pd.DataFrame
        pre_root_ids: np.ndarray, list or pd.Series if root_ids to filter on the presynaptic side
        post_root_ids: np.ndarray, list or pd.Series if root_ids to filter on the postsynaptic side

    Returns:
        synapse_table: filtered synapse table
    """

    if pre_root_ids is not None:
        assert isinstance(pre_root_ids, (np.ndarray, list, pd.core.series.Series)), (
            f"IDs have to be of type np.ndarray, list or pd.Series; got {type(pre_root_ids)}"
        )
        pre_mask = np.isin(synapse_table["pre_pt_root_id"], pre_root_ids)
    else:
        pre_mask = np.ones(len(synapse_table), dtype=bool)

    if post_root_ids is not None:
        assert isinstance(post_root_ids, (np.ndarray, list, pd.core.series.Series)), (
            f"IDs have to be of type np.ndarray, list or pd.Series; got {type(pre_root_ids)}"
        )
        post_mask = np.isin(synapse_table["post_pt_root_id"], post_root_ids)
    else:
        post_mask = np.ones(len(synapse_table), dtype=bool)

    return synapse_table[pre_mask & post_mask]
