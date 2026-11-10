"""Shared plotting helpers for the ``examples/read_plot`` notebooks.

Read helpers here are thin wrappers for tables that ``connects_common_connectivity.io``
does not (yet) expose a reader for (cluster palettes, cell-to-cluster mappings,
projection measurement matrices). Everything else should go through ``ccc.io``.

Drawing helpers place axes in absolute inches so panels keep the same physical
size (and therefore the same font-to-data ratio) whether drawn alone or composed
into a multi-panel slide figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from connects_common_connectivity.io.path_spec import MODEL_TABLE_PATHS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

INK = "#262626"  # near-black used for text and frames
MUTED = "#6b6b6b"  # secondary text
GRID = "#b5b5b5"  # cluster boundary lines
MISSING = "#ececec"  # NaN feature values

PRESENTATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "pdf.fonttype": 42,  # editable text in PDF / Illustrator / PowerPoint
    "svg.fonttype": "none",
}


def use_presentation_style(**overrides) -> None:
    """Apply slide-friendly rcParams (editable fonts, 300 dpi, no legend frames)."""
    plt.rcParams.update({**PRESENTATION_RC, **overrides})


# ---------------------------------------------------------------------------
# Read helpers (tables without a ccc.io reader)
# ---------------------------------------------------------------------------


def read_cluster_palette(
    output_root: str | Path,
    hierarchy_id: str,
    *,
    level: int | None = None,
) -> pl.DataFrame:
    """Return ``id, parent, level, hex_color`` rows of one cluster hierarchy."""
    clusters = pl.read_delta(str(Path(output_root) / MODEL_TABLE_PATHS["Cluster"]))
    clusters = clusters.filter(pl.col("hierarchy_id") == hierarchy_id)
    if level is not None:
        clusters = clusters.filter(pl.col("level") == level)
    return clusters.select("id", "parent", "level", "hex_color").unique().sort("level", "id")


def read_cell_to_cluster(
    output_root: str | Path,
    *,
    project_id: str,
    mapping_set: str,
    cluster_ids: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return ``dataitem_id, cluster, probability`` from ``CellToClusterMapping``.

    ``cluster_ids`` restricts to one taxonomy level (mappings store every level).
    """
    mapping = pl.read_delta(
        str(Path(output_root) / MODEL_TABLE_PATHS["CellToClusterMapping"])
    ).filter((pl.col("project_id") == project_id) & (pl.col("mapping_set") == mapping_set))
    if cluster_ids is not None:
        mapping = mapping.filter(pl.col("target_cluster").is_in(list(cluster_ids)))
    return mapping.select(
        pl.col("source_cell").alias("dataitem_id"),
        pl.col("target_cluster").alias("cluster"),
        "probability",
    ).unique()


def list_projection_matrices(output_root: str | Path) -> pl.DataFrame:
    """Return the ``ProjectionMeasurementMatrix`` metadata table."""
    table = pl.read_delta(
        str(Path(output_root) / MODEL_TABLE_PATHS["ProjectionMeasurementMatrix"])
    )
    return table.with_columns(
        pl.col("region_index").list.len().alias("n_regions"),
        pl.col("data_item_index").list.len().alias("n_cells"),
    ).drop("region_index", "region_coverage", "data_item_index")


def read_projection_matrix(
    output_root: str | Path, matrix_id: str
) -> tuple[dict, pl.DataFrame]:
    """Return ``(metadata, wide_matrix)`` for one projection matrix.

    The wide matrix has one row per cell (``id``) and one column per region.
    """
    root = Path(output_root)
    meta = pl.read_delta(str(root / MODEL_TABLE_PATHS["ProjectionMeasurementMatrix"]))
    meta = meta.filter(pl.col("id") == matrix_id)
    if meta.height != 1:
        raise KeyError(f"Expected one projection matrix {matrix_id!r}, found {meta.height}.")
    record = meta.row(0, named=True)
    values = pl.read_delta(str(root / record["values"])).unique(subset="id", keep="first")
    return record, values


# ---------------------------------------------------------------------------
# Brain-region grouping (Harris et al. 2019 module order + CCF major divisions)
# ---------------------------------------------------------------------------

# Colour blocks of harris_order_smgrouped.csv -> group name.
HARRIS_GROUP_NAMES = {
    "ff0000": "Prefrontal",
    "ffcc00": "Anterolateral",
    "ff6600": "Somatomotor",
    "007dff": "Visual",
    "5252a9": "Medial",
    "170017": "Auditory",
    "9ad2bd": "Olfactory",
    "7ed04b": "Hippocampal",
    "8ad587": "Cortical subplate",
    "98d6f9": "Striatum & pallidum",
    "ff7080": "Thalamus",
    "e64438": "Hypothalamus",
    "ff64ff": "Midbrain",
    "ff9b88": "Hindbrain",
}
FIBER_TRACTS = ("Fiber tracts", "#b3b3b3")
OTHER = ("Other", "#d9d9d9")

# CCF regions present in WNM projections but absent from the Harris list,
# assigned to the CCF major division they belong to.
EXTRA_REGION_GROUPS = {
    "Olfactory": ["AON", "OLF", "TTd"],
    "Hippocampal": ["HPF", "CA3", "DG-mo", "APr", "POST", "PRE", "PAR"],
    "Cortical subplate": ["CTXsp", "CLA", "BLAa", "BLAp"],
    "Striatum & pallidum": ["STR", "CP", "LSc", "CEAc", "CEAl"],
    "Thalamus": [
        "TH", "LGd-sh", "LGd-co", "LGd-ip", "LGv", "IGL", "IntG", "SubG", "MGd", "MGv",
        "MGm", "POL", "PoT", "Eth", "SPA", "RT", "CL", "PR", "AMd", "LH",
    ],
    "Hypothalamus": ["HY", "PH", "PVi", "SUM", "FF"],
    "Midbrain": [
        "MB", "SCop", "SCsg", "SCzo", "SCig", "SCiw", "SCdg", "SCdw", "APN", "NOT", "OP",
        "PPT", "MPT", "NPC", "PRC", "PBG", "SAG", "ICc", "ICd", "ICe", "CUN", "RN", "PPN",
        "DTN", "LT",
    ],
    "Hindbrain": [
        "P", "PG", "TRN", "PRNr", "PRNc", "NLL", "NTB", "PCG", "LDT", "SLC", "SLD", "NI",
        "CS", "MY", "MARN", "PRP",
    ],
}


def region_groups_table(regions: Sequence[str], harris_csv: str | Path) -> pd.DataFrame:
    """Assign each region a display group, group colour and plotting order.

    Regions listed in ``harris_csv`` keep their Harris order and colour. Other
    CCF regions are appended to the matching group (``EXTRA_REGION_GROUPS``);
    anything left over goes to ``Other`` at the end.
    """
    harris = pd.read_csv(harris_csv)
    harris["hex"] = harris["color"].str.lower().str[:6]
    harris["group"] = harris["hex"].map(HARRIS_GROUP_NAMES)
    group_order = list(dict.fromkeys(harris["group"]))
    group_color = {g: "#" + h for h, g in HARRIS_GROUP_NAMES.items()}
    group_order += [FIBER_TRACTS[0], OTHER[0]]
    group_color[FIBER_TRACTS[0]] = FIBER_TRACTS[1]
    group_color[OTHER[0]] = OTHER[1]

    rank_in_harris = dict(zip(harris["name"], harris["order"]))
    group_of = dict(zip(harris["name"], harris["group"]))
    extra_rank = {}
    for group, names in EXTRA_REGION_GROUPS.items():
        for i, name in enumerate(names):
            group_of.setdefault(name, group)
            extra_rank[name] = 1000 + i
    group_of["fiber tracts"] = FIBER_TRACTS[0]

    rows = []
    for region in regions:
        group = group_of.get(region, OTHER[0])
        rows.append(
            {
                "region": region,
                "group": group,
                "group_color": group_color[group],
                "group_rank": group_order.index(group),
                "within_rank": rank_in_harris.get(region, extra_rank.get(region, 5000)),
            }
        )
    table = pd.DataFrame(rows).sort_values(["group_rank", "within_rank", "region"])
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ordering helpers
# ---------------------------------------------------------------------------


def order_cells(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    cluster_order: Sequence[str],
    within_col: str | None = None,
) -> pd.DataFrame:
    """Sort rows by ``cluster_order`` then (optionally) ``within_col`` ascending."""
    rank = {c: i for i, c in enumerate(cluster_order)}
    out = df.assign(_rank=df[cluster_col].map(rank).fillna(len(rank)))
    keys = ["_rank"] + ([within_col] if within_col else []) + ["dataitem_id"]
    return out.sort_values(keys).drop(columns="_rank").reset_index(drop=True)


def contiguous_blocks(labels: Sequence) -> list[tuple[object, int, int]]:
    """Return ``(label, start, stop)`` runs of identical consecutive labels."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, labels.size]
    return [(labels[s], int(s), int(e)) for s, e in zip(starts, stops)]


# ---------------------------------------------------------------------------
# Low-level drawing primitives
# ---------------------------------------------------------------------------


class InchAxes:
    """Place axes on a figure using inches measured from the bottom-left corner."""

    def __init__(self, fig: plt.Figure):
        self.fig = fig
        self.w, self.h = fig.get_size_inches()

    def add(self, x: float, y: float, w: float, h: float, **kwargs) -> plt.Axes:
        return self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h], **kwargs)

    def text(self, x: float, y: float, s: str, **kwargs):
        return self.fig.text(x / self.w, y / self.h, s, **kwargs)


def _clean(ax: plt.Axes, frame: bool = True) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame)
        spine.set_linewidth(0.6)
        spine.set_color(INK)


def draw_category_strip(
    ax: plt.Axes,
    labels: Sequence,
    palette: dict[str, str],
    *,
    orientation: str = "vertical",
) -> None:
    """Colour strip: one cell per row (vertical) or per column (horizontal)."""
    rgba = np.array([to_rgba(palette.get(v, OTHER[1])) for v in labels])
    img = rgba[:, None, :] if orientation == "vertical" else rgba[None, :, :]
    ax.imshow(img, aspect="auto", interpolation="nearest")
    _clean(ax)


def draw_feature_strip(
    ax: plt.Axes,
    values: Sequence[float],
    *,
    cmap: str = "viridis",
    clip_percentiles: tuple[float, float] = (2, 98),
    orientation: str = "vertical",
) -> mpl.cm.ScalarMappable:
    """Continuous-feature strip, robust-scaled; NaNs drawn light grey."""
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin, vmax = (np.percentile(finite, clip_percentiles) if finite.size else (0, 1))
    cm = mpl.colormaps[cmap].copy()
    cm.set_bad(MISSING)
    img = np.ma.masked_invalid(vals)
    img = img[:, None] if orientation == "vertical" else img[None, :]
    im = ax.imshow(img, aspect="auto", interpolation="nearest", cmap=cm, vmin=vmin, vmax=vmax)
    _clean(ax)
    return im


def strip_title(ax: plt.Axes, text: str, *, fontsize: float = 11, pad_pt: float = 4) -> None:
    """Vertical title above a narrow vertical strip."""
    ax.annotate(
        text, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, pad_pt),
        textcoords="offset points", rotation=90, ha="center", va="bottom",
        fontsize=fontsize, color=INK,
    )


def draw_block_boundaries(
    ax: plt.Axes,
    labels: Sequence,
    *,
    axis: str,
    major_of: dict | None = None,
    minor_kw: dict | None = None,
    major_kw: dict | None = None,
) -> None:
    """Thin lines between cluster blocks; thicker where ``major_of`` changes."""
    minor_kw = {"color": GRID, "lw": 0.35, "alpha": 0.9, **(minor_kw or {})}
    major_kw = {"color": MUTED, "lw": 0.9, **(major_kw or {})}
    blocks = contiguous_blocks(labels)
    for (prev, _, _), (cur, start, _) in zip(blocks[:-1], blocks[1:]):
        is_major = major_of is not None and major_of.get(prev) != major_of.get(cur)
        kw = major_kw if is_major else minor_kw
        line = ax.axhline if axis == "rows" else ax.axvline
        line(start - 0.5, zorder=3, **kw)


def draw_block_labels(
    ax: plt.Axes,
    labels: Sequence,
    *,
    side: str,
    extent_in: float,
    min_block_in: float = 0.14,
    fontsize: float = 10,
    pad: float = 0.012,
    color: str = INK,
) -> None:
    """Label each cluster block at its centre, skipping blocks too small to read.

    ``side='right'`` labels rows to the right of ``ax``; ``side='top'`` labels
    columns above ``ax`` (rotated 90). ``extent_in`` is the axis length in inches.
    """
    labels = np.asarray(labels)
    n = labels.size
    for label, start, stop in contiguous_blocks(labels):
        if (stop - start) / n * extent_in < min_block_in:
            continue
        mid = (start + stop - 1) / 2
        if side == "right":
            tr = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1 + pad, mid, str(label), transform=tr, ha="left", va="center",
                    fontsize=fontsize, color=color, clip_on=False)
        else:
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(mid, 1 + pad * 4, str(label), transform=tr, ha="center", va="bottom",
                    rotation=90, fontsize=fontsize, color=color, clip_on=False)


def draw_connectivity_scatter(
    ax: plt.Axes,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    size: float = 0.35,
    color: str = "black",
    alpha: float = 0.75,
) -> None:
    """Adjacency scatter (one square marker per connected pair), rasterised."""
    ax.scatter(cols, rows, s=size, c=color, marker="s", linewidths=0, alpha=alpha,
               rasterized=True)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    _clean(ax)


def draw_projection_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap: str = "magma_r",
    vmin: float = 100,
    vmax: float | None = None,
) -> mpl.cm.ScalarMappable:
    """Log-scaled heatmap; zeros (no projection) are left white."""
    vmax = vmax or float(np.nanmax(matrix))
    # Drop the near-white end of the colormap so weak projections stay
    # distinguishable from "no projection" (white).
    base = mpl.colormaps[cmap]
    cm = mpl.colors.ListedColormap(base(np.linspace(0.10, 0.97, 256)), name=f"{cmap}_trim")
    cm.set_bad("white")
    cm.set_under(cm(0.0))
    img = np.ma.masked_less_equal(matrix, 0)
    im = ax.imshow(img, aspect="auto", interpolation="nearest", cmap=cm,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    _clean(ax)
    return im


def add_legend(
    ax: plt.Axes,
    sections: Sequence[tuple[str | None, Sequence[tuple[str, str]]]],
    *,
    title: str | None = None,
    fontsize: float = 11,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (0, 1),
    ncol: int = 1,
    **kwargs,
) -> mpl.legend.Legend:
    """Patch legend with optional bold section headers (e.g. Excitatory / Inhibitory)."""
    handles, header_idx = [], []
    for header, items in sections:
        if header:
            header_idx.append(len(handles))
            handles.append(Patch(facecolor="none", edgecolor="none", label=header))
        handles += [Patch(facecolor=c, edgecolor="none", label=l) for l, c in items]
    leg = ax.legend(
        handles=handles, title=title, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol,
        fontsize=fontsize, title_fontsize=fontsize + 1, handlelength=1.1, handleheight=1.0,
        handletextpad=0.5, labelspacing=0.32, borderaxespad=0, alignment="left",
        columnspacing=1.2, **kwargs,
    )
    if title:
        leg.get_title().set_fontweight("bold")
    texts = leg.get_texts()
    for i in header_idx:
        texts[i].set_fontweight("bold")
        texts[i].set_color(MUTED)
        texts[i].set_x(-22)  # pull header left over the (empty) swatch
    ax.axis("off")
    return leg


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


@dataclass
class StripSpec:
    """A vertical annotation strip to the left of a matrix."""

    column: str
    title: str
    kind: str = "feature"  # "feature" | "category"
    cmap: str = "viridis"
    palette: dict[str, str] = field(default_factory=dict)


@dataclass
class PanelGeometry:
    """Sizes in inches (fonts in points) shared by both panel types."""

    main_w: float = 5.2
    main_h: float = 5.2
    strip_w: float = 0.24
    strip_gap: float = 0.05
    strips_to_main: float = 0.10
    top_strip_h: float = 0.20
    top_strip_gap: float = 0.05
    strip_title_size: float = 11
    block_label_size: float = 10
    axis_label_size: float = 12
    caption_size: float = 15
    caption_drop: float = 1.15  # caption baseline, inches below the matrix
    row_labels: bool = True  # inline cluster names to the right of the matrix


def _draw_left_strips(
    canvas: InchAxes,
    meta: pd.DataFrame,
    strips: Sequence[StripSpec],
    *,
    x_right: float,
    y0: float,
    g: PanelGeometry,
) -> tuple[dict[str, plt.Axes], float]:
    """Draw strips right-to-left ending at ``x_right``; return axes and left x."""
    axes = {}
    x = x_right
    for spec in reversed(strips):
        x -= g.strip_w
        ax = canvas.add(x, y0, g.strip_w, g.main_h)
        if spec.kind == "category":
            draw_category_strip(ax, meta[spec.column].to_numpy(), spec.palette)
        else:
            draw_feature_strip(ax, meta[spec.column].to_numpy(), cmap=spec.cmap)
        strip_title(ax, spec.title, fontsize=g.strip_title_size)
        axes[spec.column] = ax
        x -= g.strip_gap
    return axes, x + g.strip_gap


def plot_connectivity_panel(
    fig: plt.Figure,
    *,
    origin: tuple[float, float],
    row_meta: pd.DataFrame,
    col_meta: pd.DataFrame,
    edges: pd.DataFrame,
    cluster_col: str,
    palette: dict[str, str],
    strips: Sequence[StripSpec],
    class_of: dict[str, str] | None = None,
    geometry: PanelGeometry | None = None,
    marker_size: float = 0.35,
    row_label: str = "Presynaptic cells",
    col_label: str = "Postsynaptic cells",
    caption: str | None = None,
) -> dict[str, plt.Axes]:
    """Pre (rows) x post (columns) connectivity scatter with annotation strips.

    ``origin`` is the bottom-left corner of the scatter in inches. ``edges``
    needs ``presynaptic_cell`` / ``postsynaptic_cell`` columns.
    """
    g = geometry or PanelGeometry()
    canvas = InchAxes(fig)
    x0, y0 = origin

    row_pos = pd.Series(np.arange(len(row_meta)), index=row_meta["dataitem_id"])
    col_pos = pd.Series(np.arange(len(col_meta)), index=col_meta["dataitem_id"])
    e = edges[edges["presynaptic_cell"].isin(row_pos.index)
              & edges["postsynaptic_cell"].isin(col_pos.index)]
    rows = row_pos.loc[e["presynaptic_cell"]].to_numpy()
    cols = col_pos.loc[e["postsynaptic_cell"]].to_numpy()

    axes: dict[str, plt.Axes] = {}
    main = canvas.add(x0, y0, g.main_w, g.main_h)
    draw_connectivity_scatter(main, rows, cols, n_rows=len(row_meta), n_cols=len(col_meta),
                              size=marker_size)
    row_labels = row_meta[cluster_col].to_numpy()
    col_labels = col_meta[cluster_col].to_numpy()
    draw_block_boundaries(main, row_labels, axis="rows", major_of=class_of)
    draw_block_boundaries(main, col_labels, axis="cols", major_of=class_of)
    if g.row_labels:
        draw_block_labels(main, row_labels, side="right", extent_in=g.main_h,
                          fontsize=g.block_label_size)
    axes["main"] = main

    top = canvas.add(x0, y0 + g.main_h + g.top_strip_gap, g.main_w, g.top_strip_h)
    draw_category_strip(top, col_labels, palette, orientation="horizontal")
    draw_block_labels(top, col_labels, side="top", extent_in=g.main_w,
                      fontsize=g.block_label_size, min_block_in=0.16)
    axes["top"] = top

    strip_axes, x_left = _draw_left_strips(
        canvas, row_meta, strips, x_right=x0 - g.strips_to_main, y0=y0, g=g)
    axes.update(strip_axes)

    canvas.text(x_left - 0.08, y0 + g.main_h / 2,
                f"{row_label}  (n = {len(row_meta):,})", rotation=90, ha="right",
                va="center", fontsize=g.axis_label_size)
    canvas.text(x0 + g.main_w / 2, y0 - 0.10, f"{col_label}  (n = {len(col_meta):,})",
                ha="center", va="top", fontsize=g.axis_label_size)
    if caption:
        canvas.text(x0 + g.main_w / 2, y0 - g.caption_drop, caption, ha="center", va="top",
                    fontsize=g.caption_size, fontweight="bold")
    axes["n_edges"] = len(e)  # type: ignore[assignment]
    return axes


def plot_projection_panel(
    fig: plt.Figure,
    *,
    origin: tuple[float, float],
    row_meta: pd.DataFrame,
    matrix: np.ndarray,
    regions: pd.DataFrame,
    cluster_col: str,
    strips: Sequence[StripSpec],
    geometry: PanelGeometry | None = None,
    cmap: str = "magma_r",
    vmin: float = 100,
    vmax: float | None = None,
    region_label_size: float = 5.5,
    colorbar_label: str = "Axon length (µm)",
    caption: str | None = None,
) -> dict[str, plt.Axes]:
    """Cell (rows) x brain-region (columns) projection heatmap with strips.

    ``regions`` is the output of :func:`region_groups_table` restricted to the
    columns of ``matrix`` (same order).
    """
    g = geometry or PanelGeometry()
    canvas = InchAxes(fig)
    x0, y0 = origin
    axes: dict[str, plt.Axes] = {}

    main = canvas.add(x0, y0, g.main_w, g.main_h)
    im = draw_projection_heatmap(main, matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    row_labels = row_meta[cluster_col].to_numpy()
    draw_block_boundaries(main, row_labels, axis="rows",
                          minor_kw={"color": "#9a9a9a", "lw": 0.45})
    draw_block_boundaries(main, regions["group"].to_numpy(), axis="cols",
                          minor_kw={"color": "#9a9a9a", "lw": 0.45})
    if g.row_labels:
        draw_block_labels(main, row_labels, side="right", extent_in=g.main_h,
                          fontsize=g.block_label_size)
    axes["main"] = main

    top = canvas.add(x0, y0 + g.main_h + g.top_strip_gap, g.main_w, g.top_strip_h)
    group_palette = dict(zip(regions["group"], regions["group_color"]))
    draw_category_strip(top, regions["group"].to_numpy(), group_palette,
                        orientation="horizontal")
    top.set_xticks(np.arange(len(regions)))
    top.set_xticklabels(regions["region"], rotation=90, fontsize=region_label_size)
    top.xaxis.set_ticks_position("top")
    top.tick_params(axis="x", length=0, pad=1.5)
    axes["top"] = top

    strip_axes, x_left = _draw_left_strips(
        canvas, row_meta, strips, x_right=x0 - g.strips_to_main, y0=y0, g=g)
    axes.update(strip_axes)
    canvas.text(x_left - 0.08, y0 + g.main_h / 2, f"WNM cells  (n = {len(row_meta):,})",
                rotation=90, ha="right", va="center", fontsize=g.axis_label_size)

    cax = canvas.add(x0, y0 - 0.22, min(2.0, g.main_w * 0.4), 0.09)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=g.block_label_size - 1, length=2, width=0.5, pad=1.5)
    cb.set_label(colorbar_label, fontsize=g.block_label_size, labelpad=2)
    axes["colorbar"] = cax

    if caption:
        canvas.text(x0 + g.main_w / 2, y0 - g.caption_drop, caption, ha="center", va="top",
                    fontsize=g.caption_size, fontweight="bold")
    return axes
