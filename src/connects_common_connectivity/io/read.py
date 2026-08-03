"""Read helpers for the IO layer.

Where :mod:`connects_common_connectivity.io.writers` owns the write path, this
module owns the read path. 

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from connects_common_connectivity.config import Settings, get_settings

DATASET_SUBDIR = "dataset"
DATAITEM_DATASET_ASSOCIATION_SUBDIR = "dataitem_dataset_association"
CELL_FEATURE_SET_SUBDIR = "cellfeatureset"
CELL_FEATURE_MATRIX_SUBDIR = "cellfeaturematrix"
CELL_FEATURES_SUBDIR = "cellfeatures"
CLUSTER_HIERARCHY_SUBDIR = "clusterhierarchy"
CLUSTER_SUBDIR = "cluster"
CLUSTER_MEMBERSHIP_SUBDIR = "clustermembership"
SYNAPSE_SUBDIR = "synapse"
SYNAPSE_FEATURES_SUBDIR = "synapsefeatures"

__all__ = [
    "DatasetReader",
    "read_synapse_table",
    "SYNAPSE_SUBDIR",
    "SYNAPSE_FEATURES_SUBDIR",
]

_FEATURESET_DISPLAY_SCHEMA = {
    "feature_set_id": pl.String,
    "matrix_id": pl.String,
    "description": pl.String,
    "extraction_method": pl.String,
    "feature_definition_ids": pl.List(pl.String),
    "cell_index_column": pl.String,
}
_CLUSTERSET_DISPLAY_SCHEMA = {
    "clusterset_id": pl.String,
    "run": pl.String,
    "root": pl.String,
    "clusters": pl.List(pl.String),
}


class DatasetReader:
    """Read dataset-centric wide tables from a Common Connectivity root.

    Feature sets and cluster hierarchies are discovered by overlap with the
    data items associated with a dataset. This is necessary because neither
    schema directly references a :class:`DataSet`.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist or is not a directory: "
                f"{self.dataset_root}"
            )

        self._datasets = self._read_table(DATASET_SUBDIR, required=True)
        self._associations = self._read_table(
            DATAITEM_DATASET_ASSOCIATION_SUBDIR,
            required=True,
        )

    def display_dataset_names(self) -> pl.DataFrame:
        """Return available dataset IDs and their descriptive metadata."""
        columns = [
            column
            for column in ("id", "name", "project_id", "modality")
            if column in self._datasets.columns
        ]
        return self._datasets.select(columns).sort(["project_id", "id"])

    def dataset_dataitem_ids(self, dataset_name: str) -> pl.DataFrame:
        """Return the distinct data-item IDs associated with a dataset."""
        dataset = self._dataset_record(dataset_name)
        return (
            self._associations.filter(
                (pl.col("dataset_id") == dataset_name)
                & (pl.col("project_id") == dataset["project_id"])
            )
            .select(pl.col("dataitem_id"))
            .unique()
            .sort("dataitem_id")
        )

    def discover_featuresets(self, dataset_name: str) -> pl.DataFrame:
        """Return feature sets whose matrices overlap the dataset's items."""
        dataset = self._dataset_record(dataset_name)
        items = self.dataset_dataitem_ids(dataset_name)
        matrices = self._read_table(CELL_FEATURE_MATRIX_SUBDIR)
        if matrices is None:
            return pl.DataFrame(schema=_FEATURESET_DISPLAY_SCHEMA)

        matrices = matrices.filter(pl.col("project_id") == dataset["project_id"])
        if matrices.is_empty():
            return pl.DataFrame(schema=_FEATURESET_DISPLAY_SCHEMA)

        feature_sets = self._read_table(CELL_FEATURE_SET_SUBDIR, required=True)
        feature_sets = feature_sets.filter(
            pl.col("project_id") == dataset["project_id"]
        )
        feature_set_rows = {
            row["id"]: row for row in feature_sets.iter_rows(named=True)
        }

        related: list[dict[str, Any]] = []
        for matrix_row in matrices.iter_rows(named=True):
            feature_set_id = matrix_row.get("feature_set_id")
            matrix_id = matrix_row.get("id")
            index_column = matrix_row.get("cell_index_column")
            if not feature_set_id:
                raise ValueError(
                    f"CellFeatureMatrix {matrix_id!r} has no feature_set_id."
                )
            if not index_column:
                raise ValueError(
                    f"CellFeatureMatrix {matrix_id!r} has no cell_index_column."
                )
            if feature_set_id not in feature_set_rows:
                raise ValueError(
                    f"CellFeatureMatrix {matrix_id!r} references missing "
                    f"CellFeatureSet {feature_set_id!r}."
                )

            feature_frame = self._read_feature_matrix(
                feature_set_id,
                index_column,
                str(dataset["project_id"]),
            )
            overlap = (
                feature_frame.select(
                    pl.col(index_column).alias("dataitem_id")
                )
                .join(items, on="dataitem_id", how="inner")
                .height
            )
            if not overlap:
                continue

            feature_set = feature_set_rows[feature_set_id]
            related.append(
                {
                    "feature_set_id": feature_set_id,
                    "matrix_id": matrix_id,
                    "description": feature_set.get("description"),
                    "extraction_method": feature_set.get("extraction_method"),
                    "feature_definition_ids": feature_set.get(
                        "feature_definition_ids"
                    ),
                    "cell_index_column": index_column,
                }
            )

        return pl.DataFrame(
            related,
            schema=_FEATURESET_DISPLAY_SCHEMA,
        ).sort("feature_set_id")

    def display_featuresets(self, dataset_name: str) -> pl.DataFrame:
        """Display feature sets related to a dataset."""
        return self.discover_featuresets(dataset_name)

    def discover_clustersets(self, dataset_name: str) -> pl.DataFrame:
        """Return cluster hierarchies with memberships for dataset items."""
        dataset = self._dataset_record(dataset_name)
        items = self.dataset_dataitem_ids(dataset_name).rename(
            {"dataitem_id": "item"}
        )
        memberships = self._read_table(CLUSTER_MEMBERSHIP_SUBDIR)
        if memberships is None:
            return pl.DataFrame(schema=_CLUSTERSET_DISPLAY_SCHEMA)

        related_ids = (
            memberships.filter(pl.col("project_id") == dataset["project_id"])
            .join(items, on="item", how="inner")
            .select("hierarchy_id")
            .unique()
        )
        if related_ids.is_empty():
            return pl.DataFrame(schema=_CLUSTERSET_DISPLAY_SCHEMA)

        hierarchies = self._read_table(CLUSTER_HIERARCHY_SUBDIR, required=True)
        related = related_ids.join(
            hierarchies,
            left_on="hierarchy_id",
            right_on="id",
            how="left",
        )
        missing = related.filter(pl.col("root").is_null())["hierarchy_id"].to_list()
        if missing:
            raise ValueError(
                "ClusterMembership rows reference missing ClusterHierarchy "
                f"IDs: {sorted(missing)}"
            )

        return (
            related.select(
                pl.col("hierarchy_id").alias("clusterset_id"),
                "run",
                "root",
                "clusters",
            )
            .sort("clusterset_id")
        )

    def display_clustersets(self, dataset_name: str) -> pl.DataFrame:
        """Display cluster hierarchies related to a dataset."""
        return self.discover_clustersets(dataset_name)

    def read_dataset(
        self,
        dataset_name: str,
        *,
        featuresets: str | Iterable[str] | None = None,
        clustersets: str | Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Return one wide row per data item in ``dataset_name``.

        ``None`` selects every related feature or cluster set. A string selects
        one set, an iterable selects several, and an empty iterable selects
        none.
        """
        dataset = self._dataset_record(dataset_name)
        result = self.dataset_dataitem_ids(dataset_name)

        available_features = self.discover_featuresets(dataset_name)
        selected_features = self._select_related_rows(
            available_features,
            featuresets,
            id_column="feature_set_id",
            label="feature set",
        )
        seen_feature_columns: dict[str, str] = {}
        for feature_row in selected_features.iter_rows(named=True):
            feature_frame = self._feature_join_frame(
                feature_row,
                str(dataset["project_id"]),
                result.select("dataitem_id"),
            )
            feature_columns = [
                column for column in feature_frame.columns if column != "dataitem_id"
            ]
            conflicts = {
                column: seen_feature_columns[column]
                for column in feature_columns
                if column in seen_feature_columns
            }
            if conflicts:
                details = ", ".join(
                    f"{column!r} (also in {owner!r})"
                    for column, owner in sorted(conflicts.items())
                )
                raise ValueError(
                    f"Feature set {feature_row['feature_set_id']!r} has duplicate "
                    f"feature columns: {details}."
                )
            seen_feature_columns.update(
                {
                    column: str(feature_row["feature_set_id"])
                    for column in feature_columns
                }
            )
            result = result.join(feature_frame, on="dataitem_id", how="left")

        available_clusters = self.discover_clustersets(dataset_name)
        selected_clusters = self._select_related_rows(
            available_clusters,
            clustersets,
            id_column="clusterset_id",
            label="cluster set",
        )
        for cluster_row in selected_clusters.iter_rows(named=True):
            cluster_frame = self._cluster_join_frame(
                str(cluster_row["clusterset_id"]),
                str(dataset["project_id"]),
                result.select("dataitem_id"),
            )
            result = result.join(cluster_frame, on="dataitem_id", how="left")

        return result.sort("dataitem_id")

    def _dataset_record(self, dataset_name: str) -> dict[str, Any]:
        matches = self._datasets.filter(pl.col("id") == dataset_name)
        if matches.is_empty():
            available = self._datasets["id"].unique().sort().to_list()
            raise KeyError(
                f"Unknown dataset {dataset_name!r}. Available dataset IDs: "
                f"{available}"
            )
        if matches.height > 1:
            projects = matches["project_id"].unique().sort().to_list()
            raise ValueError(
                f"Dataset ID {dataset_name!r} is ambiguous across projects: "
                f"{projects}"
            )
        return matches.row(0, named=True)

    def _read_table(
        self,
        subdir: str,
        *,
        required: bool = False,
    ) -> pl.DataFrame | None:
        path = self.dataset_root / subdir
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required Delta table is missing: {path}")
            return None
        return pl.read_delta(str(path))

    def _read_feature_matrix(
        self,
        feature_set_id: str,
        index_column: str,
        project_id: str,
    ) -> pl.DataFrame:
        path = self.dataset_root / CELL_FEATURES_SUBDIR / feature_set_id
        if not path.exists():
            raise FileNotFoundError(
                f"Feature matrix for {feature_set_id!r} is missing: {path}"
            )
        frame = pl.read_delta(str(path))
        if index_column not in frame.columns:
            raise ValueError(
                f"Feature matrix {feature_set_id!r} does not contain its "
                f"configured index column {index_column!r}. Available columns: "
                f"{frame.columns}"
            )
        if "project_id" in frame.columns:
            frame = frame.filter(pl.col("project_id") == project_id)
        return frame

    def _feature_join_frame(
        self,
        feature_row: dict[str, Any],
        project_id: str,
        items: pl.DataFrame,
    ) -> pl.DataFrame:
        feature_set_id = str(feature_row["feature_set_id"])
        index_column = str(feature_row["cell_index_column"])
        frame = self._read_feature_matrix(
            feature_set_id,
            index_column,
            project_id,
        )
        if index_column != "dataitem_id":
            frame = frame.rename({index_column: "dataitem_id"})
        frame = frame.join(items, on="dataitem_id", how="inner")

        duplicates = (
            frame.group_by("dataitem_id")
            .len()
            .filter(pl.col("len") > 1)
            .select("dataitem_id")
        )
        if not duplicates.is_empty():
            examples = duplicates["dataitem_id"].head(5).to_list()
            raise ValueError(
                f"Feature matrix {feature_set_id!r} has multiple rows for "
                f"data-item IDs such as {examples}."
            )

        metadata_columns = {"project_id", "feature_set_id"}
        return frame.drop(
            [
                column
                for column in metadata_columns
                if column in frame.columns
            ]
        )

    def _cluster_join_frame(
        self,
        hierarchy_id: str,
        project_id: str,
        items: pl.DataFrame,
    ) -> pl.DataFrame:
        memberships = self._read_table(CLUSTER_MEMBERSHIP_SUBDIR, required=True)
        clusters = self._read_table(CLUSTER_SUBDIR, required=True)
        item_keys = items.rename({"dataitem_id": "item"})
        assignments = (
            memberships.filter(
                (pl.col("project_id") == project_id)
                & (pl.col("hierarchy_id") == hierarchy_id)
            )
            .join(item_keys, on="item", how="inner")
            .join(
                clusters.filter(pl.col("hierarchy_id") == hierarchy_id).select(
                    "hierarchy_id",
                    pl.col("id").alias("cluster"),
                    "level",
                ),
                on=["hierarchy_id", "cluster"],
                how="left",
            )
        )

        missing_clusters = assignments.filter(pl.col("level").is_null())
        if not missing_clusters.is_empty():
            missing = missing_clusters["cluster"].unique().sort().to_list()
            raise ValueError(
                f"Cluster set {hierarchy_id!r} has memberships referencing "
                f"unknown clusters or clusters without levels: {missing}"
            )

        assignments = assignments.select("item", "cluster", "level").unique()
        ambiguous = (
            assignments.group_by("item", "level")
            .agg(pl.col("cluster").n_unique().alias("cluster_count"))
            .filter(pl.col("cluster_count") > 1)
        )
        if not ambiguous.is_empty():
            examples = ambiguous.select("item", "level").head(5).to_dicts()
            raise ValueError(
                f"Cluster set {hierarchy_id!r} assigns multiple clusters at "
                f"the same level for data items such as {examples}."
            )

        levels = assignments["level"].unique().sort().to_list()
        pivoted = assignments.pivot(
            "level",
            index="item",
            values="cluster",
            aggregate_function="first",
        ).select("item", *[str(level) for level in levels])
        pivoted = pivoted.rename({"item": "dataitem_id"})
        level_columns = [
            column for column in pivoted.columns if column != "dataitem_id"
        ]
        return pivoted.rename(
            {
                column: f"{hierarchy_id}_level_{column}"
                for column in level_columns
            }
        )

    @staticmethod
    def _select_related_rows(
        available: pl.DataFrame,
        requested: str | Iterable[str] | None,
        *,
        id_column: str,
        label: str,
    ) -> pl.DataFrame:
        if requested is None:
            return available
        names = [requested] if isinstance(requested, str) else list(requested)
        names = list(dict.fromkeys(names))
        available_names = available[id_column].to_list()
        unknown = sorted(set(names) - set(available_names))
        if unknown:
            raise KeyError(
                f"Unknown {label} names: {unknown}. Available: "
                f"{sorted(available_names)}"
            )
        return available.filter(pl.col(id_column).is_in(names))


def _resolve_output_root(
    settings: Settings | None,
    output_root: str | Path | None,
) -> Path:
    """Resolve the on-disk root, mirroring the writers' precedence.

    Explicit ``output_root=`` wins; otherwise ``settings`` (or the discovered
    ``ccc_config.yaml`` via :func:`get_settings`) supplies ``output_root``.
    Passing both is an error, matching :func:`write_models`.
    """
    if output_root is not None and settings is not None:
        raise TypeError(
            "Pass either settings= or output_root=, not both. "
            "output_root= is the per-call override; settings= carries the "
            "full Settings object."
        )
    if output_root is not None:
        return Path(output_root)
    return Path((settings or get_settings()).output_root)


def read_synapse_table(
    project_id: str,
    *,
    dataset_id: str | None = None,
    features: bool | Iterable[str] = False,
    feature_matrix_id: str | None = None,
    synapse_index_column: str = "id",
    output_root: str | Path | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Read the long single-synapse table, optionally with feature columns.

    Parameters
    ----------
    project_id:
        Project scope to read. Always required — the long table is
        ``ProjectScoped``.
    dataset_id:
        Optional additional filter. When given, only synapses whose
        ``dataset_id`` matches are returned. Pass it whenever a project owns
        more than one dataset of synapses.
    features:
        Controls the feature LEFT-join.

        - ``False`` (default): return only the long connectivity columns.
        - ``True``: append **all** feature columns from the feature table.
        - an iterable of column names: append only those feature columns (the
          synapse-id join key is always retained).
    feature_matrix_id:
        Subdirectory name under ``synapsefeatures/`` identifying which wide
        feature table to join. Required when ``features`` is truthy.
    synapse_index_column:
        Name of the synapse-id column in the feature table (default ``"id"``).
        Matches ``SynapseFeatureMatrix.synapse_index_column``.
    output_root, settings:
        On-disk root resolution — same semantics (and mutual exclusion) as
        :func:`connects_common_connectivity.io.write_models`.

    Returns
    -------
    polars.DataFrame
        The long synapse table, with feature columns appended when requested.
        Rows without a matching feature row keep null feature values.
    """
    root = _resolve_output_root(settings, output_root)

    long_path = root / SYNAPSE_SUBDIR
    if not long_path.exists():
        raise FileNotFoundError(
            f"No synapse table at {long_path}. Write SynapseConnectivityLong "
            f"rows first (see code/etl_v1dd_03_synapses.ipynb)."
        )

    synapses = pl.read_delta(str(long_path)).filter(pl.col("project_id") == project_id)
    if dataset_id is not None:
        synapses = synapses.filter(pl.col("dataset_id") == dataset_id)

    if not features:
        return synapses

    if feature_matrix_id is None:
        raise ValueError(
            "features=... requires feature_matrix_id to identify which "
            f"'{SYNAPSE_FEATURES_SUBDIR}/<id>/' wide table to join."
        )

    feature_path = root / SYNAPSE_FEATURES_SUBDIR / feature_matrix_id
    if not feature_path.exists():
        raise FileNotFoundError(f"No synapse feature table at {feature_path}.")

    feature_df = pl.read_delta(str(feature_path)).filter(
        pl.col("project_id") == project_id
    )
    if dataset_id is not None and "dataset_id" in feature_df.columns:
        feature_df = feature_df.filter(pl.col("dataset_id") == dataset_id)

    # Normalize the feature join key to the long table's synapse id column.
    if synapse_index_column != "id":
        feature_df = feature_df.rename({synapse_index_column: "id"})

    feature_df = _select_feature_columns(feature_df, features)

    return synapses.join(feature_df, on="id", how="left")


def _select_feature_columns(
    feature_df: pl.DataFrame,
    features: bool | Iterable[str],
) -> pl.DataFrame:
    """Return the feature frame reduced to the id key plus requested columns.

    ``features is True`` keeps every column except scope columns that already
    live on the long table (``project_id``/``dataset_id``), which would
    otherwise collide on the join. An explicit iterable keeps only the named
    columns (plus the ``id`` key).
    """
    drop_on_join = {"project_id", "dataset_id"}
    if features is True:
        keep = [c for c in feature_df.columns if c == "id" or c not in drop_on_join]
        return feature_df.select(keep)

    requested: Sequence[str] = list(features)  # type: ignore[arg-type]
    missing = [c for c in requested if c not in feature_df.columns]
    if missing:
        raise KeyError(
            f"Requested feature columns not found: {missing}. "
            f"Available: {sorted(set(feature_df.columns) - {'id'})}"
        )
    keep = ["id", *[c for c in requested if c != "id"]]
    return feature_df.select(keep)
