"""Read helpers for the IO layer.

Where :mod:`connects_common_connectivity.io.writers` owns the write path, this
module owns the read path. The first reader is :func:`read_synapse_table`,
which returns the long single-synapse connectivity table and, on request,
LEFT-joins per-synapse feature columns (position, size, ``synaptictargetlabel``,
...) from a wide-form feature Parquet.

The on-disk layout mirrors the ``SynapseConnectivityLong`` /
``SynapseFeatureMatrix`` schema pair:

- ``<output_root>/synapse/`` — long Delta table, one row per synapse
  (``id``, ``presynaptic_cell``, ``postsynaptic_cell``, ``dataset_id``,
  ``project_id``).
- ``<output_root>/synapsefeatures/<feature_matrix_id>/`` — wide Delta table,
  one row per synapse keyed by the synapse id, one column per feature.

Feature rows are LEFT-joined on the synapse id, so a synapse that is absent
from the feature table simply yields null feature columns (e.g. the
``synaptictargetlabel`` is missing for a subset of synapses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from connects_common_connectivity.config import Settings, get_settings

SYNAPSE_SUBDIR = "synapse"
SYNAPSE_FEATURES_SUBDIR = "synapsefeatures"

__all__ = ["read_synapse_table", "SYNAPSE_SUBDIR", "SYNAPSE_FEATURES_SUBDIR"]


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
