from pathlib import Path

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from connects_common_connectivity.io import write_models
from connects_common_connectivity.models import DataSet, ReferenceSpaceReference


def test_dataset_defaults_are_optional():
    """Omitted space defaults remain null through a dataset JSON round trip."""
    dataset = DataSet(project_id="p", id="dataset", name="Dataset")
    assert dataset.default_reference_space is None
    assert dataset.default_embedding_space is None
    assert DataSet.model_validate_json(dataset.model_dump_json()) == dataset


@pytest.mark.parametrize("scope", [None, "p"])
@pytest.mark.parametrize("embedding", [None, "connectivity-umap"])
def test_dataset_default_spaces_json_round_trip(scope, embedding):
    """Global and project-local defaults preserve scope and optional embedding IDs in JSON."""
    dataset = DataSet(
        project_id="p",
        id="dataset",
        name="Dataset",
        default_reference_space={"project_id": scope, "id": "atlas"},
        default_embedding_space=embedding,
    )
    assert dataset.default_reference_space == ReferenceSpaceReference(
        project_id=scope, id="atlas",
    )
    assert dataset.default_embedding_space == embedding
    assert DataSet.model_validate_json(dataset.model_dump_json()) == dataset


def test_dataset_default_embedding_without_reference_space():
    """An embedding default is accepted without requiring a spatial default."""
    dataset = DataSet(
        project_id="p", id="dataset", name="Dataset", default_embedding_space="umap",
    )
    assert dataset.default_reference_space is None
    assert dataset.default_embedding_space == "umap"


def test_default_reference_space_omitted_scope_is_global():
    """An omitted reference scope remains global rather than inheriting the dataset's project."""
    dataset = DataSet(
        project_id="p", id="dataset", name="Dataset",
        default_reference_space={"id": "atlas"},
    )
    assert dataset.default_reference_space == ReferenceSpaceReference(id="atlas")
    assert dataset.default_reference_space.project_id is None


@pytest.mark.parametrize("reference", [{}, {"project_id": "p"}, {"id": None}])
def test_default_reference_space_requires_id(reference):
    """A supplied reference with a missing or null space ID fails validation."""
    with pytest.raises(ValidationError, match="id"):
        DataSet(
            project_id="p", id="dataset", name="Dataset",
            default_reference_space=reference,
        )


def test_dataset_default_spaces_delta_round_trip_and_update(tmp_path):
    """Delta preserves scope; updating or clearing defaults leaves other datasets unchanged."""
    datasets = [
        DataSet(
            project_id="p", id="unspecified", name="Unspecified",
        ),
        DataSet(
            project_id="p", id="global", name="Global",
            default_reference_space={"id": "atlas"},
            default_embedding_space="umap",
        ),
        DataSet(
            project_id="p", id="local", name="Local",
            default_reference_space={"project_id": "p", "id": "atlas"},
            default_embedding_space="pca",
        ),
    ]
    result = write_models(datasets, output_root=tmp_path)
    rows = pl.read_delta(result.path)
    assert isinstance(rows.schema["default_reference_space"], pl.Struct)
    assert {
        row["id"]: DataSet.model_validate(row) for row in rows.to_dicts()
    } == {dataset.id: dataset for dataset in datasets}

    updated = DataSet(
        project_id="p", id="global", name="Global",
        default_reference_space={"project_id": "p", "id": "atlas"},
        default_embedding_space="pca",
    )
    write_models(updated, output_root=tmp_path)
    expected = {dataset.id: dataset for dataset in datasets}
    expected[updated.id] = updated
    assert {
        row["id"]: DataSet.model_validate(row)
        for row in pl.read_delta(result.path).to_dicts()
    } == expected

    cleared = DataSet(project_id="p", id="global", name="Global")
    write_models(cleared, output_root=tmp_path)
    expected[cleared.id] = cleared
    assert {
        row["id"]: DataSet.model_validate(row)
        for row in pl.read_delta(result.path).to_dicts()
    } == expected


def test_core_schema_declares_optional_defaults_and_explicit_scope():
    """LinkML declares optional defaults with an inline reference requiring ID but not project."""
    schema = Path(__file__).resolve().parents[1] / "schemas" / "core_schema.yaml"
    contents = yaml.safe_load(schema.read_text())
    for field in ("default_reference_space", "default_embedding_space"):
        assert field in contents["classes"]["DataSet"]["slots"]
        assert contents["slots"][field]["required"] is False
    assert contents["slots"]["default_reference_space"]["inlined"] is True
    reference = contents["classes"]["ReferenceSpaceReference"]
    assert reference["slot_usage"]["id"]["required"] is True
    assert reference["slot_usage"]["project_id"]["required"] is False
