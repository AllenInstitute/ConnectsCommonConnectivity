"""Write-time, pydantic-only validation hooked into :func:`write_models`.

The IO layer should never blindly trust that a model carries every slot
the write actually depends on. Many generated fields are ``Optional`` in
``models.py`` because the schema permits them to be missing in some
contexts, but the *write* path needs them concretely (e.g. the predicate
columns, the partition columns, the id used for dedupe).

The ``*Write`` subclasses here define required-field and cross-field row
constraints not enforced by the generated models. The :class:`WriteSpec` for
each writable class selects one through ``validation_cls``. This module
re-validates every instance through it before any IO during runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError, model_validator

from connects_common_connectivity.models import (
    CellFeatureDefinition,
    Cluster,
    ClusterMembership,
    HierarchyCategory,
    ReferenceSpace,
    SignedAxis,
    Unit,
)

if TYPE_CHECKING:
    from connects_common_connectivity.io.write_spec import WriteSpec

__all__ = [
    "DATA_AXIS_BY_SIGNED_AXIS",
    "CellFeatureDefinitionWrite",
    "ClusterMembershipWrite",
    "ClusterWrite",
    "HierarchyCategoryWrite",
    "ReferenceSpaceWrite",
    "validate_for_write",
]


DATA_AXIS_BY_SIGNED_AXIS: dict[SignedAxis, str] = {
    SignedAxis.PLUS_X: "X",
    SignedAxis.MINUS_X: "X",
    SignedAxis.PLUS_Y: "Y",
    SignedAxis.MINUS_Y: "Y",
    SignedAxis.PLUS_Z: "Z",
    SignedAxis.MINUS_Z: "Z",
}
"""Unsigned data axis carried by each signed axis, independent of direction."""


class ClusterWrite(Cluster):
    """``Cluster`` with the taxonomy scope the shared cluster table needs.

    Attributes
    ----------
    hierarchy_id:
        Owning taxonomy. Optional in the schema because a cluster is
        meaningful without one, but required here because it partitions the
        table and forms part of the merge identity.
    """

    hierarchy_id: str


class ClusterMembershipWrite(ClusterMembership):
    """``ClusterMembership`` with its complete row identity present.

    Attributes
    ----------
    hierarchy_id:
        Taxonomy that disambiguates memberships when one project has rows
        against several hierarchies.
    item:
        Member data item.
    cluster:
        Cluster the item belongs to.
    """

    hierarchy_id: str
    item: str
    cluster: str


class CellFeatureDefinitionWrite(CellFeatureDefinition):
    """``CellFeatureDefinition`` bound to the feature set it describes.

    Attributes
    ----------
    feature_set_id:
        Owning feature set. It partitions the table and forms part of the
        merge identity, so a null would merge definitions across sets.
    """

    feature_set_id: str


class HierarchyCategoryWrite(HierarchyCategory):
    """``HierarchyCategory`` with the taxonomy scope its table is keyed by.

    Attributes
    ----------
    hierarchy_id:
        Owning taxonomy, which partitions the table and forms part of the
        merge identity.
    """

    hierarchy_id: str


class ReferenceSpaceWrite(ReferenceSpace):
    """``ReferenceSpace`` with coherent optional voxel scale and default view."""

    @model_validator(mode="after")
    def validate_voxel_scale(self) -> ReferenceSpaceWrite:
        """Check supplied physical scale while allowing unspecified voxel scale.

        Returns
        -------
        ReferenceSpaceWrite
            The validated instance, unchanged.

        Raises
        ------
        ValueError
            If voxel_size and voxel_size_unit are not supplied together, scale
            is supplied for coordinates not in VOXELS, the scale unit is not a
            supported physical length unit, or a dimension is nonfinite or
            nonpositive.
        """
        if (self.voxel_size is None) != (self.voxel_size_unit is None):
            raise ValueError("voxel_size and voxel_size_unit must be supplied together")
        if self.voxel_size is None:
            return self
        if self.unit != Unit.VOXELS:
            raise ValueError("voxel_size requires reference space unit VOXELS")
        if self.voxel_size_unit not in (
            Unit.NANOMETERS_LENGTH,
            Unit.MICRONS_LENGTH,
            Unit.MILLIMETERS_LENGTH,
            Unit.CENTIMETERS_LENGTH,
        ):
            raise ValueError("voxel_size_unit must be a supported physical length unit")
        for index, dimension in enumerate(self.voxel_size):
            if not isfinite(dimension) or dimension <= 0:
                raise ValueError(
                    f"voxel_size[{index}] must be finite and strictly positive"
                )
        return self

    @model_validator(mode="after")
    def validate_default_view_axes(self) -> ReferenceSpaceWrite:
        """Require the two screen directions to come from different data axes.

        Returns
        -------
        ReferenceSpaceWrite
            The validated instance, unchanged.

        Raises
        ------
        ValueError
            If both directions resolve to the same data axis, or if either
            signed axis has no entry in :data:`DATA_AXIS_BY_SIGNED_AXIS`.
        """
        view = self.default_2d_view
        if view is None:
            return self

        axes = []
        for direction, signed_axis in (
            ("left_to_right", view.left_to_right),
            ("bottom_to_top", view.bottom_to_top),
        ):
            axis = DATA_AXIS_BY_SIGNED_AXIS.get(signed_axis)
            if axis is None:
                raise ValueError(
                    f"default_2d_view.{direction}={signed_axis!r} has no data axis; "
                    f"add it to DATA_AXIS_BY_SIGNED_AXIS"
                )
            axes.append(axis)

        if axes[0] == axes[1]:
            raise ValueError(
                "default_2d_view must use different data axes, but "
                f"{view.left_to_right} and {view.bottom_to_top} "
                f"are both axis {axes[0]}"
            )
        return self



def validate_for_write(
    models: Sequence[BaseModel], spec: WriteSpec
) -> list[BaseModel]:
    """Enforce model types and the spec's write-time constraints before IO.

    Parameters
    ----------
    models:
        A normalized, non-empty sequence whose members must each have exact
        type ``spec.model_cls``. This boundary does not normalize a single
        model or materialize an iterable.
    spec:
        The write policy for the batch. ``spec.model_cls`` determines the
        accepted exact type, and ``spec.validation_cls`` carries the write-time
        constraints every row is re-validated against: fields that must be
        present and non-null even when the generated model makes them optional,
        plus any cross-field rules declared on that class.

    Returns
    -------
    list[BaseModel]
        A new list containing the original model instances in input order. The
        validated copies are used only for checking and are not returned.

    Raises
    ------
    TypeError
        If ``models`` is not a sequence or a member's exact type differs from
        ``spec.model_cls``.
    ValueError
        If the sequence is empty, or a member fails validation against
        ``spec.validation_cls``.

    Notes
    -----
    Every row is re-validated whether or not the spec tightens any field, so
    rows built through ``model_construct`` are schema-checked too. Validation
    performs no IO and does not mutate the supplied models.
    """
    if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
        raise TypeError(
            "validate_for_write expected a non-empty sequence of pydantic "
            f"models; got {type(models).__name__}"
        )
    if len(models) == 0:
        raise ValueError("validate_for_write received an empty sequence")

    validation_cls = spec.validation_cls
    for index, model in enumerate(models):
        if type(model) is not spec.model_cls:
            raise TypeError(
                "validate_for_write requires exact spec.model_cls members; "
                f"row {index} has type {type(model).__name__}, "
                f"expected {spec.model_cls.__name__}"
            )
        try:
            validation_cls.model_validate(model.model_dump(warnings=False))
        except ValidationError as err:
            raise ValueError(_failure_message(spec, model, index, err)) from err

    return list(models)


def _failure_message(
    spec: WriteSpec, model: BaseModel, index: int, err: ValidationError
) -> str:
    """Describe one failing row by position, id, and offending slots."""
    slots = sorted(
        {
            ".".join(str(part) for part in error.get("loc", ()))
            for error in err.errors()
            if error.get("loc")
        }
    )
    row_id = getattr(model, "id", None)
    located = f"row {index}" if row_id is None else f"row {index} (id={row_id})"
    slot_text = f"; invalid slot(s): {', '.join(slots)}" if slots else ""
    return f"{spec.model_cls.__name__}: invalid write {located}{slot_text}. {err}"
