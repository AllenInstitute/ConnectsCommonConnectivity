"""Write-time, pydantic-only validation hooked into :func:`write_models`.

The IO layer should never blindly trust that a model carries every slot
the write actually depends on. Many generated fields are ``Optional`` in
``models.py`` because the schema permits them to be missing in some
contexts, but the *write* path needs them concretely (e.g. the predicate
columns, the partition columns, the id used for dedupe).

The :class:`WriteSpec` for each writable class names the class those
constraints live on through ``validation_cls``. This module re-validates every
instance through it before any IO during runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from connects_common_connectivity.io.write_spec import WriteSpec

__all__ = ["validate_for_write"]


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
