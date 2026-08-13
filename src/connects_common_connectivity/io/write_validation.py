"""Write-time, pydantic-only validation hooked into :func:`write_models`.

The IO layer should never blindly trust that a model carries every slot
the write actually depends on. Many generated fields are ``Optional`` in
``models.py`` because the schema permits them to be missing in some
contexts, but the *write* path needs them concretely (e.g. the predicate
columns, the partition columns, the id used for dedupe).

The :class:`WriteSpec` for each writable class records this in
``required_for_write``. This module turns that list into a real check by
deriving a strict pydantic subclass of the generated model and re-validating each
instance through it before any IO during runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError, create_model

from connects_common_connectivity.io.write_spec import WriteSpec

__all__ = ["strict_model_for", "validate_for_write"]


def _strip_optional(annotation: Any) -> Any:
    """Return ``annotation`` with ``None`` removed from any top-level Union.

    A field annotated ``Optional[str]`` (``str | None``) accepts ``None`` as
    a valid value even when ``Field(...)`` makes it required. For write-time
    enforcement we want ``None`` to be a validation error, so we strip the
    ``NoneType`` arm of any top-level union.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = tuple(a for a in get_args(annotation) if a is not type(None))
        if len(args) == 0:
            return annotation
        if len(args) == 1:
            return args[0]
        return Union[args]  # type: ignore[return-value]
    return annotation


def strict_model_for(spec: WriteSpec) -> type[BaseModel]:
    """Return ``spec.model_cls`` with the supplied write-required slots forced.

    The supplied spec defines the validation policy. The global registry does
    not participate in this function. Derived classes use a bounded cache. Its
    key contains the model class and a sorted tuple of required field names.
    Equivalent specs reuse a class. Different policies remain isolated.

    The generated parent model is never mutated. If no fields need tightening,
    this function returns the parent class.
    """
    required = tuple(sorted(spec.required_for_write))
    return _build_strict_model_cached(spec.model_cls, required)


@lru_cache(maxsize=128)
def _build_strict_model_cached(
    model_cls: type[BaseModel], required: tuple[str, ...]
) -> type[BaseModel]:
    """Build and cache the strict model for one complete validation policy."""
    if not required:
        return model_cls

    overrides: dict[str, Any] = {}
    for name in required:
        finfo = model_cls.model_fields.get(name)
        if finfo is None:
            raise ValueError(
                f"{model_cls.__name__}: required_for_write field {name!r} "
                f"is not declared on the model"
            )
        overrides[name] = (_strip_optional(finfo.annotation), Field(...))

    strict = create_model(
        f"{model_cls.__name__}_StrictWrite",
        __base__=model_cls,
        **overrides,
    )
    return strict


def validate_for_write(
    models: Sequence[BaseModel], spec: WriteSpec
) -> list[BaseModel]:
    """Validate one normalized non-empty model sequence against ``spec``.

    This function does not normalize inputs. It rejects single models,
    generators, strings, bytes, and other non-sequence values. Every member
    must have exact type ``spec.model_cls``. Strict subclasses are used only
    for validation. On success, the function returns the original model
    instances in a new list.
    """
    if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
        raise TypeError(
            "validate_for_write expected a non-empty sequence of pydantic "
            f"models; got {type(models).__name__}"
        )
    if not models:
        raise ValueError("validate_for_write received an empty sequence")

    for index, model in enumerate(models):
        if type(model) is not spec.model_cls:
            raise TypeError(
                "validate_for_write requires exact spec.model_cls members; "
                f"row {index} has type {type(model).__name__}, "
                f"expected {spec.model_cls.__name__}"
            )

    strict = strict_model_for(spec)
    if strict is spec.model_cls:
        return list(models)

    validated: list[BaseModel] = []
    for index, model in enumerate(models):
        try:
            strict.model_validate(model.model_dump())
        except ValidationError as err:
            invalid = sorted(
                {
                    ".".join(str(p) for p in e.get("loc", ()))
                    for e in err.errors()
                }
            )
            slot_text = ", ".join(invalid) if invalid else "(see below)"
            row_id = getattr(model, "id", None)
            row_hint = (
                f"row {index}"
                if row_id is None
                else f"row {index} (id={row_id})"
            )
            raise ValueError(
                f"{spec.model_cls.__name__}: invalid required_for_write slot(s): "
                f"{slot_text} at {row_hint}. {err}"
            ) from err
        validated.append(model)

    return validated
