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

from functools import lru_cache
from types import UnionType
from typing import Any, Iterable, Sequence, Union, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError, create_model

from connects_common_connectivity.io.write_spec import REGISTRY, WriteSpec


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


@lru_cache(maxsize=None)
def strict_model_for(model_cls: type) -> type[BaseModel]:
    """Return a pydantic subclass of ``model_cls`` with write-required slots forced.

    For each name in the registered :attr:`WriteSpec.required_for_write`
    list, the corresponding field on the returned subclass is required
    (no default, ``...`` ellipsis). The annotation, validators, and other
    metadata of the parent class are preserved — only the default is
    flipped.

    Cached on ``model_cls`` so the derived class is built once and reused
    across calls.

    Important: ``models.py`` is never mutated. The returned class is a
    runtime-only subclass; assertions on the parent class's
    ``model_fields`` continue to reflect the schema as generated.
    """
    spec = REGISTRY.get(model_cls.__name__)
    required: Sequence[str] = spec.required_for_write if spec else ()
    if not required:
        # Nothing to tighten — return the original class.
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


def _coerce_iterable(models: Any) -> tuple[bool, list[BaseModel]]:
    """Return ``(was_iterable, items)`` for the same shape contract as the hook."""
    if isinstance(models, BaseModel):
        return False, [models]
    if isinstance(models, (str, bytes)) or not isinstance(models, Iterable):
        raise TypeError(
            f"validate_for_write expected a model or iterable; "
            f"got {type(models).__name__}"
        )
    return True, list(models)


def validate_for_write(models: Any, spec: WriteSpec) -> Any:
    """Re-validate ``models`` through the strict submodel for ``spec.model_cls``.

    Single instance in returns a single instance out; an iterable in
    returns a list out. No I/O. Pydantic-only. On failure, raises
    :class:`ValueError` naming the class and the failing slot.
    """
    was_iter, items = _coerce_iterable(models)
    if not items:
        return items if was_iter else None

    cls = type(items[0])
    if cls is not spec.model_cls:
        raise TypeError(
            f"validate_for_write: spec.model_cls is {spec.model_cls.__name__!r} "
            f"but received {cls.__name__!r}"
        )

    strict = strict_model_for(cls)
    if strict is cls:
        return items if was_iter else items[0]

    revalidated: list[BaseModel] = []
    for idx, m in enumerate(items):
        try:
            revalidated.append(strict.model_validate(m.model_dump()))
        except ValidationError as err:
            missing = sorted(
                {
                    ".".join(str(p) for p in e.get("loc", ()))
                    for e in err.errors()
                    if e.get("type")
                    in ("missing", "none_not_allowed", "string_type", "value_error")
                }
            )
            slot_text = ", ".join(missing) if missing else "(see below)"
            row_id = getattr(m, "id", None)
            row_hint = f"row {idx}" if row_id is None else f"row {idx} (id={row_id})"
            raise ValueError(
                f"{cls.__name__}: missing required_for_write slot(s): "
                f"{slot_text} at {row_hint}. {err}"
            ) from err

    return revalidated if was_iter else revalidated[0]
