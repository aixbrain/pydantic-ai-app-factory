"""Compose each feature's extras slice into one request model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model
from pydantic.alias_generators import to_camel

from pydantic_ai_app_factory.feature import Feature


class _ExtrasBase(BaseModel):
    """Config base for the composed model, mirroring how pydantic-ai models the Vercel wire.

    A Vercel AI frontend sends camelCase, and unknown extras are dropped silently -- so without the
    alias a snake_case feature field never binds and the feature quietly sees its default instead of
    what the caller sent. `populate_by_name` keeps the field name working for JSON clients on `/run`.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


def compose_extras_model(
    features: Iterable[Feature], *, shared_extras_fields: frozenset[str] = frozenset()
) -> type[BaseModel]:
    """Merge every feature's `extras_type` slice into one model.

    Two features declaring the same field is a composition error unless the field is named in
    `shared_extras_fields`. For a shared field only the *type* is checked for agreement; whether it is
    required, and its default, are left to the features to keep consistent -- the composed model resolves
    those by MRO (the order features appear), so declaring one required in one feature and optional in
    another silently takes whichever wins the base order. A name in `shared_extras_fields` that no
    feature declares is rejected: the parameter resolves a collision and cannot contribute a field, so
    reaching for it to add one would otherwise leave that field out of the model and dropped from every
    request. Unknown fields are ignored (pydantic's default). With no `extras_type` anywhere, the model
    is empty.
    """
    # Collect every declaration before judging any, so one run reports all overlaps together
    # instead of failing on the first and hiding the rest.
    declarers: dict[str, list[tuple[str, Any]]] = {}  # field name -> [(feature, annotation), ...]
    bases: list[type[BaseModel]] = []
    for feature in features:
        extras_type = feature.extras_type
        if extras_type is None:
            continue
        for field_name, field in extras_type.model_fields.items():
            declarers.setdefault(field_name, []).append((feature.name, field.annotation))
        bases.append(extras_type)

    errors: list[str] = []
    # A name nothing declares resolves no collision: the caller took the parameter for a way to add a
    # field, and would otherwise get a model without it and a value silently dropped from every request.
    undeclared = sorted(shared_extras_fields - declarers.keys())
    if undeclared:
        names = ', '.join(repr(name) for name in undeclared)
        errors.append(
            f'shared_extras_fields names {names}, which no feature declares; it resolves a collision '
            'between features and cannot contribute a field -- declare it on the extras_type of the '
            'feature that reads it'
        )
    for field_name, decls in declarers.items():
        if len(decls) == 1:
            continue
        names = ', '.join(name for name, _ in decls)
        if field_name not in shared_extras_fields:
            errors.append(
                f"extras field '{field_name}' is declared by features {names}; "
                f'add it to shared_extras_fields to share it'
            )
        elif len({annotation for _, annotation in decls}) > 1:
            errors.append(f"shared extras field '{field_name}' has conflicting types across features {names}")
    if errors:
        raise ValueError('extras composition failed:\n' + '\n'.join(f'  - {error}' for error in errors))

    # Multiple inheritance merges the slices. The check above only rejects a *type* clash on an
    # unshared field; a shared field's requiredness and default are still resolved by MRO (base order).
    # `_ExtrasBase` goes last: a feature that sets its own model_config wins by MRO.
    return create_model('RequestExtras', __base__=(*bases, _ExtrasBase))
