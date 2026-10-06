import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_app_factory import Feature
from pydantic_ai_app_factory._extras import compose_extras_model
from tests._support import Storage


class _LangExtras(BaseModel):
    language: str = 'en'


class _Lang(Feature):
    name = 'lang'
    extras_type = _LangExtras


class _Bare(Feature):
    name = 'bare'


def test_features_without_extras_compose_to_an_empty_model() -> None:
    model = compose_extras_model([_Bare()])
    assert model.model_fields == {}


def test_disjoint_slices_merge_into_one_model() -> None:
    model = compose_extras_model([Storage(), _Lang()])
    validated = model.model_validate({'case_id': 'c1', 'language': 'de'})
    assert validated.model_dump() == {'case_id': 'c1', 'language': 'de'}


def test_defaults_survive_composition() -> None:
    model = compose_extras_model([_Lang()])
    assert model.model_validate({}).model_dump() == {'language': 'en'}


def test_unknown_extras_are_ignored() -> None:
    model = compose_extras_model([Storage()])
    assert model.model_validate({'case_id': 'c1', 'unclaimed': 'x'}).model_dump() == {'case_id': 'c1'}


def test_a_missing_required_field_fails_validation() -> None:
    model = compose_extras_model([Storage()])
    with pytest.raises(ValidationError):
        model.model_validate({})


class _AlsoCaseStr(BaseModel):
    case_id: str


class _CaseInt(BaseModel):
    case_id: int


class _OtherStr(Feature):
    name = 'other'
    extras_type = _AlsoCaseStr


class _OtherInt(Feature):
    name = 'other'
    extras_type = _CaseInt


def test_an_overlapping_field_fails_loudly_by_default() -> None:
    with pytest.raises(ValueError, match='shared_extras_fields'):
        compose_extras_model([Storage(), _OtherStr()])


def test_shared_extras_fields_allows_a_deliberate_overlap() -> None:
    model = compose_extras_model([Storage(), _OtherStr()], shared_extras_fields=frozenset({'case_id'}))
    assert set(model.model_fields) == {'case_id'}


def test_a_shared_field_with_conflicting_types_still_fails_loudly() -> None:
    with pytest.raises(ValueError, match='conflicting'):
        compose_extras_model([Storage(), _OtherInt()], shared_extras_fields=frozenset({'case_id'}))


def test_all_overlaps_are_reported_together() -> None:
    class _CaseAndLang(BaseModel):
        case_id: str
        language: str

    class _Wide(Feature):
        name = 'wide'
        extras_type = _CaseAndLang

    with pytest.raises(ValueError) as exc_info:
        compose_extras_model([Storage(), _Lang(), _Wide()])

    message = str(exc_info.value)
    assert 'case_id' in message and 'language' in message


def test_a_snake_case_extras_field_binds_from_a_camel_case_key() -> None:
    # A Vercel AI frontend sends camelCase, and unknown extras are dropped silently -- so without an
    # alias the field never binds and the feature sees a default instead of what the caller sent.
    model = compose_extras_model([Storage()])
    assert model.model_validate({'caseId': 'c-1'}).case_id == 'c-1'  # pyright: ignore[reportAttributeAccessIssue]


def test_the_field_name_itself_still_binds() -> None:
    model = compose_extras_model([Storage()])
    assert model.model_validate({'case_id': 'c-1'}).case_id == 'c-1'  # pyright: ignore[reportAttributeAccessIssue]


def test_a_shared_field_no_feature_declares_fails_loudly() -> None:
    with pytest.raises(ValueError, match='no feature declares'):
        compose_extras_model([Storage()], shared_extras_fields=frozenset({'language'}))


def test_an_undeclared_shared_field_is_reported_alongside_an_overlap() -> None:
    with pytest.raises(ValueError) as exc_info:
        compose_extras_model([Storage(), _Lang(), _OtherStr()], shared_extras_fields=frozenset({'colour'}))

    message = str(exc_info.value)
    assert 'colour' in message and 'case_id' in message
