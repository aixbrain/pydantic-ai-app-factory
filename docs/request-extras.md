# Request Extras

This guide shows you how to let clients pass per-request parameters to your
feature: declaring them, receiving them in a run, and sharing a field between
two features.

Prerequisites: a feature as in [Features](features.md).

## Declaring extras

A feature declares its slice of the per-request parameters as a pydantic model
on `extras_type`:

```python
from pydantic import BaseModel

from pydantic_ai_app_factory import Feature


class TranslationExtras(BaseModel):
    target_language: str = 'en'


class Translation(Feature):
    name = 'translation'
    extras_type = TranslationExtras
```

Fields may be required too: declare one without a default and a request that
omits it is rejected with 422.

## What clients send

On `/chat`, extras are extra top-level fields on the Vercel body — siblings of
`id` and `messages`, never nested:

```json
{
  "id": "c1",
  "trigger": "submit-message",
  "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "Translate the quarterly report."}]}],
  "targetLanguage": "de"
}
```

The `id` beside them is the conversation the run belongs to, not an extra --
see [the reference](api-reference.md#endpoints).

On `/run`, they live in an `extras` object, beside that same conversation id:

```json
{
  "prompt": "Translate the quarterly report.",
  "conversation_id": "c1",
  "extras": {"target_language": "de"}
}
```

Invalid extras are rejected with 422 before the run starts -- FastAPI's standard
`detail` shape on both endpoints, with `loc` naming the field as the client
spelled it, under `body.extras` on `/run` and directly under `body` on `/chat` --
and `GET /config` exposes the composed JSON schema so a frontend can discover
what the deployed feature set accepts.

Names bind in snake_case and in camelCase alike: `target_language` and
`targetLanguage` both reach the feature, so a Vercel AI frontend's camelCase
extras need no translation. The JSON schema on `GET /config` and in OpenAPI
lists the camelCase form.

That holds for the extras only, not for the `/run` body around them: `prompt`
and `conversation_id` bind by name alone. A client that camelCases everything
sends `conversationId`, which is dropped like any unknown field -- and every
turn silently starts a new conversation.

One asymmetry to know: a _value_ of the wrong type is rejected, but a field
_name_ the composed model does not know is silently ignored (pydantic's
default) and the declared default applies. Any spelling other than the two
above does not error -- it is simply not seen.

## What a run receives

The validated extras arrive as one flat mapping:
`ctx.deps.extras['target_language']`. The keys are the declared field names
-- snake_case and unprefixed, whichever spelling the client sent -- and all
features share the mapping, so name fields distinctively. Two features
declaring the same field is a build error, and composition reports every
colliding field at once, not just the first.

## Sharing a field deliberately

When two features are meant to read the same parameter, list it in
`shared_extras_fields`:

```python
app = create_agent_app(
    agent,
    auth=auth,
    features=[Translation(), Summaries()],
    shared_extras_fields=frozenset({'target_language'}),
)
```

For a shared field only the _type_ is checked for agreement; declare it
identically (same type, same default) in every feature that uses it, or the
composed default silently follows feature order.

The parameter resolves that collision and does nothing else. A name no feature
declares is rejected at build time, because it cannot contribute a field:
without the check the field would be absent from the composed model and its
value dropped from every request, and unknown extras are ignored rather than
reported. Declare the field on the `extras_type` of the feature that reads it.
