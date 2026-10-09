# Agent Endpoints

This guide covers the client side: what a client sends to `/chat` and `/run`,
how a conversation continues across turns, how files travel, and what the app
strips from a request before the agent sees it. The exact shapes are in the
[API reference](api-reference.md#endpoints).

Two things here are not the factory's to decide and are only linked: the
[Vercel AI chat protocol](https://pydantic.dev/docs/ai/integrations/ui/vercel-ai/)
that `/chat` speaks, and pydantic-ai's
[message types](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/) and
sanitizing. Where the factory decides something itself, the text says so.

Prerequisites: an app as in [Building an App](building-an-app.md).

## The two endpoints

`/chat` speaks the Vercel AI chat protocol and streams; `/run` takes plain JSON
and answers once. Both run the same agent with the same features, behind the
same `auth`, and reject a malformed body or invalid extras with FastAPI's
standard 422. `/chat` is the endpoint a frontend talks to. `/run` exists for
tests and [evals](https://pydantic.dev/docs/ai/evals/), where one JSON call with one answer is easier to script than
a stream. Neither use is enforced; a service may call `/run`, and a test may
drive `/chat` when streaming is what it checks, as [Testing](testing.md) does.

```
POST /chat   {"id": "c1", "trigger": "submit-message", "messages": [...], ...extras}
POST /run    {"prompt": "...", "conversation_id": "...", "message_history": [...], "extras": {...}}
             -> {"output": "...", "conversation_id": "...", "run_id": "...", "messages": [...]}
```

`/chat`'s body is the protocol's; `/run`'s is the factory's own, and only
`prompt` is required. Extras are covered in
[Request extras](request-extras.md#what-clients-send).

## The new turn

On `/chat` the new turn is the last user message in `messages`. On `/run` it
is `prompt`: a string, or a list of pydantic-ai
[`UserContent`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.UserContent)
parts when the turn carries more than text, for example a document URL:

```json
{"prompt": ["What is on this invoice?", {"kind": "document-url", "url": "https://example.com/inv.pdf"}]}
```

The part kinds, including `binary` for inline base64 data, are pydantic-ai's.
Two rules on files are the factory's own, and both hold on `/chat` as well. A
file URL whose media type cannot be inferred from its extension needs an
explicit `media_type`; without one the request is a 422. And a file, inline
or by URL, must have a media type pydantic-ai classifies as an image, audio,
video or document; `GET /config` lists those types as `accepts`, so a frontend
can gate its file picker on them. That is what the app accepts, not what the
model supports. Any other type is refused before the run with a 415
`file-unsupported` from the [error contract](error-contract.md), so nothing
runs and nothing is stored.

Whether the deployed model supports all four classes depends on the provider;
pydantic-ai's [input guide](https://pydantic.dev/docs/ai/core-concepts/input/)
tabulates it per model class. A model class handed a class it does not
support, audio on Anthropic or video on OpenAI, raises on the first model request,
before anything reaches the provider, and that surfaces as `internal`, because
the library gives it no type of its own to map. Map `NotImplementedError` in
`error_handlers` if a deployment needs it named.

## The conversation

Every run carries a `conversation_id`, readable inside the run as
`ctx.conversation_id`, the key a feature stores history under. The factory
takes it on `/chat` from the protocol's chat `id`, on `/run` from the body
field or, when the body has none, from the id the sent history carries;
Pydantic AI mints one when neither is present. `/run` echoes the id the run
used, so a JSON client continues by passing it back. Each turn also has a
`run_id`, unique to that run; `/run` returns it as a field and `/chat` uses it
as the streamed message's id, so a client can tie feedback or a trace link to
the answer it rendered.

A client may hold the history and send it in full (`/chat` in `messages`,
`/run` in `message_history` as pydantic-ai wire-form messages, what
[`ModelMessagesTypeAdapter`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter)
dumps), or send only the new turn and let a feature reload the history from
its store. A `/run` client that holds the history gets each turn back in the
response's `messages`, the run's own messages rather than the sent history
again, and appends them; `/chat` streams the same turn as the protocol's
message. The factory forwards what arrives and leaves owning the history to a
feature, which can trim a sent history to the current turn and inject its own,
or take the sent history as the model context, on both endpoints alike.

The response's `messages` carry the rendered `instructions`, the
`model_name` and the run's token usage. If that is more than a caller
should see, keep `/run` off the public surface -- the composed app is an
ordinary FastAPI app, so middleware or a proxy in front of it does that.

## What is stripped

Client-sent messages get the same treatment on both endpoints: pydantic-ai's
[`sanitize_messages`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.sanitize_messages)
on its defaults, run by the Vercel adapter on `/chat` and by the factory on
`/run`'s `message_history` and `prompt` parts. It drops system prompts, file
URLs outside `http`/`https`, download hints on file URLs, uploaded-file
references and a tool call left dangling at the end of a history; its docstring
says why for each. The factory's part is to keep all of that closed, with no
switch exposed, and to let a turn stripped down to nothing stay an empty turn
rather than become an error.

Sanitizing does not judge content. A client that owns its history can make the
assistant or a tool appear to have said anything, so a tool with side effects
must not treat conversation context as authorization; that belongs to `auth`
and to the tool's own checks.

## Files the model produces

A model can answer with a file of its own, for example a generated image. On
`/chat` the adapter streams it as the protocol's `file` chunk. On `/run` the
factory puts it in the response's `messages` as a `binary` part with the data
inline, not in `output`, which stays the agent's output type.

## Errors

A failed run answers through the [error contract](error-contract.md): a JSON
body under its status on `/run`, a `data-error` chunk mid-stream on `/chat`.
