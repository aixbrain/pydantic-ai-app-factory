# Serve Pydantic AI Agents as FastAPI Apps

`pydantic-ai-app-factory` lets you serve a [Pydantic AI 2.0](https://github.com/pydantic/pydantic-ai) agent as a [FastAPI](https://github.com/FastAPI/FastAPI) application. It includes support for multi-user and multi-tenant environments, capabilities, streaming, and error handling.

> [!NOTE]
> This package is in early development, the API might still change before v1.0.

## Quick Start

Install the package:

```bash
pip install pydantic-ai-app-factory
```

Create a file `main.py` with:

```python
from pydantic_ai_app_factory import UserContext, create_agent, create_agent_app


def auth() -> UserContext:
    return UserContext(user_id='demo')


agent = create_agent('test', instructions='You are a helpful assistant.')
app = create_agent_app(agent, auth=auth)
```

Launch the app like any other FastAPI app:

```bash
pip install "fastapi[standard]"
fastapi dev
```

With the server running, point any Vercel AI SDK frontend at `POST /chat` and you have a functioning chat. Or run a quick test with curl:

```bash
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"id": "test-1", "trigger": "submit-message", "messages": [{"id": "m1", "role": "user", "parts": [{"type": "text", "text": "Hello"}]}]}'
```

## Documentation

The quick start uses a test model, a mock auth function, and creates an app without features. The documentation covers how to build something real.

1. **[Building an App](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/building-an-app.md)**
   1. [Agent Endpoints](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/agent-endpoints.md)
   1. [Example app](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/pydantic_ai_app_factory_examples/quickstart.py)
1. [Features](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/features.md)
   1. [Request Extras](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/request-extras.md)
   1. [Example feature](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/pydantic_ai_app_factory_examples/notes_feature.py)

1. [Authentication and Authorization](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/auth.md)
1. [Error Contract](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/error-contract.md)
1. [Testing](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/testing.md)
1. [API Reference](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/docs/api-reference.md)

## Motivation

FastAPI gives the Python ecosystem a standardized API server on top of [Pydantic](https://github.com/pydantic/pydantic), and Pydantic AI provides the same for AI agents. `pydantic-ai-app-factory` is the missing "glue work" to compose Pydantic AI agents and their capabilities into FastAPI apps.

Serving AI agents via HTTP requires many recurring patterns, including but not limited to logic for:

- configuring multi-user authentication, request validation, and other middleware
- storing and retrieving interactions (agent runs) and conversations
- context building
- streaming intermediate results and run output to the client
- error handling

It's wasteful to create this logic from scratch in every project. That's why `pydantic-ai-app-factory` assembles any Pydantic AI Agent into a production-ready FastAPI app using framework standards, for more maintainable and secure projects. You keep full control of your agent and application, but save yourself the boilerplate code.

## Related Work

Exposing an agent through a server is an established concept, and a web search for "serve Pydantic AI with FastAPI" returns many guides, tutorials, and example projects. Pydantic AI's own documentation contains an exemplary [Chat App with FastAPI](https://pydantic.dev/docs/ai/examples/conversational-agents/chat-app/). Pydantic AI contributor Vstorm has released a [Full-Stack AI Agent Template](https://github.com/vstorm-co/full-stack-ai-agent-template) which comes with a FastAPI backend, among many other things. But we are not aware of any projects that provide the path from agent to app via an importable package as `pydantic-ai-app-factory` does.

This project builds on the contributions of Pydantic, FastAPI, and Pydantic AI. It relies heavily on the concept of [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) as _composable units of agent behavior_ introduced in Pydantic AI 2.0. It also uses the [Vercel AI Data Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) through Pydantic AI's [`VercelAIAdapter`](https://pydantic.dev/docs/ai/integrations/ui/vercel-ai/).

## Contributing

We welcome community contributions under Apache-2.0. Please create an issue to discuss an improvement, or submit a pull request. Clone the repository, then run:

```bash
make install   # uv sync
make all       # lint, format, typecheck, and tests with 100% branch coverage
```

Please ensure that `make all` passes before committing. Coding standards, file layout, and testing patterns can be found in [AGENTS.md](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/AGENTS.md).

## License

[Apache-2.0](https://github.com/aixbrain/pydantic-ai-app-factory/blob/main/LICENSE)
