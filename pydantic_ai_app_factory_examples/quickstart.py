"""The quick-start app from the README: served under `/api`, traced with Logfire, paired with a frontend.

The model is pydantic-ai's test model: every reply is "success (no tool calls)", no provider is
called. The comment at `agent = ...` shows how to use a real one.

Run with:

    uv run -m pydantic_ai_app_factory_examples.quickstart

Docs at http://127.0.0.1:8000/docs. The frontend setup is at the bottom of this file.
"""

import logfire
from fastapi import FastAPI

from pydantic_ai_app_factory import UserContext, create_agent, create_agent_app


def auth() -> UserContext:
    """Stands in for real authentication. Every request is the same demo user."""
    return UserContext(user_id='demo')


# The test model needs no API key and no provider SDK. To use a real model, install the provider
# extra (`pydantic-ai-slim` ships no provider SDK), export its key, and swap in the commented line. Any
# pydantic-ai model string works.
#
#     uv add "pydantic-ai-slim[openai]"
#     export OPENAI_API_KEY=sk-...
#
# agent = create_agent('openai:gpt-5-nano', instructions='You are a helpful assistant.')
agent = create_agent('test', instructions='You are a helpful assistant.')
app = create_agent_app(agent, auth=auth)

#######################################################################
# The rest of this file is demo scaffolding: tracing, the `/api` prefix
# the frontend expects, and how to run it.
#######################################################################

# Logfire needs no account here. Without a token, the trace of each request prints to the console
# (the request, the agent run, the model call); with `LOGFIRE_TOKEN` set, it is sent to Logfire.
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# Vercel's Python chat template (below) proxies `/api/*` to port 8000 in development, so the routes
# live under `/api` and UI and API share one origin. `include_router` does not carry the app's
# exception handlers; the last line copies them.
root = FastAPI()
root.include_router(app.router, prefix='/api')
root.exception_handlers.update(app.exception_handlers)
logfire.instrument_fastapi(root)

# A frontend for this app, from Vercel's Python streaming template.
#
# The API, in one terminal:
#
#     uv run -m pydantic_ai_app_factory_examples.quickstart
#
# The template, in another. It expects a FastAPI backend on port 8000 and needs no changes. Use
# `next-dev`, not `dev`: `dev` would also start the template's own FastAPI on port 8000.
#
#     npx giget@latest gh:vercel-labs/ai-sdk-preview-python-streaming my-ui
#     cd my-ui
#     npx pnpm install --ignore-scripts
#     npx pnpm next-dev
#
# The reply `success (no tool calls)` comes from the test model and confirms that the request
# reached this server. No provider key is needed on either side.

if __name__ == '__main__':
    import uvicorn

    print(
        'Test model: every reply is "success (no tool calls)". For a real model, see `agent = ...` in quickstart.py.\n'
        '  Docs:      http://127.0.0.1:8000/docs\n'
        '  Frontend:  bottom of quickstart.py\n',
        flush=True,  # before uvicorn's log lines, even when stdout is a pipe
    )
    uvicorn.run(root, host='127.0.0.1', port=8000)
