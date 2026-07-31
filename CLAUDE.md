# CLAUDE.md

## Repository overview

This repository contains a small self-hosted Discord bot that bridges Discord messages to an OpenAI-compatible chat-completions API. It is intentionally minimal: the application is implemented in one Python module and has two direct runtime dependencies.

When the bot is mentioned in Discord, it:

1. Checks the optional guild allow-list and per-user cooldown.
2. Removes its own mention and resolves other Discord mentions to readable names.
3. Reads recent channel history to build multi-turn context.
4. Sends the conversation to an OpenAI-compatible `/chat/completions` endpoint.
5. Replies in Discord, splitting responses at Discord's 2,000-character limit.

The bot works with backends such as llama.cpp, Ollama, vLLM, LM Studio, or OpenAI-compatible hosted services.

## Important files

- `bot.py` — complete application, configuration, Discord handlers, LLM HTTP client, reconnect supervision, and entry point.
- `requirements.txt` — pinned direct Python dependencies (`discord.py` and `aiohttp`).
- `Dockerfile` — Python 3.14 slim production image.
- `compose.yml` — runs the published bot image and loads `.env`; includes a commented llama.cpp example.
- `.env.example` — configuration template. Never commit real tokens or `.env`.
- `README.md` — user-facing setup, configuration, and operation documentation.
- `.github/workflows/docker-publish.yml` — multi-architecture GHCR build and publish workflow.

## Runtime and dependencies

- Container runtime: Python 3.14
- Local development: Python 3.10+
- Direct dependencies are pinned in `requirements.txt`.
- Use the existing `.venv` for local Python commands:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q bot.py
```

## Configuration

All configuration is read from environment variables at module import time.

- `DISCORD_TOKEN` — required; import/startup exits if missing.
- `API_BASE_URL` — defaults to `http://llama-server:8080/v1`.
- `MODEL_NAME` — defaults to `default`.
- `ALLOWED_GUILD_IDS` — comma-separated guild IDs; empty permits every guild.
- `SYSTEM_PROMPT` — prompt prepended to each API request.
- `MAX_TOKENS` — defaults to `1024`.
- `MAX_CONTEXT_MESSAGES` — defaults to `6`; `1` disables history context.
- `USER_COOLDOWN_SECONDS` — defaults to `5`; `0` disables the cooldown.

Set a dummy token when importing `bot.py` in local tests:

```bash
DISCORD_TOKEN=test .venv/bin/python -c 'import bot'
```

## Application structure

Key functions in `bot.py`:

- `_int_env()` — validates integer environment variables.
- `get_http_session()` — lazily creates the shared `aiohttp.ClientSession`.
- `query_llm()` — calls the configured chat-completions endpoint and validates its response.
- `clean_message_content()` — removes the bot mention and resolves user, role, and channel mentions.
- `split_discord_message()` — splits output into messages no longer than 2,000 characters.
- `record_user_request()` — enforces and cleans up per-user cooldown state.
- `build_context_messages()` — builds oldest-first OpenAI-style conversation turns from recent channel history.
- `create_bot()` — configures Discord intents and event handlers.
- `run_supervised()` — runs fresh Discord clients with exponential-backoff recovery and graceful signal handling.

The shared HTTP session and cooldown map intentionally live at module scope so they persist across Discord client reconnections.

## Discord requirements

The Discord application must have **Message Content Intent** enabled. Its server permissions should include at least:

- Send Messages
- Read Message History

The bot only responds when directly mentioned. If `ALLOWED_GUILD_IDS` is empty, it responds in every server to which it has been added.

## Development guidance

- Keep the project small unless a feature clearly justifies additional structure.
- Preserve the shared `aiohttp` session; do not create a session for every request.
- Preserve fresh-client construction in the supervised reconnect loop; reusing a closed `discord.Client` is intentionally avoided.
- Keep Discord responses within 2,000 characters by using `split_discord_message()`.
- Catch Discord permission/API failures where degraded behavior is acceptable.
- Keep errors shown to Discord users concise; log detailed diagnostics server-side.
- Update `README.md` whenever dependencies, supported Python versions, environment variables, or deployment behavior change.
- Never print or commit `DISCORD_TOKEN` or real `.env` contents.

## Validation after changes

There is currently no committed automated test suite. At minimum, run:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q bot.py
git diff --check
docker build --pull -t discord-bot:test .
```

Validate Compose using a temporary environment file:

```bash
cp .env.example .env
docker compose config --quiet
rm .env
```

For HTTP behavior, run the bot functions against a local mocked `aiohttp.web` `/v1/chat/completions` endpoint. A real end-to-end Discord test requires a valid bot token and a Discord server configured with Message Content Intent.

## Deployment

Pushes to `main` trigger a multi-architecture (`linux/amd64`, `linux/arm64`) image build and publish to:

```text
ghcr.io/madelponte/discord-bot:latest
```

The workflow also publishes commit SHA tags. Same-repository pull requests build and publish `pr-<number>` tags; fork pull requests are skipped because their tokens cannot write packages.
