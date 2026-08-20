# AGENTS.md

## Repository overview

This repository contains a small self-hosted Discord bot that bridges Discord messages to an OpenAI-compatible chat-completions API. The runtime application is implemented in one Python module and has two direct dependencies.

When directly mentioned, the bot:

1. Checks the optional guild allow-list and per-user cooldown.
2. Removes its own mention and resolves other Discord mentions to readable names.
3. Reads recent channel history to build multi-turn context.
4. Sends the conversation to the configured `/chat/completions` endpoint.
5. Replies in Discord, splitting responses at Discord's 2,000-character limit.

It is intended to work with OpenAI-compatible backends such as llama.cpp, Ollama, vLLM, LM Studio, and hosted services.

## Important files

- `bot.py` — configuration, Discord handlers, LLM HTTP client, reconnect supervision, and entry point.
- `requirements.txt` — pinned direct dependencies: `discord.py` and `aiohttp`.
- `Dockerfile` — production image based on Python 3.14 slim.
- `compose.yml` — runs the published bot image, loads `.env`, and contains a commented llama.cpp example.
- `.env.example` — configuration template. Never commit real tokens or `.env`.
- `README.md` — user-facing setup, configuration, and operation documentation.
- `.github/workflows/docker-publish.yml` — multi-architecture GHCR build and publishing workflow.

## Runtime and dependencies

- The container uses Python 3.14.
- The source is compatible with Python 3.10+ for local development.
- Direct dependencies are pinned in `requirements.txt`.
- Use the existing `.venv` for local Python commands. Its Python version may differ from the container version.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q bot.py
```

## Configuration

`bot.py` reads all configuration from environment variables at import time.

- `DISCORD_TOKEN` — required; importing or starting the module exits if it is missing or empty.
- `API_BASE_URL` — defaults to `http://llama-server:8080/v1` when absent or blank.
- `API_KEY` — optional API key sent as a Bearer token; empty disables API authentication.
- `MODEL_NAME` — defaults to `default` when absent or blank.
- `ALLOWED_GUILD_IDS` — comma-separated integer guild IDs; empty permits all guilds. When an allow-list is set, direct messages are rejected.
- `SYSTEM_PROMPT` — defaults to `You are a helpful assistant. Keep responses concise and under 2000 characters.` when the variable is absent. An explicitly blank value remains blank.
- `MAX_TOKENS` — defaults to `1024` and is clamped to at least `1`.
- `MAX_CONTEXT_MESSAGES` — defaults to `6` and is clamped to at least `1`; `1` disables history context.
- `USER_COOLDOWN_SECONDS` — defaults to `5` and is clamped to at least `0`; `0` disables the cooldown.

Non-integer values for the integer settings, or non-integer entries in `ALLOWED_GUILD_IDS`, cause a clean startup failure.

Set a dummy token when importing `bot.py` in local checks:

```bash
DISCORD_TOKEN=test .venv/bin/python -c 'import bot'
```

## Application structure

Key functions in `bot.py`:

- `_int_env()` — validates integer environment variables.
- `get_http_session()` — lazily creates the shared `aiohttp.ClientSession` with a 120-second timeout.
- `query_llm()` — calls the configured chat-completions endpoint and validates its response.
- `clean_message_content()` — removes the bot mention and resolves user, role, and channel mentions.
- `split_discord_message()` — splits output into messages no longer than 2,000 characters.
- `record_user_request()` — enforces and cleans up per-user cooldown state.
- `build_context_messages()` — builds oldest-first OpenAI-style conversation turns from recent channel history.
- `create_bot()` — configures Discord intents and event handlers.
- `run_supervised()` — runs fresh Discord clients with exponential-backoff recovery and graceful signal handling.

The shared HTTP session and cooldown map intentionally live at module scope so they persist across Discord client reconnections. Preserve fresh-client construction in the supervised reconnect loop; reusing a closed `discord.Client` is intentionally avoided.

## Discord requirements

The Discord application must have **Message Content Intent** enabled. The bot needs channel access and permission to send messages. **Read Message History** enables multi-turn context; history failures degrade to using only the triggering message. Adding the cooldown reaction may require **Add Reactions**, but reaction failures are intentionally ignored.

The bot only responds when directly mentioned. If `ALLOWED_GUILD_IDS` is empty, it can respond in any guild where it has the necessary access. If the allow-list is nonempty, only listed guilds are accepted.

## Development guidance

- Keep the project small unless a feature clearly justifies additional structure.
- Preserve the shared `aiohttp` session; do not create a session for every request.
- Keep Discord responses within 2,000 characters by using `split_discord_message()`.
- Catch Discord permission/API failures where degraded behavior is acceptable.
- Keep errors shown to Discord users concise and log detailed diagnostics server-side.
- Update `README.md` whenever dependencies, supported Python versions, environment variables, configuration defaults, or deployment behavior change.
- Never print or commit `DISCORD_TOKEN` or real `.env` contents.

## Validation after changes

There is currently no committed automated test suite. At minimum, run:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q bot.py
git diff --check
docker build --pull -t discord-bot:test .
```

To validate Compose without risking an existing `.env`, create one only when absent and remove it afterward:

```bash
test ! -e .env
cp .env.example .env
trap 'rm -f .env' EXIT
docker compose config --quiet
```

For HTTP behavior, exercise the bot functions against a local mocked `aiohttp.web` `/v1/chat/completions` endpoint. A real end-to-end Discord test requires a valid bot token and a Discord server configured with Message Content Intent.

## Deployment

Pushes to `main` trigger a `linux/amd64` and `linux/arm64` image build and publish to:

```text
ghcr.io/madelponte/discord-bot:latest
```

Main-branch builds also publish a full commit-SHA tag. Same-repository pull requests build and publish `pr-<number>` tags; fork pull requests are skipped because their tokens cannot write packages. The workflow can also be run manually.
