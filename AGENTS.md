# AGENTS.md

## Repository overview

This repository contains a small self-hosted Discord bot that bridges Discord messages to an OpenAI-compatible chat-completions API. The runtime application is implemented in one Python module and has two direct dependencies.

When directly mentioned, replied to, or sent a direct message, the bot:

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
- `ALLOWED_GUILD_IDS` — comma-separated integer guild IDs; empty permits all guilds. When an allow-list is set, direct messages are accepted only from members of a listed guild.
- `SYSTEM_PROMPT` — defaults to `You are a helpful assistant. Keep responses concise and under 2000 characters.` when the variable is absent. An explicitly blank value remains blank.
- `MAX_RESPONSE_CHARS` — defaults to `2000` and is clamped to at least `1`; the final answer, after reasoning is stripped, is truncated to this many characters. Values above 2,000 are split across Discord messages.
- `MAX_TOKENS` — optional `max_tokens` generation cap; defaults to `0`, which omits it from requests. Negative values are clamped to `0`. It counts reasoning tokens, so it is not a response-length control.
- `MAX_CONTEXT_MESSAGES` — defaults to `6` and is clamped to at least `1`; `1` disables history context.
- `USER_COOLDOWN_SECONDS` — defaults to `5` and is clamped to at least `0`; `0` disables the cooldown.
- `MAX_CONCURRENT_REQUESTS` — defaults to `1` and is clamped to at least `1`; limits active requests per process from history loading through reply delivery. Each user is limited to one active request. Busy requests are rejected without queuing or consuming cooldown.

Non-integer values for the integer settings, or non-integer entries in `ALLOWED_GUILD_IDS`, cause a clean startup failure.

Set a dummy token when importing `bot.py` in local checks:

```bash
DISCORD_TOKEN=test .venv/bin/python -c 'import bot'
```

## Application structure

Key functions in `bot.py`:

- `_int_env()` — validates integer environment variables.
- `get_http_session()` — lazily creates the shared `aiohttp.ClientSession` with a 120-second timeout.
- `query_llm()` — calls the configured chat-completions endpoint, validates its response, and strips inline reasoning.
- `truncate_response()` — shortens the final answer to `MAX_RESPONSE_CHARS` at a word boundary with an ellipsis.
- `strip_reasoning()` — removes `<think>…</think>` blocks, a bare leading block closed by `</think>`, and unfinished `<think>` output.
- `is_allowed_guild_member()` — checks and caches whether a DM author belongs to an allowed guild.
- `mention_participants()` — rewrites participants' display names in a reply to Discord mentions and returns the IDs to allow.
- `get_trigger()` — decides whether a message is a mention, a reply to the bot, a DM, or should be ignored.
- `clean_message_content()` — removes the bot mention and resolves user, role, and channel mentions.
- `split_discord_message()` — splits output into messages no longer than 2,000 characters.
- `record_user_request()` — enforces and cleans up per-user cooldown state.
- `build_context_messages()` — builds oldest-first OpenAI-style conversation turns from recent channel history and returns the human participants (`{user_id: display_name}`) the model can see.
- `create_bot()` — configures Discord intents and event handlers.
- `run_supervised()` — runs fresh Discord clients with exponential-backoff recovery and graceful signal handling.

The shared HTTP session, cooldown map, membership cache, and active-user set intentionally live at module scope so they persist across Discord client reconnections. Admission checks and reservation must not yield to the event loop; release active-user reservations in `finally`, including on cancellation. Preserve fresh-client construction in the supervised reconnect loop; reusing a closed `discord.Client` is intentionally avoided.

## Discord requirements

The Discord application must have **Message Content Intent** enabled. The bot needs channel access and permission to send messages. **Read Message History** enables multi-turn context; history failures degrade to using only the triggering message. Adding the cooldown reaction may require **Add Reactions**, but reaction failures are intentionally ignored.

The bot responds to human users who mention it, reply to one of its messages (with or without the reply ping), or send it a direct message; messages from bots, webhooks, and Discord system messages are ignored. Only a bare @mention gets the empty-prompt hint; text-less replies and DMs are ignored. Client-wide `AllowedMentions.none()` suppresses outgoing mention notifications, including reply-author pings. The one exception: `mention_participants()` rewrites display names of human conversation participants in model replies to `<@id>` mentions, and those replies allow pings for exactly those user IDs. Never allow `@everyone`, roles, or arbitrary users. If `ALLOWED_GUILD_IDS` is empty, it can respond in any guild where it has the necessary access. If the allow-list is nonempty, only listed guilds are accepted, and DMs are accepted only from members of a listed guild the bot is in. Membership is checked with `Guild.fetch_member()` (no privileged Members intent) and cached for 5 minutes; lookup errors fail closed and are not cached.

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
