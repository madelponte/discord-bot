# discord-bot

A small, self-hosted Discord bot that bridges a Discord server to any
**OpenAI-compatible chat-completions API** (e.g. [llama.cpp]'s `llama-server`,
Ollama, vLLM, LM Studio, or OpenAI itself). Mention the bot in a channel and it
forwards your message to the configured LLM and replies with the model's output.

It is intentionally minimal — one [`bot.py`](bot.py), two dependencies, and a
container image — so it's easy to read, audit, and run anywhere Docker runs.

[llama.cpp]: https://github.com/ggml-org/llama.cpp

## How it works

1. The bot logs in to Discord using the [discord.py] gateway client.
2. It listens for messages with a lean `discord.Client` and only acts when the bot is **@-mentioned**
   (`on_message`). Its own messages are ignored to prevent loops.
3. If an allow-list of server (guild) IDs is configured, messages from any other
   server are ignored.
4. The bot's own mention is removed from the text to form the **prompt**; any
   other mentions (users, roles, channels) are resolved to readable display
   names so the model sees "Alice" rather than a raw ID. For context, the bot
   also reads back over the channel's most recent messages (up to
   `MAX_CONTEXT_MESSAGES`) — each prior message becomes a turn (the bot's own as
   the assistant, others prefixed with the speaker's name) so it can see what's
   recently been going on, with the triggering message as the final prompt.
5. The conversation is sent as a `chat/completions` request to `API_BASE_URL`
   with a system prompt, the configured model name, `max_tokens`, and
   `temperature`. A per-user cooldown (`USER_COOLDOWN_SECONDS`) stops a single
   user from hammering the bot.
6. While the model generates, the channel shows a typing indicator. The reply is
   posted back; responses longer than Discord's 2000-character limit are split
   into multiple messages automatically.

HTTP calls use a single shared [aiohttp] session (created lazily, reused across
requests) with a 120-second total timeout. Connection and API errors are caught
and reported back to the channel as a generic `⚠️` message instead of crashing.
Internal URLs, exception details, and API error bodies are logged server-side
but are never included in Discord replies.

[discord.py]: https://github.com/Rapptz/discord.py
[aiohttp]: https://github.com/aio-libs/aiohttp

## Configuration

All configuration is via environment variables. Copy
[`.env.example`](.env.example) to `.env` and fill it in:

| Variable            | Required | Default                                | Description |
| ------------------- | :------: | -------------------------------------- | ----------- |
| `DISCORD_TOKEN`     | ✅       | —                                      | Bot token from the [Discord Developer Portal]. The bot exits if this is unset. |
| `API_BASE_URL`      |          | `http://llama-server:8080/v1`          | Base URL of the OpenAI-compatible API. `/chat/completions` is appended to it. |
| `MODEL_NAME`        |          | `default`                              | Model name sent in the request body. |
| `ALLOWED_GUILD_IDS` |          | *(empty = all servers)*                | Comma-separated Discord server IDs the bot is allowed to respond in. |
| `SYSTEM_PROMPT`     |          | `You are a helpful assistant. …`       | System prompt prepended to every request. |
| `MAX_TOKENS`        |          | `1024`                                 | Maximum tokens to generate per reply. |
| `MAX_CONTEXT_MESSAGES` |       | `6`                                    | How many of the channel's most recent messages to include as context (counting the triggering message). `1` = one-shot, no context. |
| `USER_COOLDOWN_SECONDS` |      | `5`                                    | Minimum seconds between requests from the same user. `0` disables the cooldown. |

> **Note:** if `ALLOWED_GUILD_IDS` is left empty the bot will respond in **every**
> server it has been added to. Set it to lock the bot to specific servers.

### Discord setup

1. Create an application and bot at the [Discord Developer Portal].
2. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent**
   (the bot needs to read message text to build the prompt).
3. Invite the bot to your server with the *Send Messages* and *Read Message
   History* permissions.
4. In a channel, mention it: `@YourBot what's the capital of France?`

[Discord Developer Portal]: https://discord.com/developers/applications

## Running

The container runs as the unprivileged `nobody` user and needs no extra
privileges or writable mounts.

### With Docker Compose (recommended)

```bash
cp .env.example .env   # then edit .env
docker compose up -d --build
docker compose logs -f
```

The bundled [`compose.yml`](compose.yml) runs the bot and reads `.env`. It also
includes a commented-out `llama-server` service you can enable to have Compose
manage your LLM backend on the same network — in that case keep the default
`API_BASE_URL` of `http://llama-server:8080/v1`.

### With the prebuilt image

Images are published to the GitHub Container Registry on every push to `main`
(see below):

```bash
docker run -d --env-file .env ghcr.io/madelponte/discord-bot:latest
```

### Locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)   # or set the variables yourself
python -u bot.py
```

## Requirements

- **Python 3.14** (the container is based on `python:3.14-slim`; 3.10+ works locally)
- [`discord.py`](requirements.txt) `2.7.1`
- [`aiohttp`](requirements.txt) `3.14.3`

## Testing

The test suite uses the standard-library `unittest` framework and mocks Discord
and the LLM API, so it needs neither a real Discord token nor a running server.
Install the development dependencies and run it with enforced 100% statement
and branch coverage:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
DISCORD_TOKEN=test .venv/bin/python -m coverage run -m unittest discover -s tests -v
.venv/bin/python -m coverage report -m
```

The coverage threshold is configured in [`.coveragerc`](.coveragerc), and CI
runs the suite on Python 3.10 and 3.14.

## License

[MIT](LICENSE)
