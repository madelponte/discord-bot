# discord-bot

A small, self-hosted Discord bot that bridges a Discord server to any
**OpenAI-compatible chat-completions API** (e.g. [llama.cpp]'s `llama-server`,
Ollama, vLLM, LM Studio, or OpenAI itself). Mention the bot in a channel, reply
to one of its messages, or DM it, and it forwards your message to the configured
LLM and replies with the model's output.

It is intentionally minimal — one [`bot.py`](bot.py), two dependencies, and a
container image — so it's easy to read, audit, and run anywhere Docker runs.

[llama.cpp]: https://github.com/ggml-org/llama.cpp

## How it works

1. The bot logs in to Discord using the [discord.py] gateway client.
2. It listens for messages with a lean `discord.Client` (`on_message`) and acts
   when the bot is **@-mentioned**, when someone **replies to one of its
   messages** (with or without the reply ping), or on any **direct message**.
   Messages from itself, other bots, and webhooks, plus Discord system messages
   (pins, joins, …), are ignored to prevent feedback loops. A bare @mention with
   no text gets a short usage hint; a text-less reply or DM is ignored.
3. If an allow-list of server (guild) IDs is configured, messages from any other
   server are ignored, and direct messages are answered only for members of an
   allowed server. Membership is checked with a Discord API lookup (no extra
   intent needed) and cached for 5 minutes.
4. The bot's own mention is removed from the text to form the **prompt**; any
   other mentions (users, roles, channels) are resolved to readable display
   names so the model sees "Alice" rather than a raw ID. For context, the bot
   also reads back over the channel's most recent messages (up to
   `MAX_CONTEXT_MESSAGES`) — each prior message becomes a turn (the bot's own as
   the assistant, others prefixed with the speaker's name) so it can see what's
   recently been going on, with the triggering message (also prefixed with its
   author's name) as the final prompt. Consecutive messages from the same role
   are merged and the conversation always starts with a user turn, so models
   whose chat templates require strictly alternating roles work. The bot's own
   error and busy notices are left out of the context.
5. The conversation is sent as a `chat/completions` request to `API_BASE_URL`
   with a system prompt, the configured model name, `max_tokens`, and
   `temperature`. If `API_KEY` is configured, it is sent as an
   `Authorization: Bearer …` header. A per-user cooldown
   (`USER_COOLDOWN_SECONDS`) stops a single user from hammering the bot.
6. While the model generates, the channel shows a typing indicator. The reply is
   posted back. Inline reasoning from "thinking" models (`<think>…</think>`) is
   removed first, so only the final answer counts toward `MAX_RESPONSE_CHARS`
   (default 2000, one Discord message); longer answers are cut off at a word
   boundary with "…". If `MAX_RESPONSE_CHARS` is raised above Discord's
   2000-character limit, the answer is split across multiple messages. If an
   optional `MAX_TOKENS` cap is set and the model spends it all thinking, the
   bot reports an empty response and logs a hint.
7. When the answer names someone taking part in the conversation (an author
   of, or user mentioned in, the messages the model was shown), the name is
   turned into a real Discord mention and that person is pinged. Models write
   names as plain text, which Discord never pings, so the bot does this itself.
   Names are matched by display name, case-sensitively and as whole words
   (`Alice` or `@Alice`); names under 3 characters, names shared by two
   participants, and text inside code are left alone. Nobody else can be
   pinged: `@everyone`/`@here`, roles, other bots, users outside the
   conversation, and the reply-author ping are always suppressed.

By default, only **one request at a time** is processed across all channels and
servers in this bot process. `MAX_CONCURRENT_REQUESTS` can raise that limit, but
each user is always limited to one active request, even with cooldown disabled.
Capacity is held from history loading through delivery of the final reply and
released on completion, failure, or cancellation. Limits survive Discord client
reconnections. Additional requests receive a brief busy reply; they are not
queued and do not consume the user's cooldown. Limits are per process, not
shared between separately deployed bot instances.

HTTP calls use a single shared [aiohttp] session (created lazily, reused across
requests) with a 120-second total timeout. Connection errors, timeouts, and API
errors are caught and reported back to the channel as a short `⚠️` message
instead of crashing. If the triggering message is deleted while the model is
generating, the response is still posted as a normal channel message.
Internal URLs, exception details, and API error bodies are logged server-side
but are never included in Discord replies.

Transient Discord gateway disconnects are resumed by `discord.py`. If one
session disconnects at least five times within 60 seconds, the bot abandons that
session and automatically starts a fresh Discord client instead of remaining in
a rapid resume loop.

[discord.py]: https://github.com/Rapptz/discord.py
[aiohttp]: https://github.com/aio-libs/aiohttp

## Configuration

All configuration is via environment variables. Copy
[`.env.example`](.env.example) to `.env` and fill it in:

| Variable            | Required | Default                                | Description |
| ------------------- | :------: | -------------------------------------- | ----------- |
| `DISCORD_TOKEN`     | ✅       | —                                      | Bot token from the [Discord Developer Portal]. The bot exits if this is unset. |
| `API_BASE_URL`      |          | `http://llama-server:8080/v1`          | Base URL of the OpenAI-compatible API. `/chat/completions` is appended to it. |
| `API_KEY`           |          | *(empty)*                              | API key sent as an `Authorization: Bearer <key>` header. Leave empty for unauthenticated servers. |
| `MODEL_NAME`        |          | `default`                              | Model name sent in the request body. |
| `ALLOWED_GUILD_IDS` |          | *(empty = all servers)*                | Comma-separated Discord server IDs the bot is allowed to respond in. When set, DMs are accepted only from members of these servers. |
| `SYSTEM_PROMPT`     |          | `You are a helpful assistant. …`       | System prompt prepended to every request. |
| `MAX_RESPONSE_CHARS` |         | `2000`                                 | Maximum length of the posted answer, in characters, after reasoning is removed. Longer answers are truncated with "…". Values above 2000 are split across multiple messages. Clamped to at least `1`. |
| `MAX_TOKENS`        |          | *(empty = server default)*             | Optional `max_tokens` cap sent to the API. Counts reasoning tokens too, so keep it generous for thinking models. `0` or empty omits it. |
| `MAX_CONTEXT_MESSAGES` |       | `6`                                    | How many of the channel's most recent messages to include as context (counting the triggering message). `1` = one-shot, no context. |
| `USER_COOLDOWN_SECONDS` |      | `5`                                    | Minimum seconds between requests from the same user. `0` disables the cooldown. |
| `MAX_CONCURRENT_REQUESTS` |    | `1`                                    | Maximum active requests across the bot process. Clamped to at least `1`; each user can have only one active request. Busy requests are rejected, not queued. |

> **Note:** if `ALLOWED_GUILD_IDS` is left empty the bot will respond in **every**
> server it has been added to, and to direct messages from **anyone** who can
> reach it (e.g. members of any server it shares). Set it to lock the bot to
> specific servers; DMs are then limited to members of those servers.
>
> For a llama.cpp server started with `--api-key`, set `API_KEY` to the same
> value. The key is never written to the bot's logs.

### Discord setup

1. Create an application and bot at the [Discord Developer Portal].
2. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent**
   (the bot needs to read message text to build the prompt).
3. Invite the bot to your server with the *Send Messages* and *Read Message
   History* permissions.
4. In a channel, mention it: `@YourBot what's the capital of France?` To follow
   up, reply to its answer — no mention needed. You can also DM the bot
   directly (with `ALLOWED_GUILD_IDS` set, only members of those servers can).

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
- [`aiohttp`](requirements.txt) `3.14.4`

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
runs the suite on Python 3.10, 3.14, and 3.15.

## License

[MIT](LICENSE)
