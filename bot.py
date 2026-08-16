import asyncio
import logging
import os
import signal
import sys
import time
import traceback

import aiohttp
import discord

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("discord-llm-bot")

# Silence noisy libraries — these log on every heartbeat/websocket frame
logging.getLogger("discord.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.client").setLevel(logging.WARNING)
logging.getLogger("discord.http").setLevel(logging.WARNING)


def _int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` when blank.

    A non-numeric value is a configuration mistake, so we log a clean fatal
    message and exit instead of dumping a raw ValueError traceback.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.fatal("%s must be an integer, got %r.", name, raw)
        sys.exit(1)


# --- Configuration from environment variables ---
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    log.fatal("DISCORD_TOKEN is not set! Add it to your .env file.")
    sys.exit(1)

# os.environ.get's default only applies when the key is absent. The shipped
# .env.example sets these keys to an empty string, so a user who leaves them
# blank would get "" rather than the default — hence the `.strip() or default`.
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip() or "http://llama-server:8080/v1"
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip() or "default"
ALLOWED_GUILD_IDS = os.environ.get("ALLOWED_GUILD_IDS", "")  # comma-separated
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Keep responses concise and under 2000 characters.",
)
MAX_TOKENS = max(1, _int_env("MAX_TOKENS", 1024))
# How many of the channel's most recent messages to include as context
# (counting the triggering message). 1 means one-shot, no surrounding context.
MAX_CONTEXT_MESSAGES = max(1, _int_env("MAX_CONTEXT_MESSAGES", 6))
# Minimum seconds between requests from the same user. 0 disables the cooldown.
USER_COOLDOWN_SECONDS = max(0, _int_env("USER_COOLDOWN_SECONDS", 5))
DISCORD_MESSAGE_LIMIT = 2000
API_TIMEOUT_SECONDS = 120

# Parse allowed guild IDs into a set of ints. A typo here should produce a
# readable fatal error, not a raw ValueError traceback at import time.
allowed_guilds: set[int] = set()
if ALLOWED_GUILD_IDS.strip():
    for gid in ALLOWED_GUILD_IDS.split(","):
        gid = gid.strip()
        if not gid:
            continue
        try:
            allowed_guilds.add(int(gid))
        except ValueError:
            log.fatal(
                "ALLOWED_GUILD_IDS contains a non-numeric value: %r. "
                "Expected comma-separated integer guild IDs.",
                gid,
            )
            sys.exit(1)

log.info("--- Configuration ---")
log.info("API_BASE_URL         = %s", API_BASE_URL)
log.info("MODEL_NAME           = %s", MODEL_NAME)
log.info("MAX_TOKENS           = %d", MAX_TOKENS)
log.info("MAX_CONTEXT_MESSAGES = %d", MAX_CONTEXT_MESSAGES)
log.info("USER_COOLDOWN_SECONDS= %d", USER_COOLDOWN_SECONDS)
log.info("ALLOWED_GUILDS       = %s", allowed_guilds or "(all servers)")
log.info("SYSTEM_PROMPT        = %s", SYSTEM_PROMPT[:80] + ("..." if len(SYSTEM_PROMPT) > 80 else ""))
log.info("---------------------")

# Per-user cooldown tracking. Kept at module scope so it survives the
# reconnect loop's fresh-client construction below.
_last_request: dict[int, float] = {}

# Persistent aiohttp session — reused across all requests.
# Creating a new ClientSession per request (the old code) spins up a new
# TCP connector and SSL context each time, which is wasteful.
_http_session: aiohttp.ClientSession | None = None


async def get_http_session() -> aiohttp.ClientSession:
    """Return the shared aiohttp session, creating it on first use."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS),
        )
    return _http_session


async def query_llm(messages: list[dict]) -> str:
    """Send a chat completion request to the OpenAI-compatible API.

    ``messages`` is the conversation turns (user/assistant); the system prompt
    is prepended here.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    log.info("POST %s  (%d message(s))", url, len(messages))

    try:
        session = await get_http_session()
        async with session.post(url, json=payload) as resp:
            log.info("API response status: %d", resp.status)
            if resp.status != 200:
                error_text = await resp.text()
                log.error("API error body: %s", error_text[:500])
                return "⚠️ The LLM server returned an error."
            data = await resp.json(content_type=None)
            try:
                reply = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                log.error("Unexpected API response shape: %r", data)
                return "⚠️ The LLM server returned an unexpected response."
            if not isinstance(reply, str) or not reply.strip():
                log.error("LLM response was empty or non-text: %r", reply)
                return "⚠️ The LLM server returned an empty response."
            log.info("LLM reply: %d chars", len(reply))
            return reply
    except aiohttp.ClientConnectorError as e:
        log.error("Cannot connect to LLM API at %s: %s", url, e)
        return "⚠️ Cannot reach the LLM server right now."
    except Exception as e:
        log.error("Unexpected error in query_llm: %s\n%s", e, traceback.format_exc())
        return "⚠️ An unexpected error occurred while contacting the LLM server."


def clean_message_content(message: discord.Message, bot_user: discord.ClientUser) -> str:
    """Strip the bot's own mention and resolve every other mention to a name.

    The old code deleted *all* ``<@id>`` mentions, which erased references to
    other users from the prompt (and ignored role/channel mentions entirely).
    Here we drop only the bot's mention and rewrite the rest to readable
    display names so the model sees "Alice" instead of a raw ID.
    """
    content = message.content

    # Remove the bot's own mention (both the <@id> and legacy <@!id> forms).
    content = content.replace(f"<@{bot_user.id}>", "").replace(f"<@!{bot_user.id}>", "")

    # Resolve user mentions to display names.
    for user in message.mentions:
        if user.id == bot_user.id:
            continue
        name = getattr(user, "display_name", user.name)
        content = content.replace(f"<@{user.id}>", name).replace(f"<@!{user.id}>", name)

    # Resolve role and channel mentions too.
    for role in message.role_mentions:
        content = content.replace(f"<@&{role.id}>", f"@{role.name}")
    for channel in message.channel_mentions:
        content = content.replace(f"<#{channel.id}>", f"#{channel.name}")

    return content.strip()


def split_discord_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split a Discord response without exceeding the message length limit."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit

        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def record_user_request(user_id: int, now: float) -> float | None:
    """Record a request and return remaining cooldown seconds if blocked."""
    if USER_COOLDOWN_SECONDS <= 0:
        return None

    cutoff = now - USER_COOLDOWN_SECONDS
    stale_user_ids = [uid for uid, last_seen in _last_request.items() if last_seen < cutoff]
    for uid in stale_user_ids:
        del _last_request[uid]

    last = _last_request.get(user_id)
    if last is not None:
        remaining = USER_COOLDOWN_SECONDS - (now - last)
        if remaining > 0:
            return remaining

    _last_request[user_id] = now
    return None


async def build_context_messages(
    message: discord.Message,
    bot_user: discord.ClientUser,
    max_messages: int,
) -> list[dict]:
    """Build a multi-turn ``messages`` array from the channel's recent history.

    Rather than following the reply chain, we include the ``max_messages`` most
    recent messages in the channel so the model sees what's recently been going
    on — not just the thread that was replied to. The messages immediately
    preceding the trigger become the context (each classified as an assistant
    turn for the bot's own messages, or a user turn prefixed with the author's
    display name so the model can tell participants apart), and the triggering
    message is always appended as the final, plain user turn. The result is
    oldest-first, ready to hand to the chat-completions API.
    """
    context: list[dict] = []

    # Pull the messages just before the trigger for channel context. (max=1
    # means no history — just the triggering message, i.e. one-shot.)
    if max_messages > 1:
        try:
            recent = [m async for m in message.channel.history(limit=max_messages - 1, before=message)]
        except (discord.Forbidden, discord.HTTPException) as e:
            log.warning("Could not read channel history: %s", e)
            recent = []
        for msg in reversed(recent):  # history is newest-first -> walk oldest-first
            content = clean_message_content(msg, bot_user)
            if not content:
                continue
            if msg.author.id == bot_user.id:
                context.append({"role": "assistant", "content": content})
            else:
                name = getattr(msg.author, "display_name", msg.author.name)
                context.append({"role": "user", "content": f"{name}: {content}"})

    # The triggering message is always the final user turn (the actual prompt).
    prompt = clean_message_content(message, bot_user)
    if prompt:
        context.append({"role": "user", "content": prompt})

    return context


def create_bot() -> discord.Client:
    """Construct a fresh Discord client with its event handlers registered.

    The supervised loop calls this each iteration so every reconnect gets a
    brand-new client. That sidesteps re-arming a closed Bot via ``bot.clear()``,
    whose reuse-after-close semantics aren't officially guaranteed by
    discord.py and have been fragile across versions.
    """
    # Only enable the intents we actually need. Every extra intent means
    # more gateway events the bot must receive, deserialize, and discard.
    intents = discord.Intents.none()
    intents.guilds = True          # needed to resolve guild info
    intents.message_content = True # needed to read the prompt text
    intents.messages = True        # needed to receive message events

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        log.info("✅ Logged in as %s (ID: %s)", client.user, client.user.id if client.user else "unknown")
        if allowed_guilds:
            log.info("🔒 Restricted to guild IDs: %s", allowed_guilds)
        else:
            log.warning("No ALLOWED_GUILD_IDS set — bot will respond in ALL servers!")
        log.info("🔗 API endpoint: %s", API_BASE_URL)
        log.info("🤖 Model: %s", MODEL_NAME)

    @client.event
    async def on_disconnect():
        # Fired whenever the gateway connection drops. discord.py reconnects
        # automatically on transient blips; this just makes outages visible.
        log.warning("⚠️  Disconnected from Discord gateway — attempting to reconnect…")

    @client.event
    async def on_resumed():
        log.info("🔄 Reconnected and resumed Discord session.")

    @client.event
    async def on_message(message: discord.Message):
        if client.user is None:
            return

        # Ignore messages from the bot itself
        if message.author == client.user:
            return

        # Only respond when the bot is mentioned
        if client.user not in message.mentions:
            return

        log.info(
            "Mentioned by %s in guild=%s channel=%s",
            message.author,
            message.guild.id if message.guild else "DM",
            message.channel.id,
        )

        # Guild restriction check
        if allowed_guilds and (message.guild is None or message.guild.id not in allowed_guilds):
            log.info("Ignoring — guild not in allow list")
            return

        prompt = clean_message_content(message, client.user)
        if not prompt:
            await message.reply("You mentioned me but didn't ask anything! Try: `@BotName your question here`")
            return

        # Per-user cooldown so one person can't hammer the bot.
        remaining = record_user_request(message.author.id, time.monotonic())
        if remaining is not None:
            log.info("User %s on cooldown (%.1fs left) — ignoring", message.author, remaining)
            try:
                await message.add_reaction("🕒")
            except discord.HTTPException:
                pass
            return

        # Build context from the most recent messages in this channel.
        messages = await build_context_messages(message, client.user, MAX_CONTEXT_MESSAGES)
        log.info("Prompt: %s  (context: %d message(s))", prompt[:120], len(messages))

        # Show typing indicator while generating
        async with message.channel.typing():
            response = await query_llm(messages)

        for i, chunk in enumerate(split_discord_message(response)):
            if i == 0:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)

    return client


async def _close_http_session() -> None:
    """Close the shared aiohttp session, if it's open."""
    if _http_session and not _http_session.closed:
        await _http_session.close()
        log.info("HTTP session closed.")


async def run_supervised() -> None:
    """Run the bot, restarting it automatically if the connection is lost.

    discord.py already reconnects internally on transient gateway hiccups
    (see Client.connect). But during a real network/DNS outage — e.g.
    "Temporary failure in name resolution" right after the container starts —
    its internal reconnect can raise out of ``bot.start()`` and kill the
    process. This outer loop is the safety net: on any such failure we wait
    (exponential backoff) and start over, instead of exiting.

    Each iteration builds a *fresh* client via ``create_bot()`` rather than
    reusing a closed one, so we never depend on ``bot.clear()`` reuse-after-
    close behaviour.

    Genuine misconfiguration (bad token, missing privileged intents) is
    treated as fatal — retrying those would just spin forever.

    SIGINT/SIGTERM close the running client immediately so shutdown
    completes well inside a container's stop grace period.
    """
    BASE_DELAY = 1.0      # first retry waits ~1s
    MAX_DELAY = 300.0     # …capped at 5 minutes
    STABLE_AFTER = 60.0   # a connection lasting this long resets the backoff
    delay = BASE_DELAY

    # Flip a flag on SIGINT/SIGTERM so Ctrl-C / `docker stop` shut us down
    # cleanly instead of fighting the restart loop.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    bot: discord.Client | None = None

    def _handle_stop() -> None:
        # Setting the flag alone is not enough: it is only re-checked after
        # bot.start() returns, and nothing else would make it return. Also
        # close the client — that unwinds its connect loop and brings
        # start() back, so we shut down within a fraction of a second
        # instead of waiting for SIGKILL. Re-signalling is harmless: close()
        # is idempotent and is_closed() becomes true as soon as it starts.
        stop.set()
        if bot is not None and not bot.is_closed():
            asyncio.ensure_future(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            pass  # add_signal_handler isn't available on some platforms (Windows)

    try:
        while not stop.is_set():
            started_at = loop.time()
            bot = create_bot()
            try:
                async with bot:
                    await bot.start(DISCORD_TOKEN)
            except (discord.LoginFailure, discord.PrivilegedIntentsRequired) as e:
                log.fatal("Fatal startup error (not retrying): %s", e)
                break
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Bot stopped with an exception.")
            else:
                log.info("Bot closed cleanly.")
                break

            if stop.is_set():
                break

            # If we'd been connected for a while, treat this as a fresh outage
            # and restart the backoff from the bottom.
            if loop.time() - started_at >= STABLE_AFTER:
                delay = BASE_DELAY

            log.warning("Restarting bot in %.1fs…", delay)
            try:
                # Sleep, but wake immediately if asked to shut down.
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, MAX_DELAY)
    finally:
        if bot is not None and not bot.is_closed():
            await bot.close()
        await _close_http_session()
        log.info("Shutdown complete.")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting bot...")
    try:
        asyncio.run(run_supervised())
    except KeyboardInterrupt:
        log.info("Interrupted — exiting.")
