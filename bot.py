import asyncio
import logging
import os
import re
import signal
import sys
import time
import traceback
from collections import deque

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


DISCORD_MESSAGE_LIMIT = 2000


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
API_KEY = os.environ.get("API_KEY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip() or "default"
ALLOWED_GUILD_IDS = os.environ.get("ALLOWED_GUILD_IDS", "")  # comma-separated
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Keep responses concise and under 2000 characters.",
)
# Longest final answer the bot will post, in characters, after any reasoning
# is stripped. The default fits one Discord message; larger values are split.
MAX_RESPONSE_CHARS = max(1, _int_env("MAX_RESPONSE_CHARS", DISCORD_MESSAGE_LIMIT))
# Optional generation cap sent as ``max_tokens``. It counts reasoning tokens
# too, so a low value can starve thinking models of room to answer. 0 (the
# default) omits it and leaves generation length to the server.
MAX_TOKENS = max(0, _int_env("MAX_TOKENS", 0))
# How many of the channel's most recent messages to include as context
# (counting the triggering message). 1 means one-shot, no surrounding context.
MAX_CONTEXT_MESSAGES = max(1, _int_env("MAX_CONTEXT_MESSAGES", 6))
# Minimum seconds between requests from the same user. 0 disables the cooldown.
USER_COOLDOWN_SECONDS = max(0, _int_env("USER_COOLDOWN_SECONDS", 5))
# Maximum active requests across this process; each user gets at most one.
MAX_CONCURRENT_REQUESTS = max(1, _int_env("MAX_CONCURRENT_REQUESTS", 1))
API_TIMEOUT_SECONDS = 120
# A normal gateway interruption resumes once. If one Discord session rapidly
# disconnects over and over, abandon it so the supervisor can IDENTIFY a fresh
# session (often on a different gateway host) instead of resuming forever.
GATEWAY_RECONNECT_LIMIT = 5
GATEWAY_RECONNECT_WINDOW_SECONDS = 60.0

# Fixed status replies the bot posts instead of model output. They are listed
# together so channel history can recognise them and keep them out of the
# model's context, where they would read as things the assistant "said".
MSG_API_ERROR = "⚠️ The LLM server returned an error."
MSG_API_UNEXPECTED = "⚠️ The LLM server returned an unexpected response."
MSG_API_EMPTY = "⚠️ The LLM server returned an empty response."
MSG_API_UNREACHABLE = "⚠️ Cannot reach the LLM server right now."
MSG_API_TIMEOUT = "⚠️ The LLM server took too long to respond."
MSG_API_FAILED = "⚠️ An unexpected error occurred while contacting the LLM server."
MSG_EMPTY_PROMPT = "You mentioned me but didn't ask anything! Try: `@BotName your question here`"
MSG_BUSY = "🕒 I'm busy right now. Please try again shortly."
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
BOT_STATUS_MESSAGES = frozenset({
    MSG_API_ERROR,
    MSG_API_UNEXPECTED,
    MSG_API_EMPTY,
    MSG_API_UNREACHABLE,
    MSG_API_TIMEOUT,
    MSG_API_FAILED,
    MSG_EMPTY_PROMPT,
    MSG_BUSY,
})

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
log.info("API_AUTHENTICATION   = %s", "enabled" if API_KEY else "disabled")
log.info("MODEL_NAME           = %s", MODEL_NAME)
log.info("MAX_RESPONSE_CHARS   = %d", MAX_RESPONSE_CHARS)
log.info("MAX_TOKENS           = %s", MAX_TOKENS or "(server default)")
log.info("MAX_CONTEXT_MESSAGES = %d", MAX_CONTEXT_MESSAGES)
log.info("USER_COOLDOWN_SECONDS= %d", USER_COOLDOWN_SECONDS)
log.info("MAX_CONCURRENT_REQUESTS = %d", MAX_CONCURRENT_REQUESTS)
log.info("ALLOWED_GUILDS       = %s", allowed_guilds or "(all servers)")
log.info("SYSTEM_PROMPT        = %s", SYSTEM_PROMPT[:80] + ("..." if len(SYSTEM_PROMPT) > 80 else ""))
log.info("---------------------")

# Per-user cooldown tracking. Kept at module scope so it survives the
# reconnect loop's fresh-client construction below.
_last_request: dict[int, float] = {}
# Shared across fresh Discord clients so reconnects cannot bypass the limit.
_active_users: set[int] = set()
# user_id -> (expires_at, is_member) for DM access when an allow-list is set.
_membership_cache: dict[int, tuple[float, bool]] = {}
MEMBERSHIP_CACHE_SECONDS = 300.0

# Persistent aiohttp session — reused across all requests.
# Creating a new ClientSession per request (the old code) spins up a new
# TCP connector and SSL context each time, which is wasteful.
_http_session: aiohttp.ClientSession | None = None


async def get_http_session() -> aiohttp.ClientSession:
    """Return the shared aiohttp session, creating it on first use."""
    global _http_session
    if _http_session is None or _http_session.closed:
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else None
        _http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS),
            headers=headers,
        )
    return _http_session


def strip_reasoning(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning that some models emit inline.

    Reasoning models (DeepSeek-R1, Qwen3, QwQ, …) served without a separate
    reasoning channel put their chain of thought in ``content``. Three shapes
    occur: complete ``<think>…</think>`` blocks; a bare ``</think>`` when the
    chat template already opened the block in the prompt; and an unclosed
    ``<think>`` when generation hit ``max_tokens`` mid-thought. In the last
    case nothing after it is an answer, so everything from it onward goes.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    _, closed, after = text.rpartition("</think>")
    if closed:
        text = after
    text = text.split("<think>", 1)[0]
    return text.strip()


def truncate_response(text: str, limit: int) -> str:
    """Shorten ``text`` to at most ``limit`` characters, ending with an ellipsis.

    Cuts at the last whitespace when that keeps at least half the text, so
    words aren't split mid-way.
    """
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary >= len(cut) // 2:
        cut = cut[:boundary]
    return cut.rstrip() + "…"


async def query_llm(messages: list[dict]) -> str:
    """Send a chat completion request to the OpenAI-compatible API.

    ``messages`` is the conversation turns (user/assistant); the system prompt
    is prepended here.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "temperature": 0.7,
    }
    if MAX_TOKENS:
        payload["max_tokens"] = MAX_TOKENS
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    log.info("POST %s  (%d message(s))", url, len(messages))

    try:
        session = await get_http_session()
        async with session.post(url, json=payload) as resp:
            log.info("API response status: %d", resp.status)
            if resp.status != 200:
                error_text = await resp.text()
                log.error("API error body: %s", error_text[:500])
                return MSG_API_ERROR
            data = await resp.json(content_type=None)
            try:
                reply = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                log.error("Unexpected API response shape: %r", data)
                return MSG_API_UNEXPECTED
            if not isinstance(reply, str) or not reply.strip():
                log.error("LLM response was empty or non-text: %r", reply)
                return MSG_API_EMPTY
            answer = strip_reasoning(reply)
            if not answer:
                log.error(
                    "LLM response contained only reasoning (%d chars); the model may "
                    "have run out of tokens while thinking — raise or unset MAX_TOKENS.",
                    len(reply),
                )
                return MSG_API_EMPTY
            log.info("LLM reply: %d chars (%d after removing reasoning)", len(reply), len(answer))
            if len(answer) > MAX_RESPONSE_CHARS:
                log.info("Truncating reply to MAX_RESPONSE_CHARS=%d", MAX_RESPONSE_CHARS)
                answer = truncate_response(answer, MAX_RESPONSE_CHARS)
            return answer
    except aiohttp.ClientConnectorError as e:
        log.error("Cannot connect to LLM API at %s: %s", url, e)
        return MSG_API_UNREACHABLE
    except asyncio.TimeoutError:
        # aiohttp's timeout errors subclass asyncio.TimeoutError. This is an
        # expected condition with slow local models, so no traceback.
        log.error("LLM API at %s did not respond within %ds.", url, API_TIMEOUT_SECONDS)
        return MSG_API_TIMEOUT
    except Exception as e:
        log.error("Unexpected error in query_llm: %s\n%s", e, traceback.format_exc())
        return MSG_API_FAILED


def get_trigger(message: discord.Message, bot_user: discord.ClientUser) -> str | None:
    """Return why a message should get a response, or ``None`` to ignore it.

    ``"mention"`` for an explicit @mention (including a reply with the ping on),
    ``"reply"`` for a reply to one of the bot's messages with the ping off, and
    ``"dm"`` for any direct message. With a guild allow-list, DMs are later
    limited to members of an allowed guild.
    """
    if bot_user in message.mentions:
        return "mention"
    reference = message.reference
    # ``resolved`` is the replied-to Message, a DeletedReferencedMessage (no
    # author), or None if Discord didn't include it.
    replied_to = getattr(reference.resolved, "author", None) if reference else None
    if replied_to is not None and replied_to.id == bot_user.id:
        return "reply"
    if message.guild is None:
        return "dm"
    return None


def _display_name(user: discord.abc.User) -> str:
    """Return a user's server nickname/display name, falling back to username."""
    return getattr(user, "display_name", user.name)


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
        name = _display_name(user)
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


async def is_allowed_guild_member(client: discord.Client, user_id: int) -> bool:
    """Return whether a user belongs to any allowed guild, for DM access.

    Uses one REST lookup per allowed guild (no privileged Members intent
    needed). Results are cached so a burst of DMs doesn't hit Discord's rate
    limits; lookups that failed for reasons other than "not a member" are not
    cached, and fail closed.
    """
    now = time.monotonic()
    for uid in [uid for uid, (expires, _) in _membership_cache.items() if expires <= now]:
        del _membership_cache[uid]
    cached = _membership_cache.get(user_id)
    if cached is not None:
        return cached[1]

    lookup_failed = False
    for guild_id in allowed_guilds:
        guild = client.get_guild(guild_id)
        if guild is None:
            continue  # the bot isn't in this guild
        try:
            await guild.fetch_member(user_id)
        except discord.NotFound:
            continue
        except discord.HTTPException as e:
            log.warning("Could not check membership of user %s in guild %s: %s", user_id, guild_id, e)
            lookup_failed = True
            continue
        _membership_cache[user_id] = (now + MEMBERSHIP_CACHE_SECONDS, True)
        return True

    if not lookup_failed:
        _membership_cache[user_id] = (now + MEMBERSHIP_CACHE_SECONDS, False)
    return False


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
    message, prefixed the same way, is always the final user turn. The bot's
    fixed status replies (errors, busy notices) are left out.

    Many chat templates (Mistral, Gemma, …) reject conversations whose roles
    don't strictly alternate user/assistant starting with a user turn, which a
    busy channel or a split reply would otherwise produce. So consecutive
    same-role turns are merged, and leading assistant turns are dropped. The
    result is oldest-first, ready to hand to the chat-completions API.
    """
    turns: list[tuple[str, str]] = []

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
                if content not in BOT_STATUS_MESSAGES:
                    turns.append(("assistant", content))
            else:
                turns.append(("user", f"{_display_name(msg.author)}: {content}"))

    # The triggering message is always the final user turn (the actual prompt).
    prompt = clean_message_content(message, bot_user)
    if prompt:
        turns.append(("user", f"{_display_name(message.author)}: {prompt}"))

    context: list[dict] = []
    for role, content in turns:
        if not context and role == "assistant":
            continue
        if context and context[-1]["role"] == role:
            context[-1]["content"] += f"\n{content}"
        else:
            context.append({"role": role, "content": content})
    return context


def create_bot(recycle_requested: asyncio.Event | None = None) -> discord.Client:
    """Construct a fresh Discord client with its event handlers registered.

    The supervised loop calls this each iteration so every reconnect gets a
    brand-new client. That sidesteps re-arming a closed Bot via ``bot.clear()``,
    whose reuse-after-close semantics aren't officially guaranteed by
    discord.py and have been fragile across versions.

    ``discord.py`` normally resumes transient gateway disconnects internally.
    If those disconnects become a rapid loop, ``recycle_requested`` tells the
    supervisor that this client closed intentionally and needs a fresh session.
    """
    # Only enable the intents we actually need. Every extra intent means
    # more gateway events the bot must receive, deserialize, and discard.
    intents = discord.Intents.none()
    intents.guilds = True          # needed to resolve guild info
    intents.message_content = True # needed to read the prompt text
    intents.messages = True        # needed to receive message events

    # Treat generated text as untrusted: suppress user/role/everyone mentions
    # and implicit reply-author pings on every outgoing message.
    client = discord.Client(intents=intents, allowed_mentions=discord.AllowedMentions.none())
    if recycle_requested is None:
        recycle_requested = asyncio.Event()
    disconnect_times: deque[float] = deque()

    @client.event
    async def on_ready():
        # READY means Discord created a fresh session rather than resuming the
        # previous one, so any reconnect burst associated with it is over.
        disconnect_times.clear()
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
        # automatically on transient blips. However, its internal reconnect
        # loop keeps trying to RESUME the same session, so it never returns to
        # our outer supervisor if that session starts rapidly cycling.
        now = time.monotonic()
        disconnect_times.append(now)
        cutoff = now - GATEWAY_RECONNECT_WINDOW_SECONDS
        while disconnect_times and disconnect_times[0] < cutoff:
            disconnect_times.popleft()

        log.warning(
            "⚠️  Disconnected from Discord gateway — attempting to reconnect… "
            "(%d/%d in the last %.0fs)",
            len(disconnect_times),
            GATEWAY_RECONNECT_LIMIT,
            GATEWAY_RECONNECT_WINDOW_SECONDS,
        )
        if len(disconnect_times) >= GATEWAY_RECONNECT_LIMIT and not recycle_requested.is_set():
            log.error(
                "Gateway is reconnecting too frequently; discarding the current "
                "Discord session and creating a fresh one."
            )
            recycle_requested.set()
            await client.close()

    @client.event
    async def on_resumed():
        log.info("🔄 Reconnected and resumed Discord session.")

    @client.event
    async def on_message(message: discord.Message):
        if client.user is None:
            return

        # Never trigger on ourselves, other bots, or webhooks (feedback loops),
        # or on system messages such as pins, which can reference bot messages.
        if (
            message.author == client.user
            or message.author.bot
            or message.webhook_id is not None
            or message.is_system()
        ):
            return

        trigger = get_trigger(message, client.user)
        if trigger is None:
            return

        log.info(
            "Triggered (%s) by %s in guild=%s channel=%s",
            trigger,
            message.author,
            message.guild.id if message.guild else "DM",
            message.channel.id,
        )

        # Guild restriction check. With an allow-list, DMs are accepted only
        # from members of an allowed guild.
        if allowed_guilds:
            if message.guild is None:
                if not await is_allowed_guild_member(client, message.author.id):
                    log.info("Ignoring DM — author is not a member of an allowed guild")
                    return
            elif message.guild.id not in allowed_guilds:
                log.info("Ignoring — guild not in allow list")
                return

        user_id = message.author.id
        prompt = clean_message_content(message, client.user)
        if not prompt:
            # Only a bare @mention gets the usage hint; a text-less reply or DM
            # (e.g. just an image) is ignored. The hint is subject to the
            # cooldown so repeated bare mentions can't spam it.
            if trigger == "mention" and record_user_request(user_id, time.monotonic()) is None:
                try:
                    await message.reply(MSG_EMPTY_PROMPT)
                except discord.HTTPException:
                    log.warning("Could not send empty-prompt reply in channel %s", message.channel.id)
            return

        if user_id in _active_users or len(_active_users) >= MAX_CONCURRENT_REQUESTS:
            try:
                await message.reply(MSG_BUSY)
            except discord.HTTPException:
                log.warning("Could not send busy reply in channel %s", message.channel.id)
            return

        # Busy rejections do not consume cooldown. There is no await between
        # checking capacity above and reserving it below, so admission is atomic
        # on the single asyncio event loop without a lock or a waiting queue.
        remaining = record_user_request(user_id, time.monotonic())
        if remaining is not None:
            log.info("User %s on cooldown (%.1fs left) — ignoring", message.author, remaining)
            try:
                await message.add_reaction("🕒")
            except discord.HTTPException:
                pass
            return

        _active_users.add(user_id)
        try:
            # Hold capacity through history, generation, and delivery.
            messages = await build_context_messages(message, client.user, MAX_CONTEXT_MESSAGES)
            log.info("Prompt: %s  (context: %d message(s))", prompt[:120], len(messages))

            async with message.channel.typing():
                response = await query_llm(messages)

            # message.reply() would fail if the user deleted their message
            # during generation, losing the response. This reference degrades
            # to a plain channel message instead.
            reference = message.to_reference(fail_if_not_exists=False)
            for i, chunk in enumerate(split_discord_message(response)):
                if i == 0:
                    await message.channel.send(chunk, reference=reference)
                else:
                    await message.channel.send(chunk)
        finally:
            # Includes API/Discord failures and task cancellation.
            _active_users.remove(user_id)

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
            recycle_requested = asyncio.Event()
            bot = create_bot(recycle_requested)
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
                if not recycle_requested.is_set():
                    log.info("Bot closed cleanly.")
                    break
                log.warning("Discord client recycled after a gateway reconnect burst.")

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
