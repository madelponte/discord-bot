import logging
import os
import re
import sys
import traceback

import aiohttp
import discord
from discord.ext import commands

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

# --- Configuration from environment variables ---
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    log.fatal("DISCORD_TOKEN is not set! Add it to your .env file.")
    sys.exit(1)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://llama-server:8080/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "default")
ALLOWED_GUILD_IDS = os.environ.get("ALLOWED_GUILD_IDS", "")  # comma-separated
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Keep responses concise and under 2000 characters.",
)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

# Parse allowed guild IDs into a set of ints
allowed_guilds: set[int] = set()
if ALLOWED_GUILD_IDS.strip():
    for gid in ALLOWED_GUILD_IDS.split(","):
        gid = gid.strip()
        if gid:
            allowed_guilds.add(int(gid))

log.info("--- Configuration ---")
log.info("API_BASE_URL   = %s", API_BASE_URL)
log.info("MODEL_NAME     = %s", MODEL_NAME)
log.info("MAX_TOKENS     = %d", MAX_TOKENS)
log.info("ALLOWED_GUILDS = %s", allowed_guilds or "(all servers)")
log.info("SYSTEM_PROMPT  = %s", SYSTEM_PROMPT[:80] + ("..." if len(SYSTEM_PROMPT) > 80 else ""))
log.info("---------------------")

# --- Bot setup ---
# Only enable the intents we actually need. Every extra intent means
# more gateway events the bot must receive, deserialize, and discard.
intents = discord.Intents.none()
intents.guilds = True          # needed to resolve guild info
intents.message_content = True # needed to read the prompt text
intents.messages = True        # needed to receive message events

bot = commands.Bot(command_prefix="!", intents=intents)

# Persistent aiohttp session — reused across all requests.
# Creating a new ClientSession per request (the old code) spins up a new
# TCP connector and SSL context each time, which is wasteful.
_http_session: aiohttp.ClientSession | None = None


async def get_http_session() -> aiohttp.ClientSession:
    """Return the shared aiohttp session, creating it on first use."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
        )
    return _http_session


async def query_llm(prompt: str) -> str:
    """Send a chat completion request to the OpenAI-compatible API."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    log.info("POST %s  (prompt: %d chars)", url, len(prompt))

    try:
        session = await get_http_session()
        async with session.post(url, json=payload) as resp:
            log.info("API response status: %d", resp.status)
            if resp.status != 200:
                error_text = await resp.text()
                log.error("API error body: %s", error_text[:500])
                return f"⚠️ API error ({resp.status}): {error_text[:200]}"
            data = await resp.json()
            reply = data["choices"][0]["message"]["content"]
            log.info("LLM reply: %d chars", len(reply))
            return reply
    except aiohttp.ClientConnectorError as e:
        log.error("Cannot connect to LLM API at %s: %s", url, e)
        return f"⚠️ Cannot reach the LLM server at `{url}`. Is it running?"
    except Exception as e:
        log.error("Unexpected error in query_llm: %s\n%s", e, traceback.format_exc())
        return f"⚠️ Internal error: {type(e).__name__}: {e}"


@bot.event
async def on_ready():
    log.info("✅ Logged in as %s (ID: %s)", bot.user, bot.user.id)
    if allowed_guilds:
        log.info("🔒 Restricted to guild IDs: %s", allowed_guilds)
    else:
        log.warning("No ALLOWED_GUILD_IDS set — bot will respond in ALL servers!")
    log.info("🔗 API endpoint: %s", API_BASE_URL)
    log.info("🤖 Model: %s", MODEL_NAME)


@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only respond when the bot is mentioned
    if bot.user not in message.mentions:
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

    # Strip the mention from the message to get the prompt
    prompt = re.sub(r"<@!?\d+>", "", message.content).strip()

    if not prompt:
        await message.reply("You mentioned me but didn't ask anything! Try: `@BotName your question here`")
        return

    log.info("Prompt: %s", prompt[:120])

    # Show typing indicator while generating
    async with message.channel.typing():
        response = await query_llm(prompt)

    # Discord has a 2000-char limit; split if needed
    if len(response) <= 2000:
        await message.reply(response)
    else:
        chunks = [response[i : i + 1990] for i in range(0, len(response), 1990)]
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)


@bot.event
async def on_close():
    """Clean up the shared HTTP session when the bot shuts down."""
    if _http_session and not _http_session.closed:
        await _http_session.close()
        log.info("HTTP session closed.")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting bot...")
    bot.run(DISCORD_TOKEN, log_handler=None)  # log_handler=None since we set up our own
