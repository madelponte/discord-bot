import asyncio
import os
import runpy
import signal
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, AsyncMock, Mock, call, patch

os.environ.setdefault("DISCORD_TOKEN", "test")

import aiohttp
import discord

import bot


class AsyncContextManager:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class TypingContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class RawResponse:
    status = 500
    reason = "Server Error"
    headers = {}


class FakeChannel:
    def __init__(self, history_messages=None, history_error=None):
        self.id = 99
        self.history_messages = history_messages or []
        self.history_error = history_error
        self.send = AsyncMock()

    async def history(self, *, limit, before):
        if self.history_error is not None:
            raise self.history_error
        for message in self.history_messages[:limit]:
            yield message

    def typing(self):
        return TypingContext()


class FakeMessage:
    def __init__(
        self,
        content,
        author,
        *,
        mentions=None,
        role_mentions=None,
        channel_mentions=None,
        channel=None,
        guild=None,
        webhook_id=None,
        reference=None,
        system=False,
    ):
        self.content = content
        # SimpleNamespace authors default to human users in these fixtures.
        author.__dict__.setdefault("bot", False)
        author.__dict__.setdefault("name", f"user{author.id}")
        self.author = author
        self.webhook_id = webhook_id
        self.mentions = mentions or []
        self.role_mentions = role_mentions or []
        self.channel_mentions = channel_mentions or []
        self.channel = channel or FakeChannel()
        self.guild = guild
        self.reply = AsyncMock()
        self.add_reaction = AsyncMock()
        # ``reference`` is the message this one replies to, as on
        # discord.Message; ``outgoing_reference`` is what replying to *this*
        # message via to_reference() produces.
        self.reference = reference
        self.outgoing_reference = object()
        self.to_reference = Mock(return_value=self.outgoing_reference)
        self.is_system = Mock(return_value=system)


class FakeSupervisedClient:
    def __init__(self, start_effect=None, *, close_on_exit=True):
        self.start_effect = start_effect
        self.close_on_exit = close_on_exit
        self.closed = False
        self.started = False
        self.close_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self.close_on_exit:
            await self.close()
        return False

    async def start(self, token):
        self.started = True
        if callable(self.start_effect):
            result = self.start_effect()
            if asyncio.iscoroutine(result):
                await result
        elif self.start_effect is not None:
            raise self.start_effect

    async def close(self):
        self.close_calls += 1
        self.closed = True

    def is_closed(self):
        return self.closed


def stage_target(stage, build, query, message):
    """Return the mock and 1-based call number that a request stage maps to.

    The first response chunk is sent as a reply and later chunks as plain
    messages, both through ``channel.send``.
    """
    return {
        "history": (build, 1),
        "query": (query, 1),
        "reply": (message.channel.send, 1),
        "send": (message.channel.send, 2),
    }[stage]


def on_call(call_number, effect):
    """Build a side effect that runs ``effect`` only on the given call."""
    calls = 0

    async def side_effect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == call_number:
            return await effect(*args, **kwargs)
        return DEFAULT  # fall back to the mock's return_value

    return side_effect


class EnvironmentTests(unittest.TestCase):
    def test_int_env_default_integer_and_invalid(self):
        with patch.dict(os.environ, {"SETTING": ""}, clear=False):
            self.assertEqual(bot._int_env("SETTING", 7), 7)
        with patch.dict(os.environ, {"SETTING": " 12 "}, clear=False):
            self.assertEqual(bot._int_env("SETTING", 7), 12)
        with patch.dict(os.environ, {"SETTING": "bad"}, clear=False), self.assertRaises(SystemExit):
            bot._int_env("SETTING", 7)

    def test_import_exits_without_token(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            runpy.run_path(bot.__file__, run_name="not_main")

    def test_import_exits_for_invalid_integer(self):
        env = {"DISCORD_TOKEN": "test", "MAX_TOKENS": "invalid"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(SystemExit):
            runpy.run_path(bot.__file__, run_name="not_main")

    def test_import_exits_for_invalid_guild_and_parses_valid_guilds(self):
        invalid = {"DISCORD_TOKEN": "test", "ALLOWED_GUILD_IDS": "1,bad"}
        with patch.dict(os.environ, invalid, clear=True), self.assertRaises(SystemExit):
            runpy.run_path(bot.__file__, run_name="not_main")

        valid = {
            "DISCORD_TOKEN": "test",
            "ALLOWED_GUILD_IDS": "1, ,2",
            "SYSTEM_PROMPT": "x" * 81,
            "MAX_TOKENS": "-5",
            "MAX_RESPONSE_CHARS": "0",
            "MAX_CONTEXT_MESSAGES": "0",
            "USER_COOLDOWN_SECONDS": "-1",
        }
        with patch.dict(os.environ, valid, clear=True):
            namespace = runpy.run_path(bot.__file__, run_name="not_main")
        self.assertEqual(namespace["allowed_guilds"], {1, 2})
        self.assertEqual(namespace["MAX_TOKENS"], 0)
        self.assertEqual(namespace["MAX_RESPONSE_CHARS"], 1)
        self.assertEqual(namespace["MAX_CONTEXT_MESSAGES"], 1)
        self.assertEqual(namespace["USER_COOLDOWN_SECONDS"], 0)

    def test_response_length_defaults(self):
        with patch.dict(os.environ, {"DISCORD_TOKEN": "test"}, clear=True):
            namespace = runpy.run_path(bot.__file__, run_name="not_main")
        self.assertEqual(namespace["MAX_RESPONSE_CHARS"], 2000)
        self.assertEqual(namespace["MAX_TOKENS"], 0)

    def test_concurrency_configuration(self):
        for raw, expected in ((None, 1), ("", 1), ("0", 1), ("-2", 1), ("3", 3)):
            env = {"DISCORD_TOKEN": "test"}
            if raw is not None:
                env["MAX_CONCURRENT_REQUESTS"] = raw
            with self.subTest(raw=raw), patch.dict(os.environ, env, clear=True):
                namespace = runpy.run_path(bot.__file__, run_name="not_main")
                self.assertEqual(namespace["MAX_CONCURRENT_REQUESTS"], expected)
        with patch.dict(os.environ, {
            "DISCORD_TOKEN": "test", "MAX_CONCURRENT_REQUESTS": "bad"
        }, clear=True), self.assertRaises(SystemExit):
            runpy.run_path(bot.__file__, run_name="not_main")

    def test_entry_point_runs_and_handles_keyboard_interrupt(self):
        env = {"DISCORD_TOKEN": "test"}
        with patch.dict(os.environ, env, clear=True), patch("asyncio.run") as run:
            runpy.run_path(bot.__file__, run_name="__main__")
            run.assert_called_once()
            run.call_args.args[0].close()

        with patch.dict(os.environ, env, clear=True), patch(
            "asyncio.run", side_effect=KeyboardInterrupt
        ) as run:
            runpy.run_path(bot.__file__, run_name="__main__")
            run.call_args.args[0].close()


class HttpSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bot._http_session = None

    async def asyncTearDown(self):
        await bot._close_http_session()
        bot._http_session = None

    async def test_get_http_session_creates_reuses_and_replaces_closed_session(self):
        first = SimpleNamespace(closed=False, close=AsyncMock())
        second = SimpleNamespace(closed=False, close=AsyncMock())
        with (
            patch.object(bot, "API_KEY", "secret-key"),
            patch.object(bot.aiohttp, "ClientSession", side_effect=[first, second]) as constructor,
        ):
            self.assertIs(await bot.get_http_session(), first)
            self.assertIs(await bot.get_http_session(), first)
            first.closed = True
            self.assertIs(await bot.get_http_session(), second)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(
            constructor.call_args.kwargs["timeout"].total,
            bot.API_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            constructor.call_args.kwargs["headers"],
            {"Authorization": "Bearer secret-key"},
        )

    async def test_get_http_session_omits_auth_header_without_api_key(self):
        session = SimpleNamespace(closed=False, close=AsyncMock())
        with (
            patch.object(bot, "API_KEY", ""),
            patch.object(bot.aiohttp, "ClientSession", return_value=session) as constructor,
        ):
            self.assertIs(await bot.get_http_session(), session)
        self.assertIsNone(constructor.call_args.kwargs["headers"])

    async def test_close_http_session_handles_none_closed_and_open(self):
        await bot._close_http_session()

        closed = SimpleNamespace(closed=True, close=AsyncMock())
        bot._http_session = closed
        await bot._close_http_session()
        closed.close.assert_not_awaited()

        opened = SimpleNamespace(closed=False, close=AsyncMock())
        bot._http_session = opened
        await bot._close_http_session()
        opened.close.assert_awaited_once()
        opened.closed = True


class QueryLlmTests(unittest.IsolatedAsyncioTestCase):
    def response(self, *, status=200, data=None, text="error"):
        response = SimpleNamespace(
            status=status,
            text=AsyncMock(return_value=text),
            json=AsyncMock(return_value=data),
        )
        session = SimpleNamespace(post=Mock(return_value=AsyncContextManager(response)))
        return response, session

    async def test_success_builds_expected_request(self):
        response, session = self.response(
            data={"choices": [{"message": {"content": " answer "}}]}
        )
        messages = [{"role": "user", "content": "question"}]
        with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
            result = await bot.query_llm(messages)
        self.assertEqual(result, "answer")
        url, = session.post.call_args.args
        self.assertEqual(url, f"{bot.API_BASE_URL.rstrip('/')}/chat/completions")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], bot.MODEL_NAME)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["messages"][1:], messages)
        response.json.assert_awaited_once_with(content_type=None)

    async def test_max_tokens_is_sent_only_when_configured(self):
        _, session = self.response(data={"choices": [{"message": {"content": "answer"}}]})
        with (
            patch.object(bot, "get_http_session", AsyncMock(return_value=session)),
            patch.object(bot, "MAX_TOKENS", 4096),
        ):
            await bot.query_llm([])
        self.assertEqual(session.post.call_args.kwargs["json"]["max_tokens"], 4096)

    async def test_final_answer_is_limited_after_reasoning_is_removed(self):
        # Reasoning far beyond the limit doesn't count; only the answer does.
        reply = "<think>" + "x" * 50 + "</think>short answer"
        _, session = self.response(data={"choices": [{"message": {"content": reply}}]})
        with (
            patch.object(bot, "get_http_session", AsyncMock(return_value=session)),
            patch.object(bot, "MAX_RESPONSE_CHARS", 12),
        ):
            self.assertEqual(await bot.query_llm([]), "short answer")

        _, session = self.response(
            data={"choices": [{"message": {"content": "one two three four"}}]}
        )
        with (
            patch.object(bot, "get_http_session", AsyncMock(return_value=session)),
            patch.object(bot, "MAX_RESPONSE_CHARS", 12),
        ):
            self.assertEqual(await bot.query_llm([]), "one two…")

    async def test_non_200_response(self):
        _, session = self.response(status=503, text="backend unavailable")
        with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
            result = await bot.query_llm([])
        self.assertEqual(result, bot.MSG_API_ERROR)
        self.assertNotIn("503", result)
        self.assertNotIn("backend unavailable", result)

    async def test_unexpected_response_shapes(self):
        for data in ({}, {"choices": []}, {"choices": [None]}):
            with self.subTest(data=data):
                _, session = self.response(data=data)
                with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
                    result = await bot.query_llm([])
                self.assertIn("unexpected response", result)

    async def test_empty_blank_and_non_text_replies(self):
        for reply in (None, "", "   "):
            with self.subTest(reply=reply):
                data = {"choices": [{"message": {"content": reply}}]}
                _, session = self.response(data=data)
                with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
                    result = await bot.query_llm([])
                self.assertIn("empty response", result)

    async def test_connector_and_unexpected_errors(self):
        connection_key = SimpleNamespace(host="llm", port=80, ssl=False)
        connector_error = aiohttp.ClientConnectorError(connection_key, OSError("offline"))
        with patch.object(bot, "get_http_session", AsyncMock(side_effect=connector_error)):
            result = await bot.query_llm([])
        self.assertEqual(result, bot.MSG_API_UNREACHABLE)
        self.assertNotIn(bot.API_BASE_URL, result)
        self.assertNotIn("llm:80", result)

        with patch.object(bot, "get_http_session", AsyncMock(side_effect=RuntimeError("boom"))):
            result = await bot.query_llm([])
        self.assertEqual(result, bot.MSG_API_FAILED)
        self.assertNotIn("RuntimeError", result)
        self.assertNotIn("boom", result)

    async def test_reasoning_is_stripped_and_reasoning_only_is_empty(self):
        cases = {
            "<think>plan</think>\n\nanswer": "answer",
            "<think>only thinking, cut off by max_tokens": bot.MSG_API_EMPTY,
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                data = {"choices": [{"message": {"content": reply}}]}
                _, session = self.response(data=data)
                with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
                    result = await bot.query_llm([])
                self.assertEqual(result, expected)

    async def test_timeouts_report_a_timeout(self):
        for error in (asyncio.TimeoutError(), aiohttp.ServerTimeoutError("read timed out")):
            with self.subTest(error=error):
                _, session = self.response()
                session.post.side_effect = error
                with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
                    result = await bot.query_llm([])
                self.assertEqual(result, bot.MSG_API_TIMEOUT)


class MessageUtilityTests(unittest.TestCase):
    def setUp(self):
        bot._last_request.clear()

    def test_clean_message_content_resolves_all_mentions(self):
        bot_user = SimpleNamespace(id=1, name="Bot")
        displayed_user = SimpleNamespace(id=2, name="fallback", display_name="Alice")
        fallback_user = SimpleNamespace(id=3, name="Bob")
        role = SimpleNamespace(id=4, name="Admins")
        channel = SimpleNamespace(id=5, name="general")
        message = FakeMessage(
            " <@1> <@!1> hi <@2> <@!3> <@&4> <#5> ",
            fallback_user,
            mentions=[bot_user, displayed_user, fallback_user],
            role_mentions=[role],
            channel_mentions=[channel],
        )
        self.assertEqual(
            bot.clean_message_content(message, bot_user),
            "hi Alice Bob @Admins #general",
        )

    def test_strip_reasoning(self):
        cases = {
            "plain answer": "plain answer",
            "<think>a\nb</think>\nanswer": "answer",
            "<think>a</think>one <think>b</think>two": "one two",
            # The chat template opened the block, so only the close appears.
            "reasoning here\n</think>\n\nanswer": "answer",
            # Generation stopped mid-thought: nothing after <think> is an answer.
            "intro <think>unfinished": "intro",
            "<think>unfinished": "",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(bot.strip_reasoning(text), expected)

    def test_truncate_response(self):
        cases = [
            ("fits", 10, "fits"),
            ("one two three", 10, "one two…"),
            ("one\ntwo three", 10, "one\ntwo…"),
            # No usable word boundary: hard cut.
            ("abcdefghijkl", 5, "abcd…"),
            ("a bcdefghijkl", 6, "a bcd…"),
            ("abc", 1, "…"),
        ]
        for text, limit, expected in cases:
            with self.subTest(text=text, limit=limit):
                result = bot.truncate_response(text, limit)
                self.assertEqual(result, expected)
                self.assertLessEqual(len(result), limit)

    def test_get_trigger(self):
        bot_user = SimpleNamespace(id=1, name="Bot")
        alice = SimpleNamespace(id=2, name="Alice")
        guild = SimpleNamespace(id=10)

        def reply_to(author):
            resolved = SimpleNamespace(author=author) if author else SimpleNamespace()
            return SimpleNamespace(resolved=resolved)

        cases = [
            (FakeMessage("<@1> hi", alice, mentions=[bot_user], guild=guild), "mention"),
            # A reply with the ping on also lists the bot in mentions.
            (FakeMessage("hi", alice, mentions=[bot_user], guild=guild,
                         reference=reply_to(bot_user)), "mention"),
            (FakeMessage("hi", alice, guild=guild, reference=reply_to(bot_user)), "reply"),
            (FakeMessage("hi", alice), "dm"),
            (FakeMessage("hi", alice, guild=guild), None),
            (FakeMessage("hi", alice, guild=guild, reference=reply_to(alice)), None),
            # Deleted replied-to message (no author) or one Discord didn't include.
            (FakeMessage("hi", alice, guild=guild, reference=reply_to(None)), None),
            (FakeMessage("hi", alice, guild=guild,
                         reference=SimpleNamespace(resolved=None)), None),
        ]
        for message, expected in cases:
            with self.subTest(content=message.content, expected=expected):
                self.assertEqual(bot.get_trigger(message, bot_user), expected)

    def test_split_short_newline_space_hard_limit_and_whitespace(self):
        self.assertEqual(bot.split_discord_message("short", 10), ["short"])
        self.assertEqual(bot.split_discord_message("123456\n7890", 10), ["123456", "7890"])
        self.assertEqual(bot.split_discord_message("123456 7890", 10), ["123456", "7890"])
        self.assertEqual(bot.split_discord_message("12345678901", 10), ["1234567890", "1"])
        self.assertEqual(bot.split_discord_message("          x", 10), ["x"])
        self.assertEqual(bot.split_discord_message("1234567890 ", 10), ["1234567890"])

    def test_cooldown_disabled_blocked_expired_and_stale_cleanup(self):
        with patch.object(bot, "USER_COOLDOWN_SECONDS", 0):
            self.assertIsNone(bot.record_user_request(1, 1.0))
            self.assertEqual(bot._last_request, {})

        with patch.object(bot, "USER_COOLDOWN_SECONDS", 5):
            bot._last_request.update({2: 1.0, 3: 9.0})
            self.assertIsNone(bot.record_user_request(1, 10.0))
            self.assertNotIn(2, bot._last_request)
            self.assertEqual(bot.record_user_request(1, 12.0), 3.0)
            self.assertIsNone(bot.record_user_request(1, 15.0))
            self.assertEqual(bot._last_request[1], 15.0)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_shot_and_empty_prompt(self):
        bot_user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        message = FakeMessage("<@1> hello", author, mentions=[bot_user])
        self.assertEqual(
            await bot.build_context_messages(message, bot_user, 1),
            [{"role": "user", "content": "Alice: hello"}],
        )
        message.content = "<@1>"
        self.assertEqual(await bot.build_context_messages(message, bot_user, 1), [])

    async def test_history_is_oldest_first_and_skips_empty_messages(self):
        bot_user = SimpleNamespace(id=1, name="Bot")
        alice = SimpleNamespace(id=2, name="fallback", display_name="Alice")
        bob = SimpleNamespace(id=3, name="Bob")
        oldest = FakeMessage("old", alice)
        assistant = FakeMessage("answer", bot_user)
        empty = FakeMessage("   ", bob)
        channel = FakeChannel([empty, assistant, oldest])
        trigger = FakeMessage("<@1> now", bob, mentions=[bot_user], channel=channel)
        self.assertEqual(
            await bot.build_context_messages(trigger, bot_user, 4),
            [
                {"role": "user", "content": "Alice: old"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "Bob: now"},
            ],
        )

    async def test_history_alternates_roles_and_skips_status_replies(self):
        bot_user = SimpleNamespace(id=1, name="Bot")
        alice = SimpleNamespace(id=2, name="Alice")
        bob = SimpleNamespace(id=3, name="Bob")
        history = [  # oldest-first here; FakeChannel yields newest-first
            FakeMessage("stale answer", bot_user),
            FakeMessage("hi", alice),
            FakeMessage("hey", bob),
            FakeMessage("part one", bot_user),
            FakeMessage(bot.MSG_API_ERROR, bot_user),
            FakeMessage("part two", bot_user),
            FakeMessage(bot.MSG_BUSY, bot_user),
            FakeMessage("more", alice),
        ]
        channel = FakeChannel(list(reversed(history)))
        trigger = FakeMessage("<@1> go", bob, mentions=[bot_user], channel=channel)
        self.assertEqual(
            await bot.build_context_messages(trigger, bot_user, len(history) + 1),
            [
                {"role": "user", "content": "Alice: hi\nBob: hey"},
                {"role": "assistant", "content": "part one\npart two"},
                {"role": "user", "content": "Alice: more\nBob: go"},
            ],
        )

    async def test_history_http_error_degrades_to_trigger(self):
        error = discord.HTTPException(RawResponse(), "history failed")
        bot_user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        channel = FakeChannel(history_error=error)
        trigger = FakeMessage("<@1> hello", author, mentions=[bot_user], channel=channel)
        self.assertEqual(
            await bot.build_context_messages(trigger, bot_user, 3),
            [{"role": "user", "content": "Alice: hello"}],
        )


class MembershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bot._membership_cache.clear()

    def tearDown(self):
        bot._membership_cache.clear()

    def make_client(self, guilds):
        return SimpleNamespace(get_guild=lambda guild_id: guilds.get(guild_id))

    def guild(self, members=(), error=None):
        async def fetch_member(user_id):
            if error is not None:
                raise error
            if user_id not in members:
                raise discord.NotFound(RawResponse(), "Unknown Member")
            return SimpleNamespace(id=user_id)

        return SimpleNamespace(fetch_member=AsyncMock(side_effect=fetch_member))

    async def test_member_of_any_allowed_guild_is_accepted_and_cached(self):
        other, home = self.guild(), self.guild(members={2})
        # Guild 12 is allowed but the bot isn't in it.
        client = self.make_client({10: other, 11: home})
        with patch.object(bot, "allowed_guilds", {10, 11, 12}):
            self.assertTrue(await bot.is_allowed_guild_member(client, 2))
            self.assertTrue(await bot.is_allowed_guild_member(client, 2))
        home.fetch_member.assert_awaited_once_with(2)

    async def test_non_member_is_rejected_cached_and_expires(self):
        home = self.guild()
        client = self.make_client({10: home})
        with (
            patch.object(bot, "allowed_guilds", {10}),
            patch.object(bot.time, "monotonic", return_value=100.0) as clock,
        ):
            self.assertFalse(await bot.is_allowed_guild_member(client, 2))
            self.assertFalse(await bot.is_allowed_guild_member(client, 2))
            self.assertEqual(home.fetch_member.await_count, 1)
            clock.return_value = 100.0 + bot.MEMBERSHIP_CACHE_SECONDS
            self.assertFalse(await bot.is_allowed_guild_member(client, 2))
            self.assertEqual(home.fetch_member.await_count, 2)

    async def test_lookup_errors_fail_closed_without_caching(self):
        error = discord.HTTPException(RawResponse(), "rate limited")
        failing, home = self.guild(error=error), self.guild(members={2})
        with patch.object(bot, "allowed_guilds", {10}):
            client = self.make_client({10: failing})
            self.assertFalse(await bot.is_allowed_guild_member(client, 2))
            self.assertNotIn(2, bot._membership_cache)
        # A failure in one guild doesn't block membership found in another.
        with patch.object(bot, "allowed_guilds", {10, 11}):
            client = self.make_client({10: failing, 11: home})
            self.assertTrue(await bot.is_allowed_guild_member(client, 2))


class ClientEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bot._last_request.clear()
        bot._active_users.clear()
        bot._membership_cache.clear()

    def tearDown(self):
        self.assertEqual(bot._active_users, set())

    def make_client(self, user=None):
        client = bot.create_bot()
        client._connection.user = user
        return client

    async def test_mentions_are_disabled_for_every_fresh_client(self):
        for _ in range(2):
            client = self.make_client()
            self.assertEqual(client.allowed_mentions.to_dict(), {"parse": []})
            self.assertFalse(client.allowed_mentions.replied_user)

    async def test_other_bots_and_webhooks_cannot_trigger_requests(self):
        user = SimpleNamespace(id=1, name="Bot")
        client = self.make_client(user)
        with patch.object(bot, "query_llm", AsyncMock()) as query:
            for is_bot, webhook_id in ((True, None), (True, 123), (False, 123)):
                author = SimpleNamespace(id=2, name="Other", bot=is_bot)
                message = FakeMessage(
                    "<@1> hello", author, mentions=[user], webhook_id=webhook_id
                )
                await client.on_message(message)
                message.reply.assert_not_awaited()
                message.add_reaction.assert_not_awaited()
            query.assert_not_awaited()
        self.assertEqual(bot._last_request, {})

    async def test_lifecycle_events_and_ready_branches(self):
        user = SimpleNamespace(id=1, name="Bot")
        client = self.make_client(user)
        with patch.object(bot, "allowed_guilds", {10}):
            await client.on_ready()
        with patch.object(bot, "allowed_guilds", set()):
            await client.on_ready()
        client._connection.user = None
        await client.on_ready()
        await client.on_disconnect()
        await client.on_resumed()

    async def test_disconnect_burst_requests_one_fresh_gateway_session(self):
        recycle_requested = asyncio.Event()
        client = bot.create_bot(recycle_requested)
        client.close = AsyncMock()

        # The first timestamp ages out. The next five all fit in the window and
        # trigger a recycle; further disconnect events must not close it twice.
        with patch.object(bot.time, "monotonic", side_effect=[0, 61, 62, 63, 64, 65, 66]):
            for _ in range(7):
                await client.on_disconnect()

        self.assertTrue(recycle_requested.is_set())
        client.close.assert_awaited_once()

    async def test_on_message_ignores_unusable_messages(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        client = self.make_client(None)
        message = FakeMessage("hello", author)
        await client.on_message(message)

        client._connection.user = user
        self_message = FakeMessage("<@1> hello", user, mentions=[user])
        await client.on_message(self_message)
        unmentioned = FakeMessage("hello", author, guild=SimpleNamespace(id=10))
        await client.on_message(unmentioned)
        for ignored in (message, self_message, unmentioned):
            ignored.reply.assert_not_awaited()
            ignored.channel.send.assert_not_awaited()

    async def test_on_message_answers_replies_and_dms_but_not_system_messages(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        guild = SimpleNamespace(id=10)
        reply_ref = SimpleNamespace(resolved=SimpleNamespace(author=user))
        client = self.make_client(user)
        with (
            patch.object(bot, "allowed_guilds", set()),
            patch.object(bot, "USER_COOLDOWN_SECONDS", 0),
            patch.object(bot, "MAX_CONTEXT_MESSAGES", 1),
            patch.object(bot, "query_llm", AsyncMock(return_value="answer")) as query,
        ):
            reply = FakeMessage("and then?", author, guild=guild, reference=reply_ref)
            await client.on_message(reply)
            dm = FakeMessage("hello", author)
            await client.on_message(dm)
            self.assertEqual(query.await_count, 2)
            reply.channel.send.assert_awaited_once()
            dm.channel.send.assert_awaited_once()

            # A pin notice references the pinned bot message but isn't a request.
            pin = FakeMessage("", author, guild=guild, reference=reply_ref, system=True)
            await client.on_message(pin)
            # Text-less replies and DMs (e.g. an image) get no usage hint.
            empty_reply = FakeMessage("", author, guild=guild, reference=reply_ref)
            await client.on_message(empty_reply)
            empty_dm = FakeMessage("", author)
            await client.on_message(empty_dm)
            self.assertEqual(query.await_count, 2)
            for message in (pin, empty_reply, empty_dm):
                message.reply.assert_not_awaited()
                message.channel.send.assert_not_awaited()

        with (
            patch.object(bot, "allowed_guilds", {10}),
            patch.object(bot, "USER_COOLDOWN_SECONDS", 0),
            patch.object(bot, "MAX_CONTEXT_MESSAGES", 1),
            patch.object(bot, "query_llm", AsyncMock(return_value="answer")) as query,
            patch.object(bot, "is_allowed_guild_member", AsyncMock(return_value=False)) as member,
        ):
            outsider_dm = FakeMessage("hello", author)
            await client.on_message(outsider_dm)
            member.assert_awaited_once_with(client, author.id)
            outsider_dm.reply.assert_not_awaited()
            outsider_dm.channel.send.assert_not_awaited()

            member.return_value = True
            member_dm = FakeMessage("hello", author)
            await client.on_message(member_dm)
            query.assert_awaited_once()
            member_dm.channel.send.assert_awaited_once()

    async def test_on_message_enforces_guild_allowlist_and_rejects_empty_prompt(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        client = self.make_client(user)

        with patch.object(bot, "allowed_guilds", {10}):
            wrong_guild = FakeMessage(
                "<@1> hello", author, mentions=[user], guild=SimpleNamespace(id=11)
            )
            await client.on_message(wrong_guild)
            dm = FakeMessage("<@1> hello", author, mentions=[user])
            await client.on_message(dm)
            wrong_guild.reply.assert_not_awaited()
            dm.reply.assert_not_awaited()

        with patch.object(bot, "allowed_guilds", set()), patch.object(
            bot, "USER_COOLDOWN_SECONDS", 5
        ):
            empty = FakeMessage("<@1>", author, mentions=[user])
            await client.on_message(empty)
            empty.reply.assert_awaited_once_with(bot.MSG_EMPTY_PROMPT)
            # Repeated bare mentions inside the cooldown get no further hints.
            repeat = FakeMessage("<@1>", author, mentions=[user])
            await client.on_message(repeat)
            repeat.reply.assert_not_awaited()

        with patch.object(bot, "allowed_guilds", set()), patch.object(
            bot, "USER_COOLDOWN_SECONDS", 0
        ):
            forbidden = FakeMessage("<@1>", author, mentions=[user])
            forbidden.reply.side_effect = discord.Forbidden(RawResponse(), "denied")
            await client.on_message(forbidden)
            forbidden.reply.assert_awaited_once()

    async def test_on_message_cooldown_reacts_and_ignores_reaction_failure(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        client = self.make_client(user)

        for error in (None, discord.HTTPException(RawResponse(), "reaction failed")):
            with self.subTest(error=error), patch.object(
                bot, "record_user_request", return_value=2.0
            ):
                message = FakeMessage("<@1> hello", author, mentions=[user])
                message.add_reaction.side_effect = error
                await client.on_message(message)
                message.add_reaction.assert_awaited_once_with("🕒")
                message.reply.assert_not_awaited()

    async def test_on_message_queries_and_sends_all_chunks(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        guild = SimpleNamespace(id=10)
        channel = FakeChannel()
        message = FakeMessage(
            "<@1> hello", author, mentions=[user], channel=channel, guild=guild
        )
        client = self.make_client(user)
        context = [{"role": "user", "content": "hello"}]
        with (
            patch.object(bot, "allowed_guilds", {10}),
            patch.object(bot, "record_user_request", return_value=None),
            patch.object(bot, "build_context_messages", AsyncMock(return_value=context)) as build,
            patch.object(bot, "query_llm", AsyncMock(return_value="first\nsecond")) as query,
            patch.object(bot, "split_discord_message", return_value=["first", "second"]),
        ):
            await client.on_message(message)
        build.assert_awaited_once_with(message, user, bot.MAX_CONTEXT_MESSAGES)
        query.assert_awaited_once_with(context)
        message.to_reference.assert_called_once_with(fail_if_not_exists=False)
        message.reply.assert_not_awaited()
        self.assertEqual(
            channel.send.await_args_list,
            [call("first", reference=message.outgoing_reference), call("second")],
        )

    async def test_global_limit_rejects_without_queue_or_cooldown_across_clients(self):
        user = SimpleNamespace(id=1, name="Bot")
        first = FakeMessage("<@1> first", SimpleNamespace(id=2), mentions=[user])
        second = FakeMessage("<@1> second", SimpleNamespace(id=3), mentions=[user])
        client = self.make_client(user)
        started, release = asyncio.Event(), asyncio.Event()

        async def generate(messages):
            started.set()
            await release.wait()
            return "answer"

        with (
            patch.object(bot, "allowed_guilds", set()),
            patch.object(bot, "MAX_CONCURRENT_REQUESTS", 1),
            patch.object(bot, "USER_COOLDOWN_SECONDS", 5),
            patch.object(bot, "MAX_CONTEXT_MESSAGES", 1),
            patch.object(bot, "query_llm", AsyncMock(side_effect=generate)) as query,
        ):
            task = asyncio.create_task(client.on_message(first))
            try:
                await asyncio.wait_for(started.wait(), 1)
                # A replacement client must share the existing request budget.
                replacement = self.make_client(user)
                await asyncio.wait_for(replacement.on_message(second), 1)
                self.assertIn("busy", second.reply.call_args.args[0])
                self.assertNotIn(3, bot._last_request)
                self.assertEqual(query.await_count, 1)
                self.assertEqual(bot._active_users, {2})
            finally:
                release.set()
                await task
            self.assertEqual(bot._active_users, set())
            await replacement.on_message(second)
            self.assertEqual(query.await_count, 2)

    async def test_per_user_limit_with_cooldown_disabled_and_global_capacity_available(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2)
        first = FakeMessage("<@1> first", author, mentions=[user])
        duplicate = FakeMessage("<@1> again", author, mentions=[user])
        other = FakeMessage("<@1> other", SimpleNamespace(id=3), mentions=[user])
        overflow = FakeMessage("<@1> overflow", SimpleNamespace(id=4), mentions=[user])
        client = self.make_client(user)
        started, release = asyncio.Event(), asyncio.Event()

        async def generate(messages):
            started.set()
            await release.wait()
            return "answer"

        with (
            patch.object(bot, "allowed_guilds", set()),
            patch.object(bot, "MAX_CONCURRENT_REQUESTS", 2),
            patch.object(bot, "USER_COOLDOWN_SECONDS", 0),
            patch.object(bot, "MAX_CONTEXT_MESSAGES", 1),
            patch.object(bot, "query_llm", AsyncMock(side_effect=generate)) as query,
        ):
            tasks = [asyncio.create_task(client.on_message(first))]
            try:
                await asyncio.wait_for(started.wait(), 1)
                await asyncio.wait_for(client.on_message(duplicate), 1)
                self.assertIn("busy", duplicate.reply.call_args.args[0])
                self.assertEqual(query.await_count, 1)
                started.clear()
                tasks.append(asyncio.create_task(client.on_message(other)))
                await asyncio.wait_for(started.wait(), 1)
                self.assertEqual(bot._active_users, {2, 3})
                await asyncio.wait_for(client.on_message(overflow), 1)
                self.assertIn("busy", overflow.reply.call_args.args[0])
                self.assertEqual(query.await_count, 2)
            finally:
                release.set()
                await asyncio.gather(*tasks)

    async def test_capacity_held_during_history_and_delivery_and_released_on_cancellation(self):
        for stage in ("history", "query", "reply", "send"):
            with self.subTest(stage=stage):
                user = SimpleNamespace(id=1, name="Bot")
                client = self.make_client(user)
                message = FakeMessage("<@1> hello", SimpleNamespace(id=2), mentions=[user])
                rejected = FakeMessage("<@1> hi", SimpleNamespace(id=3), mentions=[user])
                started = asyncio.Event()

                async def block(*args, **kwargs):
                    started.set()
                    await asyncio.Event().wait()

                build = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
                query = AsyncMock(return_value="x" * 2001)
                target, call_number = stage_target(stage, build, query, message)
                target.side_effect = on_call(call_number, block)
                with (
                    patch.object(bot, "allowed_guilds", set()),
                    patch.object(bot, "MAX_CONCURRENT_REQUESTS", 1),
                    patch.object(bot, "USER_COOLDOWN_SECONDS", 0),
                    patch.object(bot, "build_context_messages", build),
                    patch.object(bot, "query_llm", query),
                ):
                    task = asyncio.create_task(client.on_message(message))
                    try:
                        await asyncio.wait_for(started.wait(), 1)
                        self.assertEqual(bot._active_users, {2})
                        # Failed busy notifications must not disturb the active slot.
                        rejected.reply.side_effect = discord.Forbidden(RawResponse(), "denied")
                        await client.on_message(rejected)
                        self.assertEqual(bot._active_users, {2})
                    finally:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    self.assertEqual(bot._active_users, set())

    async def test_capacity_released_on_history_api_and_delivery_failures(self):
        for stage in ("history", "query", "reply", "send"):
            with self.subTest(stage=stage):
                user = SimpleNamespace(id=1, name="Bot")
                client = self.make_client(user)
                message = FakeMessage("<@1> hello", SimpleNamespace(id=2), mentions=[user])
                build = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
                query = AsyncMock(return_value="x" * 2001)
                target, call_number = stage_target(stage, build, query, message)

                async def fail(*args, **kwargs):
                    raise RuntimeError("failed")

                target.side_effect = on_call(call_number, fail)
                with (
                    patch.object(bot, "allowed_guilds", set()),
                    patch.object(bot, "USER_COOLDOWN_SECONDS", 0),
                    patch.object(bot, "build_context_messages", build),
                    patch.object(bot, "query_llm", query),
                ):
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        await client.on_message(message)
                    self.assertEqual(bot._active_users, set())
                    target.side_effect = None
                    await client.on_message(message)


class SupervisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        bot._http_session = None

    async def test_clean_exit_and_signal_handler_closes_running_client(self):
        callbacks = []
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(side_effect=lambda sig, callback: callbacks.append(callback)),
            time=Mock(return_value=0.0),
        )
        client = FakeSupervisedClient()

        async def start_and_signal():
            callbacks[0]()
            await asyncio.sleep(0)

        client.start_effect = start_and_signal
        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", return_value=client),
            patch.object(bot, "_close_http_session", AsyncMock()) as close_http,
        ):
            await bot.run_supervised()
        self.assertTrue(client.closed)
        self.assertGreaterEqual(client.close_calls, 1)
        close_http.assert_awaited_once()
        self.assertEqual(
            {call.args[0] for call in fake_loop.add_signal_handler.call_args_list},
            {signal.SIGINT, signal.SIGTERM},
        )

    async def test_signal_before_client_creation_skips_loop(self):
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(side_effect=lambda sig, callback: callback()),
            time=Mock(return_value=0.0),
        )
        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot") as create,
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        create.assert_not_called()

    async def test_unsupported_signal_handlers_and_fatal_startup(self):
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(side_effect=NotImplementedError),
            time=Mock(return_value=0.0),
        )
        client = FakeSupervisedClient(discord.LoginFailure("bad token"))
        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", return_value=client),
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        self.assertTrue(client.closed)

    async def test_cancellation_closes_client_in_finally(self):
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(),
            time=Mock(return_value=0.0),
        )
        client = FakeSupervisedClient(asyncio.CancelledError(), close_on_exit=False)
        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", return_value=client),
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        self.assertEqual(client.close_calls, 1)

    async def test_exception_retries_after_timeout_and_resets_stable_backoff(self):
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(),
            time=Mock(side_effect=[0.0, 61.0, 62.0]),
        )
        failed = FakeSupervisedClient(RuntimeError("gateway down"))
        clean = FakeSupervisedClient()
        async def timeout(awaitable, *, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", side_effect=[failed, clean]),
            patch.object(bot.asyncio, "wait_for", AsyncMock(side_effect=timeout)) as wait,
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        wait.assert_awaited_once()
        self.assertTrue(clean.started)

    async def test_exception_retries_when_wait_completes(self):
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(),
            time=Mock(side_effect=[0.0, 1.0, 2.0]),
        )
        failed = FakeSupervisedClient(RuntimeError("gateway down"))
        clean = FakeSupervisedClient()
        async def finish_wait(awaitable, *, timeout):
            awaitable.close()
            return True

        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", side_effect=[failed, clean]),
            patch.object(bot.asyncio, "wait_for", AsyncMock(side_effect=finish_wait)) as wait,
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        wait.assert_awaited_once()
        self.assertTrue(clean.started)

    async def test_requested_gateway_recycle_starts_a_fresh_client(self):
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(),
            time=Mock(side_effect=[0.0, 61.0, 62.0]),
        )
        recycled = FakeSupervisedClient()
        clean = FakeSupervisedClient()
        clients = iter((recycled, clean))

        def create(recycle_requested):
            client = next(clients)
            if client is recycled:
                client.start_effect = recycle_requested.set
            return client

        async def timeout(awaitable, *, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", side_effect=create),
            patch.object(bot.asyncio, "wait_for", AsyncMock(side_effect=timeout)) as wait,
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        wait.assert_awaited_once()
        self.assertTrue(recycled.started)
        self.assertTrue(clean.started)

    async def test_exception_after_signal_does_not_restart(self):
        callbacks = []
        fake_loop = SimpleNamespace(
            add_signal_handler=Mock(side_effect=lambda sig, callback: callbacks.append(callback)),
            time=Mock(return_value=0.0),
        )

        def signal_then_fail():
            callbacks[0]()
            raise RuntimeError("gateway down")

        client = FakeSupervisedClient(signal_then_fail)
        with (
            patch.object(bot.asyncio, "get_running_loop", return_value=fake_loop),
            patch.object(bot, "create_bot", return_value=client) as create,
            patch.object(bot, "_close_http_session", AsyncMock()),
        ):
            await bot.run_supervised()
        create.assert_called_once()
        await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
