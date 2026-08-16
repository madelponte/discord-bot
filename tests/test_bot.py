import asyncio
import os
import runpy
import signal
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
    ):
        self.content = content
        self.author = author
        self.mentions = mentions or []
        self.role_mentions = role_mentions or []
        self.channel_mentions = channel_mentions or []
        self.channel = channel or FakeChannel()
        self.guild = guild
        self.reply = AsyncMock()
        self.add_reaction = AsyncMock()


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
            "MAX_TOKENS": "0",
            "MAX_CONTEXT_MESSAGES": "0",
            "USER_COOLDOWN_SECONDS": "-1",
        }
        with patch.dict(os.environ, valid, clear=True):
            namespace = runpy.run_path(bot.__file__, run_name="not_main")
        self.assertEqual(namespace["allowed_guilds"], {1, 2})
        self.assertEqual(namespace["MAX_TOKENS"], 1)
        self.assertEqual(namespace["MAX_CONTEXT_MESSAGES"], 1)
        self.assertEqual(namespace["USER_COOLDOWN_SECONDS"], 0)

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
        with patch.object(bot.aiohttp, "ClientSession", side_effect=[first, second]) as constructor:
            self.assertIs(await bot.get_http_session(), first)
            self.assertIs(await bot.get_http_session(), first)
            first.closed = True
            self.assertIs(await bot.get_http_session(), second)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(
            constructor.call_args.kwargs["timeout"].total,
            bot.API_TIMEOUT_SECONDS,
        )

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
        self.assertEqual(result, " answer ")
        url, = session.post.call_args.args
        self.assertEqual(url, f"{bot.API_BASE_URL.rstrip('/')}/chat/completions")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], bot.MODEL_NAME)
        self.assertEqual(payload["max_tokens"], bot.MAX_TOKENS)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["messages"][1:], messages)
        response.json.assert_awaited_once_with(content_type=None)

    async def test_non_200_response(self):
        _, session = self.response(status=503, text="backend unavailable")
        with patch.object(bot, "get_http_session", AsyncMock(return_value=session)):
            result = await bot.query_llm([])
        self.assertEqual(result, "⚠️ The LLM server returned an error.")
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
        self.assertEqual(result, "⚠️ Cannot reach the LLM server right now.")
        self.assertNotIn(bot.API_BASE_URL, result)
        self.assertNotIn("llm:80", result)

        with patch.object(bot, "get_http_session", AsyncMock(side_effect=RuntimeError("boom"))):
            result = await bot.query_llm([])
        self.assertEqual(
            result,
            "⚠️ An unexpected error occurred while contacting the LLM server.",
        )
        self.assertNotIn("RuntimeError", result)
        self.assertNotIn("boom", result)


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
            [{"role": "user", "content": "hello"}],
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
                {"role": "user", "content": "now"},
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
            [{"role": "user", "content": "hello"}],
        )


class ClientEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bot._last_request.clear()

    def make_client(self, user=None):
        client = bot.create_bot()
        client._connection.user = user
        return client

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

    async def test_on_message_ignores_unusable_messages(self):
        user = SimpleNamespace(id=1, name="Bot")
        author = SimpleNamespace(id=2, name="Alice")
        client = self.make_client(None)
        message = FakeMessage("hello", author)
        await client.on_message(message)

        client._connection.user = user
        self_message = FakeMessage("<@1> hello", user, mentions=[user])
        await client.on_message(self_message)
        unmentioned = FakeMessage("hello", author)
        await client.on_message(unmentioned)
        message.reply.assert_not_awaited()
        self_message.reply.assert_not_awaited()
        unmentioned.reply.assert_not_awaited()

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

        with patch.object(bot, "allowed_guilds", set()):
            empty = FakeMessage("<@1>", author, mentions=[user])
            await client.on_message(empty)
            empty.reply.assert_awaited_once()

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
        message.reply.assert_awaited_once_with("first")
        channel.send.assert_awaited_once_with("second")


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
