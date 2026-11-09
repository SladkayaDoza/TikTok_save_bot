"""Exercise the real response parser with a schema that rejects new blocks.

The local aiogram 3.17 predates rich_message entirely and accepts it as an extra
field. These models reproduce the stricter discriminated union on the server.
"""

import json
from typing import Annotated, ClassVar, Literal
import unittest

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import ClientDecodeError, TelegramBadRequest
from aiogram.methods import GetUpdates, SendMessage
from aiogram.types import Message, Update
from pydantic import BaseModel, Field

from telegram_session import CompatibleTelegramSession


class Paragraph(BaseModel):
    type: Literal["paragraph"]
    text: str


class Photo(BaseModel):
    type: Literal["photo"]
    file_id: str


class RichMessage(BaseModel):
    blocks: list[Annotated[Paragraph | Photo, Field(discriminator="type")]]


class StrictMessage(Message):
    rich_message: RichMessage | None = None
    reply_to_message: "StrictMessage | None" = None


class StrictUpdate(Update):
    message: StrictMessage | None = None


class StrictGetUpdates(GetUpdates):
    __returning__: ClassVar = list[StrictUpdate]


class StrictSendMessage(SendMessage):
    __returning__: ClassVar = StrictMessage


def message(**fields):
    return {
        "message_id": 1,
        "date": 1794223171,
        "chat": {"id": -1001234567890, "type": "supergroup", "title": "Test"},
        **fields,
    }


def unknown_rich_message():
    return {"blocks": [
        {"type": "paragraph", "text": "Test"},
        {"type": "photo", "file_id": "test"},
        {"type": "buttons", "buttons": [{"url": "https://example.com"}]},
    ]}


class CompatibleTelegramSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = CompatibleTelegramSession()
        self.bot = Bot(token="123456:offline-test-token", session=self.session)

    def parse(self, updates, session=None):
        return (session or self.session).check_response(
            self.bot, StrictGetUpdates(), 200,
            json.dumps({"ok": True, "result": updates}),
        ).result

    def test_reproduces_server_error_without_compatibility(self):
        with self.assertRaises(ClientDecodeError) as caught:
            self.parse([
                {"update_id": 1, "message": message(rich_message=unknown_rich_message())},
            ], session=AiohttpSession())
        error = caught.exception.original.errors()[0]
        self.assertEqual(error["type"], "union_tag_invalid")
        self.assertEqual(error["loc"], ("result", 0, "message", "rich_message", "blocks", 2))

    def test_unknown_block_does_not_block_batch_or_lose_other_fields(self):
        updates = [
            {"update_id": 1, "message": message(
                rich_message=unknown_rich_message(), text="Original text", sender_tag="Test",
            )},
            {"update_id": 2, "message": message(text="https://vt.tiktok.com/test/")},
            {"update_id": 3, "message_reaction": {
                "chat": message()["chat"], "message_id": 1, "date": 1794223171,
                "user": {"id": 123, "is_bot": False, "first_name": "Test"},
                "old_reaction": [], "new_reaction": [{"type": "emoji", "emoji": "❤"}],
            }},
        ]
        with self.assertLogs("telegram_session", level="WARNING") as logs:
            result = self.parse(updates)
        self.assertEqual([update.update_id for update in result], [1, 2, 3])
        self.assertIsNone(result[0].message.rich_message)
        self.assertEqual(result[0].message.text, "Original text")
        self.assertEqual(result[0].message.model_extra["sender_tag"], "Test")
        self.assertEqual(result[1].message.text, "https://vt.tiktok.com/test/")
        self.assertEqual(result[2].message_reaction.new_reaction[0].emoji, "❤")
        self.assertIn("rich_message", updates[0]["message"])
        self.assertNotIn("example.com", logs.output[0])

    def test_handles_multiple_messages_and_nested_replies(self):
        result = self.parse([
            {"update_id": 1, "message": message(rich_message=unknown_rich_message())},
            {"update_id": 2, "message": message(
                text="Reply", reply_to_message=message(rich_message=unknown_rich_message()),
            )},
        ])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].message.text, "Reply")
        self.assertIsNone(result[1].message.reply_to_message.rich_message)

    def test_supported_rich_message_is_unchanged(self):
        result = self.parse([{"update_id": 1, "message": message(
            rich_message={"blocks": [{"type": "paragraph", "text": "Keep me"}]},
        )}])
        self.assertEqual(result[0].message.rich_message.blocks[0].text, "Keep me")

    def test_invalid_known_block_still_raises(self):
        with self.assertRaises(ClientDecodeError):
            self.parse([{"update_id": 1, "message": message(
                rich_message={"blocks": [{"type": "paragraph"}]},
            )}])

    def test_unrelated_validation_error_in_same_batch_still_raises(self):
        with self.assertRaises(ClientDecodeError):
            self.parse([
                {"update_id": 1, "message": message(rich_message=unknown_rich_message())},
                {"update_id": "invalid", "message": message(text="Test")},
            ])

    def test_non_polling_response_still_raises(self):
        with self.assertRaises(ClientDecodeError):
            self.session.check_response(
                self.bot, StrictSendMessage(chat_id=123, text="Test"), 200,
                json.dumps({"ok": True, "result": message(rich_message=unknown_rich_message())}),
            )

    def test_api_errors_and_invalid_json_still_raise(self):
        with self.assertRaises(TelegramBadRequest):
            self.session.check_response(
                self.bot, GetUpdates(), 400,
                json.dumps({"ok": False, "description": "Bad Request"}),
            )
        with self.assertRaises(ClientDecodeError):
            self.session.check_response(self.bot, GetUpdates(), 200, "not json")


if __name__ == "__main__":
    unittest.main()
