"""Compatibility with rich-message blocks newer than aiogram's schema."""

from copy import deepcopy
import logging

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import ClientDecodeError
from aiogram.methods import GetUpdates
from aiogram.methods.base import Response, TelegramMethod, TelegramType
from pydantic import ValidationError


logger = logging.getLogger(__name__)


class CompatibleTelegramSession(AiohttpSession):
    def check_response(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        status_code: int,
        content: str,
    ) -> Response[TelegramType]:
        try:
            return super().check_response(bot, method, status_code, content)
        except ClientDecodeError as exc:
            if (
                not isinstance(method, GetUpdates)
                or not isinstance(exc.original, ValidationError)
                or not isinstance(exc.data, dict)
                or exc.data.get("ok") is not True
            ):
                raise

            # A single unknown block otherwise prevents polling from advancing
            # past the entire batch. Only remove rich_message from messages
            # whose top-level block discriminator aiogram cannot recognize.
            data = deepcopy(exc.data)
            removed = 0
            for error in exc.original.errors():
                location = error["loc"]
                if error["type"] != "union_tag_invalid" or len(location) < 6:
                    continue
                if (
                    location[0] != "result"
                    or location[-3:-1] != ("rich_message", "blocks")
                    or not isinstance(location[-1], int)
                ):
                    continue

                message = data
                try:
                    for key in location[:-3]:
                        message = message[key]
                except (KeyError, IndexError, TypeError):
                    continue
                if isinstance(message, dict) and "rich_message" in message:
                    del message["rich_message"]
                    removed += 1

            if not removed:
                raise

            # Validate again, preserving every update and all other message
            # fields. Unrelated validation and Telegram API errors still raise.
            response = super().check_response(
                bot, method, status_code, self.json_dumps(data)
            )
            logger.warning(
                "Ignored unsupported rich_message blocks in %d message(s) "
                "while parsing getUpdates; all updates were retained. "
                "Upgrade aiogram for full rich-message support.",
                removed,
            )
            return response
