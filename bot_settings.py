"""Настройки, которые админ меняет из Telegram (/settings) без правки .env.

Значение по умолчанию берётся из .env (то, что модуль прочитал при старте),
переопределение хранится в bot_settings.json и применяется сразу — прямо
в переменную модуля, которую читает код. Перезапуск не нужен.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

SETTINGS_FILE = "bot_settings.json"


@dataclass
class Setting:
    key: str                    # имя в bot_settings.json (совпадает с переменной .env)
    title: str                  # подпись на кнопке
    module: ModuleType          # где лежит переменная
    attr: str                   # имя переменной в модуле
    choices: list[tuple[Any, str]]   # (значение, подпись); кнопка листает по кругу
    hint: str = ""              # пояснение в тексте панели
    default: Any = field(default=None, init=False)  # значение из .env на старте

    def current(self) -> Any:
        return getattr(self.module, self.attr)

    def label(self, value: Any) -> str:
        for v, text in self.choices:
            if v == value:
                return text
        return str(value)  # значение из .env, которого нет среди вариантов

    def next_value(self) -> Any:
        values = [v for v, _ in self.choices]
        cur = self.current()
        if cur in values:
            return values[(values.index(cur) + 1) % len(values)]
        return values[0]


class SettingsStore:
    def __init__(self, settings: list[Setting], path: str = SETTINGS_FILE):
        self.settings = settings
        self.path = path
        self.by_key = {s.key: s for s in settings}
        for s in settings:
            s.default = s.current()

    def _read(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logging.error(f"Не удалось прочитать {self.path}: {e}")
            return {}

    def _write(self, data: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def load(self) -> None:
        """Применяет сохранённые переопределения поверх значений из .env."""
        for key, value in self._read().items():
            s = self.by_key.get(key)
            if s is None:
                continue
            setattr(s.module, s.attr, value)
            logging.info(f"Настройка {key} = {value!r} (из {self.path})")

    def set(self, key: str, value: Any) -> None:
        s = self.by_key[key]
        setattr(s.module, s.attr, value)
        data = self._read()
        if value == s.default:
            data.pop(key, None)   # совпадает с .env — переопределение не нужно
        else:
            data[key] = value
        self._write(data)

    def cycle(self, key: str) -> Any:
        value = self.by_key[key].next_value()
        self.set(key, value)
        return value

    def reset(self) -> None:
        for s in self.settings:
            setattr(s.module, s.attr, s.default)
        if os.path.exists(self.path):
            os.remove(self.path)

    def is_overridden(self, key: str) -> bool:
        s = self.by_key[key]
        return s.current() != s.default
