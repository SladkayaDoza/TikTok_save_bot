"""Распознавание речи: выбор провайдера в одном месте.

Потребителей два — команда `meow` (ручная транскрипция реплаем) и воркер
эмбеддингов. Оба ходят сюда, поэтому провайдер настраивается один раз.

Провайдер задаётся переменной TRANSCRIBE_PROVIDER:
    auto        — DeepInfra, если есть её ключ, иначе OpenRouter, иначе выключено
                  (по умолчанию; при обновлении ничего не меняется само собой)
    deepinfra   — только DeepInfra
    openrouter  — только OpenRouter
    off         — распознавание выключено

Ключи не обязательны: без них бот запускается и работает, а функции,
которым нужна речь, честно сообщают, что не настроены.
"""
from __future__ import annotations

import logging
import os
import re

import requests

log = logging.getLogger(__name__)

PROVIDERS = {
    "deepinfra": {
        "env": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "title": "DeepInfra",
        "keys_url": "https://deepinfra.com/dash/api_keys",
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "title": "OpenRouter",
        "keys_url": "https://openrouter.ai/keys",
    },
}

# Порядок автоподбора. DeepInfra первой намеренно: у кого она уже настроена,
# после обновления продолжит работать ровно как раньше, без сюрпризов.
_AUTO_ORDER = ("deepinfra", "openrouter")

TRANSCRIBE_PROVIDER = (os.getenv("TRANSCRIBE_PROVIDER") or "auto").strip().lower()
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-large-v3")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "ru")

# Whisper на музыке и тишине выдаёт русские «титры». В поисковый индекс их
# пускать нельзя, да и пользователю показывать незачем.
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"субтитры\s+(делал|сделал|создавал|подготовил)",
    r"редактор\s+субтитров",
    r"корректор\s+субтитров",
    r"продолжение\s+следует",
    r"dimatorzok",
    r"спасибо\s+за\s+просмотр",
    r"подписывайтесь\s+на\s+канал",
]]


def resolve() -> tuple[str | None, str | None, str | None]:
    """Возвращает (provider, base_url, api_key) или (None, None, None)."""
    if TRANSCRIBE_PROVIDER == "off":
        return None, None, None

    if TRANSCRIBE_PROVIDER in PROVIDERS:
        cfg = PROVIDERS[TRANSCRIBE_PROVIDER]
        key = os.getenv(cfg["env"])
        if key:
            return TRANSCRIBE_PROVIDER, cfg["base_url"], key
        return None, None, None

    # auto (и любое неизвестное значение — падаем в безопасный автоподбор)
    for name in _AUTO_ORDER:
        cfg = PROVIDERS[name]
        key = os.getenv(cfg["env"])
        if key:
            return name, cfg["base_url"], key
    return None, None, None


def is_available() -> bool:
    return resolve()[0] is not None


def provider_title() -> str:
    name = resolve()[0]
    return PROVIDERS[name]["title"] if name else "выключено"


def unavailable_reason() -> str | None:
    """Человекочитаемая причина, почему распознавание не работает."""
    if is_available():
        return None

    if TRANSCRIBE_PROVIDER == "off":
        return "распознавание отключено (TRANSCRIBE_PROVIDER=off)"

    if TRANSCRIBE_PROVIDER in PROVIDERS:
        cfg = PROVIDERS[TRANSCRIBE_PROVIDER]
        return f"не задан {cfg['env']} для провайдера {cfg['title']}"

    envs = " или ".join(PROVIDERS[n]["env"] for n in _AUTO_ORDER)
    return f"не задан ни один ключ ({envs})"


def setup_hint() -> str:
    """Подсказка, что сделать. Без HTML — вызывающий сам оформит."""
    if TRANSCRIBE_PROVIDER == "off":
        return "Убери TRANSCRIBE_PROVIDER=off из .env, чтобы включить обратно."
    links = ", ".join(f"{PROVIDERS[n]['title']}: {PROVIDERS[n]['keys_url']}"
                      for n in _AUTO_ORDER)
    return f"Добавь ключ в .env любого из провайдеров — {links}"


def looks_like_hallucination(text: str) -> bool:
    return any(p.search(text) for p in _HALLUCINATION_RE)


def transcribe(audio_path: str, filter_hallucinations: bool = True) -> tuple[str, float]:
    """Аудиофайл -> (текст, стоимость в USD).

    Возвращает ('', 0.0) при любой проблеме — вызывающий сам решает,
    что показать пользователю. Стоимость отдают не все провайдеры;
    если её нет в ответе, вернётся 0.0.
    """
    provider, base_url, api_key = resolve()
    if not provider:
        log.warning("Распознавание не настроено: %s", unavailable_reason())
        return "", 0.0

    headers = {"Authorization": f"Bearer {api_key}"}
    if provider == "openrouter":
        headers["X-Title"] = "TikTok Save Bot"

    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{base_url}/audio/transcriptions",
                headers=headers,
                files={"file": (os.path.basename(audio_path), f, "audio/mpeg")},
                data={"model": ASR_MODEL, "language": ASR_LANGUAGE},
                timeout=180,
            )
    except requests.RequestException as e:
        log.error("Распознавание (%s) недоступно: %s", provider, e)
        return "", 0.0

    if response.status_code != 200:
        log.error("Распознавание (%s) HTTP %s: %s",
                  provider, response.status_code, response.text[:200])
        return "", 0.0

    try:
        payload = response.json()
    except ValueError:
        log.error("Распознавание (%s): ответ не JSON", provider)
        return "", 0.0

    cost = float((payload.get("usage") or {}).get("cost") or 0.0)
    text = (payload.get("text") or "").strip()

    # Одно короткое слово почти всегда шум, а не содержание
    if not text or len(text) < 4:
        return "", cost

    if filter_hallucinations and looks_like_hallucination(text):
        log.info("Отброшена галлюцинация Whisper: %r", text[:60])
        return "", cost

    return text, cost
