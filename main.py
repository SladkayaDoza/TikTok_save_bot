import asyncio
import hashlib
import logging
import random
import glob
import os
import html
import re
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
import subprocess
import sys
import time
import requests as http_requests
import aiohttp
from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile, InputMediaVideo, InputMediaPhoto,
    InputMediaAnimation, InputMediaDocument,
    MessageReactionUpdated, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
)
import yt_dlp
import dotenv

dotenv.load_dotenv()

from database import (
    init_db, add_video_record, update_bot_message_id,
    sync_user_reactions, get_chat_stats, find_duplicate,
    get_embedding_details, get_video_id_by_message, retry_failed_jobs,
    get_video_record_by_message, get_message_reaction_count,
    find_cached_media,
)
import embedding
import transcription
from bot_settings import Setting, SettingsStore
from telegram_session import CompatibleTelegramSession

# --- CONFIGURATION ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID"))
DATA_FILE = "active_groups.json"
TEMP_DIR = "temporary_videos"
MEDIA_DIR = "media"

# Распознавание речи живёт в transcription.py: провайдер (DeepInfra / OpenRouter /
# выключено) выбирается там, клиент не создаётся на старте. Бот запускается
# и работает даже без единого ключа — просто без этих функций.

bot = Bot(token=TOKEN, session=CompatibleTelegramSession())
dp = Dispatcher()
active_groups = set()
# Группы, где включён семантический поиск. Включается только админом,
# отдельной командой на каждую группу.
embedding_groups = set()

# Global aiohttp session — reused across all raw API calls (avoids TCP/TLS overhead per request)
_http_session: aiohttp.ClientSession | None = None

# Cached file_id for the 1x1 stub GIF — after first upload, Telegram serves it instantly
_stub_gif_file_id: str | None = None

# Minimal 1x1 transparent GIF (43 bytes)
_STUB_GIF_BYTES = (
    b'GIF89a\x01\x00\x01\x00\x80\x00\x00'
    b'\xff\xff\xff\x00\x00\x00'
    b'!\xf9\x04\x01\x00\x00\x00\x00'
    b',\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
)

async def get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session


async def send_stub_animation(chat_id: int, caption: str, reply_to_message_id: int = None) -> Message:
    """Sends a tiny 1x1 GIF stub. Uses cached file_id after first upload for instant delivery."""
    global _stub_gif_file_id
    kwargs = {
        "chat_id": chat_id,
        "caption": caption,
        "parse_mode": "HTML",
        "disable_notification": True,
    }
    if reply_to_message_id:
        kwargs["reply_to_message_id"] = reply_to_message_id

    if _stub_gif_file_id:
        # Instant — Telegram already has the file, no upload needed
        return await bot.send_animation(animation=_stub_gif_file_id, **kwargs)
    else:
        # First time — upload the tiny GIF and cache its file_id
        from aiogram.types import BufferedInputFile
        gif_file = BufferedInputFile(_STUB_GIF_BYTES, filename="stub.gif")
        msg = await bot.send_animation(animation=gif_file, **kwargs)
        if msg.animation:
            _stub_gif_file_id = msg.animation.file_id
        elif msg.document:
            _stub_gif_file_id = msg.document.file_id
        return msg

# =====================================================================
#  TELEGRAM RICH MESSAGES HELPERS
# =====================================================================

def get_domain_name(url: str) -> str:
    """Extracts a clean domain name from a URL."""
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1) if match else "Ссылка"

def to_fallback_html(html: str) -> str:
    """Converts Rich HTML elements to basic Telegram HTML elements."""
    res = html
    # p -> newlines
    res = re.sub(r'<p[^>]*>', '', res)
    res = re.sub(r'</p>', '\n', res)
    # br -> newline
    res = re.sub(r'<br\s*/?>', '\n', res)
    # h1-h6 -> bold with newlines
    res = re.sub(r'<h[1-6][^>]*>', '\n<b>', res)
    res = re.sub(r'</h[1-6]>', '</b>\n', res)
    # hr -> divider line
    res = re.sub(r'<hr\s*/?>', '\n━───────────────━\n', res)
    # details / summary
    res = re.sub(r'<details[^>]*>', '', res)
    res = re.sub(r'</details>', '', res)
    res = re.sub(r'<summary[^>]*>', '<b>🔽 ', res)
    res = re.sub(r'</summary>', '</b>\n', res)
    # lists
    res = re.sub(r'<ul[^>]*>', '', res)
    res = re.sub(r'</ul>', '', res)
    res = re.sub(r'<ol[^>]*>', '', res)
    res = re.sub(r'</ol>', '', res)
    res = re.sub(r'<li[^>]*>', '• ', res)
    res = re.sub(r'</li>', '\n', res)
    # marks
    res = re.sub(r'<mark[^>]*>', '<u>', res)
    res = re.sub(r'</mark>', '</u>', res)
    # sub / sup
    res = re.sub(r'<sub[^>]*>', '(', res)
    res = re.sub(r'</sub>', ')', res)
    res = re.sub(r'<sup[^>]*>', '^', res)
    res = re.sub(r'</sup>', '', res)
    # tables
    res = re.sub(r'</?table[^>]*>', '', res)
    res = re.sub(r'<tr[^>]*>', '', res)
    res = re.sub(r'</tr>', '\n', res)
    res = re.sub(r'<th[^>]*>', ' | <b>', res)
    res = re.sub(r'</th>', '</b>', res)
    res = re.sub(r'<td[^>]*>', ' | ', res)
    res = re.sub(r'</td>', '', res)
    # Clean up double/multiple newlines
    res = re.sub(r'\n{3,}', '\n\n', res)
    return res.strip()

async def send_rich_message(chat_id: int, html: str, reply_to_message_id: int = None) -> dict:
    """Sends a Telegram Rich Message with fallback to standard sendMessage."""
    session = await get_http_session()
    url = f"https://api.telegram.org/bot{TOKEN}/sendRichMessage"
    payload = {
        "chat_id": chat_id,
        "rich_message": {"html": html}
    }
    if reply_to_message_id:
        payload["reply_parameters"] = {"message_id": reply_to_message_id}

    try:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            if result.get("ok"):
                return result
            logging.warning(f"sendRichMessage failed: {result.get('description')}. Falling back.")
    except Exception as e:
        logging.error(f"Error in sendRichMessage HTTP request: {e}. Falling back.")

    # Fallback to standard sendMessage
    fallback_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    fallback_html = to_fallback_html(html)
    fallback_payload = {
        "chat_id": chat_id,
        "text": fallback_html,
        "parse_mode": "HTML"
    }
    if reply_to_message_id:
        fallback_payload["reply_parameters"] = {"message_id": reply_to_message_id}

    try:
        async with session.post(fallback_url, json=fallback_payload) as resp:
            return await resp.json()
    except Exception as e:
        logging.error(f"Fallback sendMessage failed: {e}")
        return {"ok": False, "description": str(e)}

async def edit_rich_message(chat_id: int, message_id: int, html: str) -> dict:
    """Edits a Telegram Rich Message with fallback to standard editMessageText."""
    session = await get_http_session()
    url = f"https://api.telegram.org/bot{TOKEN}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "rich_message": {"html": html}
    }

    try:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            if result.get("ok"):
                return result
            logging.warning(f"editMessageText (Rich) failed: {result.get('description')}. Falling back.")
    except Exception as e:
        logging.error(f"Error in editMessageText HTTP request: {e}. Falling back.")

    # Fallback to standard editMessageText
    fallback_url = f"https://api.telegram.org/bot{TOKEN}/editMessageText"
    fallback_html = to_fallback_html(html)
    fallback_payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": fallback_html,
        "parse_mode": "HTML"
    }

    try:
        async with session.post(fallback_url, json=fallback_payload) as resp:
            return await resp.json()
    except Exception as e:
        logging.error(f"Fallback editMessageText failed: {e}")
        return {"ok": False, "description": str(e)}

def load_data():
    global active_groups, embedding_groups
    if not os.path.exists(DATA_FILE):
        return
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Старый формат — плоский список активных групп.
            # Читаем как есть, при первом save_data() файл переедет на новый.
            active_groups = set(data)
            embedding_groups = set()
            logging.info(f"Загружен старый формат {DATA_FILE}: {len(active_groups)} групп")
        else:
            active_groups = set(data.get("active_groups", []))
            embedding_groups = set(data.get("embedding_groups", []))
    except Exception as e:
        logging.error(f"Failed to load active groups: {e}")

def save_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "active_groups": list(active_groups),
                "embedding_groups": list(embedding_groups),
            }, f)
    except Exception as e:
        logging.error(f"Failed to save active groups: {e}")

# Регулярное выражение для поиска ссылок YouTube (в т.ч. Shorts), TikTok, Instagram и X (Twitter)
URL_REGEX = re.compile(
    r'(https?://(?:www\.)?(?:'
    r'(?:youtube\.com|youtu\.be|tiktok\.com|vm\.tiktok\.com|vt\.tiktok\.com|instagram\.com)[^\s]+'
    r'|(?:mobile\.)?(?:x|twitter)\.com/[^\s]+'
    r'))'
)

TIKTOK_URL_REGEX = re.compile(
    r'https?://(?:www\.)?(?:tiktok\.com|vm\.tiktok\.com|vt\.tiktok\.com)'
)

# Твит: x.com/<user>/status/<id> (или /i/status/<id>), а также twitter.com
TWITTER_STATUS_REGEX = re.compile(
    r'https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/(?:[^/\s]+/)?status(?:es)?/(\d+)'
)

# Лимит Bot API на загрузку файла — 50 МБ, оставляем запас
MAX_UPLOAD_BYTES = 45 * 1024 * 1024


# =====================================================================
#  DEDUPLICATION & TIKTOK SPEEDUP HELPERS
# =====================================================================

@dataclass
class TikTokContent:
    """Результат анализа TikTok URL через tikwm API."""
    content_type: str = "unknown"         # "video" | "photos" | "unknown"
    video_url: str | None = None          # Прямая ссылка на видео (для скачивания)
    photo_paths: list[str] = field(default_factory=list)  # Пути к скачанным фото
    direct_url: str | None = None         # Прямая ссылка для сохранения в БД
    video_source_id: str | None = None    # "tiktok:7234567890"
    photo_total: int = 0                  # Сколько фото в посте (для «скачано N из M»)
    video_file: str | None = None         # Видео уже скачано сюда (свой парсер TikTok)
    via: str = ""                         # Кто разобрал: tiktok_page | tikwm | fxtwitter


def extract_video_source_id(url: str) -> str | None:
    """Извлекает уникальный ID видео из URL (без HTTP-запросов).
    Для коротких ссылок (vm.tiktok.com) возвращает None — ID будет
    получен позже из tikwm API."""

    # TikTok: /video/1234567890 or /photo/1234567890
    m = re.search(r'tiktok\.com/.*/(?:video|photo)/(\d+)', url)
    if m:
        return f"tiktok:{m.group(1)}"

    # YouTube: ?v=xxx or youtu.be/xxx or /shorts/xxx
    m = re.search(r'(?:youtube\.com/.*[?&]v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})', url)
    if m:
        return f"youtube:{m.group(1)}"

    # Instagram: /reel/xxx/ or /p/xxx/
    m = re.search(r'instagram\.com/(?:reel|p)/([\w-]+)', url)
    if m:
        return f"instagram:{m.group(1)}"

    # X / Twitter: /status/1234567890 — один и тот же твит на x.com и twitter.com
    m = TWITTER_STATUS_REGEX.search(url)
    if m:
        return f"twitter:{m.group(1)}"

    return None


def get_message_link(chat_id: int, message_id: int,
                     chat_username: str | None = None) -> str | None:
    """Constructs a Telegram deep link to a specific message in a group.

    Публичная группа (есть username) — t.me/<username>/<id>, работает для всех.
    Приватная супергруппа — t.me/c/<id>/<msg>, работает для участников.
    Обычная (не супер-) группа ссылок на сообщения не поддерживает.
    """
    if not message_id:
        return None
    if chat_username:
        return f"https://t.me/{chat_username}/{message_id}"
    if chat_id < 0 and str(chat_id).startswith("-100"):
        # Supergroup: chat_id is -100XXXXXXXXXX → need XXXXXXXXXX
        clean_id = str(abs(chat_id))[3:]
        return f"https://t.me/c/{clean_id}/{message_id}"
    return None


_DL_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
# CDN TikTok/X режет скорость на одно соединение (~1-2 МБ/с), поэтому
# файл качаем кусками в несколько параллельных Range-запросов
_DL_PART_SIZE = 1024 * 1024
_DL_MAX_PARALLEL = 8


async def _download_single_stream(session, url: str, output_path: str,
                                  headers: dict | None = None) -> bool:
    async with session.get(url, headers=headers or _DL_HEADERS) as resp:
        if resp.status != 200:
            logging.warning(f"Direct download failed: HTTP {resp.status}")
            return False
        with open(output_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(65536):
                f.write(chunk)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 1000


async def download_video_direct(url: str, output_path: str,
                                session: aiohttp.ClientSession | None = None,
                                headers: dict | None = None) -> bool:
    """Скачивает видео напрямую через aiohttp (без yt-dlp).
    Используется, когда API (tikwm / fxtwitter) дал прямую ссылку.
    Если сервер поддерживает Range — качает в несколько потоков.
    session/headers — для CDN, которому нужны куки страницы (TikTok)."""
    session = session or await get_http_session()
    base_headers = headers or _DL_HEADERS
    timeout = aiohttp.ClientTimeout(total=120, sock_read=30)
    try:
        # Первый кусок заодно сообщает полный размер (Content-Range: bytes 0-N/TOTAL)
        first_end = _DL_PART_SIZE - 1
        async with session.get(url, headers={**base_headers, "Range": f"bytes=0-{first_end}"},
                               timeout=timeout) as resp:
            if resp.status == 200:
                # Range не поддерживается — сервер отдаёт файл целиком, просто дочитываем
                with open(output_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        f.write(chunk)
                return os.path.getsize(output_path) > 1000
            if resp.status != 206:
                logging.warning(f"Direct download failed: HTTP {resp.status}")
                return False
            total = int((resp.headers.get("Content-Range") or "").rsplit("/", 1)[-1] or 0)
            first_part = await resp.read()

        if total <= 0:
            return await _download_single_stream(session, url, output_path, base_headers)

        with open(output_path, "wb") as f:
            f.truncate(total)
            f.write(first_part)

        ranges = [(start, min(start + _DL_PART_SIZE, total) - 1)
                  for start in range(len(first_part), total, _DL_PART_SIZE)]
        sem = asyncio.Semaphore(_DL_MAX_PARALLEL)

        async def fetch(start: int, end: int):
            async with sem:
                for attempt in range(3):
                    try:
                        async with session.get(
                            url, headers={**base_headers, "Range": f"bytes={start}-{end}"},
                            timeout=timeout,
                        ) as r:
                            if r.status != 206:
                                raise RuntimeError(f"HTTP {r.status}")
                            data = await r.read()
                        if len(data) != end - start + 1:
                            raise RuntimeError(f"короткий кусок {len(data)}")
                        with open(output_path, "r+b") as f:
                            f.seek(start)
                            f.write(data)
                        return
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(0.5 * (attempt + 1))

        await asyncio.gather(*(fetch(a, b) for a, b in ranges))
        return os.path.getsize(output_path) == total and total > 1000
    except Exception as e:
        logging.error(f"Direct download error: {e}")
        return False


# =====================================================================
#  WHISPER TRANSCRIPTION HELPERS
# =====================================================================

def extract_audio_to_mp3(input_path: str, output_path: str) -> bool:
    """Конвертирует любой медиа-файл в mp3 через ffmpeg.
    Возвращает True при успехе, False при ошибке."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vn",                # убрать видеодорожку
                "-acodec", "libmp3lame",
                "-ar", "16000",       # 16 kHz — оптимально для Whisper
                "-ac", "1",           # моно
                "-b:a", "64k",
                output_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logging.error(f"ffmpeg error: {result.stderr.decode(errors='replace')[:500]}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except FileNotFoundError:
        logging.error("ffmpeg is not installed or not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logging.error("ffmpeg conversion timed out")
        return False
    except Exception as e:
        logging.error(f"ffmpeg unexpected error: {e}")
        return False


def transcribe_audio(mp3_path: str) -> str | None:
    """Распознаёт речь через выбранного провайдера (см. transcription.py).
    Возвращает текст или None, если речи нет либо провайдер недоступен."""
    text, _cost = transcription.transcribe(mp3_path)
    return text or None


def get_media_from_message(message: Message) -> tuple[str, str] | None:
    """Извлекает file_id и расширение из сообщения с медиа.
    Поддерживает: voice, audio, video, video_note, document (audio/video MIME).
    Возвращает (file_id, extension) или None."""

    if message.voice:
        return (message.voice.file_id, ".ogg")

    if message.audio:
        return (message.audio.file_id, ".mp3")

    if message.video:
        return (message.video.file_id, ".mp4")

    if message.video_note:
        return (message.video_note.file_id, ".mp4")

    if message.document:
        mime = (message.document.mime_type or "").lower()
        if mime.startswith("audio/") or mime.startswith("video/"):
            # Определяем расширение из MIME или имени файла
            ext = ".bin"
            if message.document.file_name:
                _, ext = os.path.splitext(message.document.file_name)
                if not ext:
                    ext = ".bin"
            elif "mp3" in mime:
                ext = ".mp3"
            elif "mp4" in mime:
                ext = ".mp4"
            elif "ogg" in mime:
                ext = ".ogg"
            elif "wav" in mime:
                ext = ".wav"
            elif "webm" in mime:
                ext = ".webm"
            elif "flac" in mime:
                ext = ".flac"
            elif "m4a" in mime:
                ext = ".m4a"
            return (message.document.file_id, ext)

    return None


# =====================================================================
#  TIKTOK CONTENT ANALYSIS (фото/видео определение + прямые ссылки)
# =====================================================================

_IMG_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


async def _download_one_image(session: aiohttp.ClientSession, img_url: str,
                              index: int, source_name: str, attempts: int = 3) -> str | None:
    """Скачивает одно изображение с повторами. Возвращает путь или None."""
    timeout = aiohttp.ClientTimeout(total=20)
    for attempt in range(1, attempts + 1):
        try:
            async with session.get(img_url, headers=_IMG_HEADERS, timeout=timeout) as img_resp:
                if img_resp.status != 200:
                    raise RuntimeError(f"HTTP {img_resp.status}")
                content = await img_resp.read()
                if not content or len(content) < 100:
                    raise RuntimeError("пустой ответ")
                ct = img_resp.headers.get("Content-Type", "")
                ext = ".jpg"
                if "png" in ct:
                    ext = ".png"
                elif "webp" in ct:
                    ext = ".webp"
                photo_path = os.path.join(TEMP_DIR, f"photo_{uuid.uuid4()}{ext}")
                with open(photo_path, "wb") as f:
                    f.write(content)
                return photo_path
        except Exception as e:
            logging.warning(f"Photo {index} via {source_name}, попытка {attempt}/{attempts}: {e}")
            if attempt < attempts:
                await asyncio.sleep(attempt)
    return None


async def _download_image_list(image_urls: list[str], source_name: str,
                               session: aiohttp.ClientSession | None = None) -> list[str]:
    """Скачивает ВСЕ изображения параллельно (порядок сохраняется), с повторами.
    Возвращает пути к успешно скачанным файлам."""
    session = session or await get_http_session()
    results = await asyncio.gather(*(
        _download_one_image(session, u, i, source_name)
        for i, u in enumerate(image_urls)
    ))
    downloaded = [p for p in results if p]
    if len(downloaded) < len(image_urls):
        logging.error(f"{source_name}: скачано {len(downloaded)} из {len(image_urls)} фото")
    return downloaded


# tikwm бесплатно пускает ~1 запрос в секунду — несколько ссылок подряд
# выстраиваем в очередь, а не получаем отказ
_tikwm_lock = asyncio.Lock()
_tikwm_last_call = 0.0


async def _tikwm_request(url: str, attempts: int = 3) -> dict | None:
    """Запрос к tikwm с повторами. Отвечает он за 3-5 секунд, поэтому таймаут щедрый."""
    global _tikwm_last_call
    session = await get_http_session()
    timeout = aiohttp.ClientTimeout(total=15)
    loop = asyncio.get_running_loop()
    for attempt in range(1, attempts + 1):
        try:
            async with _tikwm_lock:
                wait = 1.1 - (loop.time() - _tikwm_last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    async with session.post(
                        "https://www.tikwm.com/api/", data={"url": url, "hd": 1},
                        headers=_IMG_HEADERS, timeout=timeout,
                    ) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        api_data = await resp.json(content_type=None)
                finally:
                    _tikwm_last_call = loop.time()
            if api_data.get("code") == 0 and api_data.get("data"):
                return api_data["data"]
            raise RuntimeError(f"code={api_data.get('code')} msg={api_data.get('msg')}")
        except Exception as e:
            kind = "timeout" if isinstance(e, asyncio.TimeoutError) else str(e)
            logging.warning(f"tikwm, попытка {attempt}/{attempts} для {url}: {kind}")
            if attempt < attempts:
                await asyncio.sleep(attempt)
    return None


async def resolve_tiktok_short_url(url: str) -> str:
    """vt./vm.tiktok.com → полная ссылка с ID (по редиректу, без скачивания страницы)."""
    session = await get_http_session()
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with session.get(url, headers=_IMG_HEADERS, allow_redirects=False,
                               timeout=timeout) as resp:
            location = resp.headers.get("Location")
            if location and extract_video_source_id(location):
                # Хвост ?_r=1&_t=... не нужен ни tikwm, ни нам
                return location.split("?")[0]
    except Exception as e:
        logging.debug(f"Не удалось раскрыть {url}: {e}")
    return url


# Свой парсер страницы TikTok вместо tikwm (tikwm остаётся запасным).
# 0 — сразу идти в tikwm, как раньше.
TIKTOK_SCRAPER = os.getenv("TIKTOK_SCRAPER", "1") != "0"

# Интервал автообновления yt-dlp в часах, 0 — выключено (см. ytdlp_auto_updater)
YTDLP_UPDATE_HOURS = float(os.getenv("YTDLP_UPDATE_HOURS", "24"))

_TT_PAGE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
_TT_DATA_RE = re.compile(
    r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', re.S
)


def _pick_tiktok_bitrate(video: dict) -> str | None:
    """Самый качественный вариант видео, который влезает в лимит Telegram."""
    variants = []
    for b in video.get("bitrateInfo") or []:
        addr = b.get("PlayAddr") or {}
        urls = addr.get("UrlList") or []
        if not urls:
            continue
        size = int(addr.get("DataSize") or 0)
        variants.append((int(b.get("Bitrate") or 0), size, urls[0]))
    variants.sort(reverse=True)
    for _bitrate, size, u in variants:
        if 0 < size < MAX_UPLOAD_BYTES:
            return u
    if variants:
        return variants[-1][2]
    return video.get("playAddr") or None


async def scrape_tiktok_content(tiktok_id: str, output_path: str) -> TikTokContent | None:
    """Свой разбор страницы TikTok (~0.7 с против 2-4 с у tikwm).

    Данные лежат JSON-ом прямо в HTML. CDN отдаёт видео только с куками
    этой же сессии, поэтому видео сразу и качаем — той же сессией.
    Любая неожиданность → None, и работает запасной tikwm.
    """
    result = TikTokContent(video_source_id=f"tiktok:{tiktok_id}", via="tiktok_page")
    # Отдельная сессия: свои куки на каждый ролик, не смешиваются с остальным ботом
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        try:
            t0 = time.monotonic()
            # /video/ работает и для фото-постов; имя автора TikTok не проверяет
            page_url = f"https://www.tiktok.com/@_/video/{tiktok_id}"
            timeout = aiohttp.ClientTimeout(total=6)
            async with session.get(page_url, headers=_TT_PAGE_HEADERS, timeout=timeout) as resp:
                if resp.status != 200:
                    logging.warning(f"TikTok page HTTP {resp.status} for {tiktok_id}")
                    return None
                page = await resp.text()
            m = _TT_DATA_RE.search(page)
            if not m:
                logging.warning(f"TikTok page без данных для {tiktok_id} (капча/блок?)")
                return None
            detail = json.loads(m.group(1)).get("__DEFAULT_SCOPE__", {}).get("webapp.video-detail") or {}
            if detail.get("statusCode") not in (0, None):
                logging.warning(f"TikTok page statusCode={detail.get('statusCode')} for {tiktok_id}")
                return None
            item = (detail.get("itemInfo") or {}).get("itemStruct") or {}
            if not item:
                return None

            images = ((item.get("imagePost") or {}).get("images")) or []
            image_urls = [
                (im.get("imageURL") or {}).get("urlList", [None])[0] for im in images
            ]
            image_urls = [u for u in image_urls if u]
            if image_urls:
                result.content_type = "photos"
                result.photo_total = len(image_urls)
                result.direct_url = image_urls[0]
                result.photo_paths = await _download_image_list(image_urls, "tiktok-page", session)
                if not result.photo_paths:
                    return None
                logging.info(f"TikTok slideshow via page: {len(result.photo_paths)}/{len(image_urls)} photos")
                return result

            t_page = time.monotonic() - t0
            video_url = _pick_tiktok_bitrate(item.get("video") or {})
            if not video_url:
                return None
            headers = {**_TT_PAGE_HEADERS, "Referer": "https://www.tiktok.com/"}
            if not await download_video_direct(video_url, output_path, session=session, headers=headers):
                logging.warning(f"TikTok page: видео не скачалось для {tiktok_id}")
                return None
            result.content_type = "video"
            result.video_file = output_path
            result.direct_url = video_url
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            t_dl = time.monotonic() - t0 - t_page
            logging.info(f"TikTok video via page: page {t_page:.2f}s, "
                         f"download {t_dl:.2f}s ({size_mb:.1f} MB)")
            return result
        except Exception as e:
            logging.warning(f"TikTok page parse failed for {tiktok_id}: {e}")
            return None


async def analyze_tiktok_content(url: str, output_path: str | None = None) -> TikTokContent:
    """Анализирует TikTok URL через tikwm API.
    Определяет тип контента (фото/видео), извлекает прямые ссылки.
    Для фото — скачивает файлы. Для видео — возвращает direct URL.
    НЕ использует yt-dlp для анализа — только лёгкие HTTP API."""

    result = TikTokContent()

    if not TIKTOK_URL_REGEX.match(url):
        return result

    # Сначала свой парсер страницы — он в 3 раза быстрее tikwm
    m = re.search(r'tiktok\.com/.*/(?:video|photo)/(\d+)', url)
    if m and output_path and TIKTOK_SCRAPER:
        scraped = await scrape_tiktok_content(m.group(1), output_path)
        if scraped:
            return scraped
        # Наполовину скачанное не оставляем
        if os.path.exists(output_path):
            os.remove(output_path)

    result.via = "tikwm"
    data = await _tikwm_request(url)
    if not data:
        logging.error(f"tikwm не ответил для {url}, пробуем yt-dlp")
        return result

    # Извлекаем video_source_id из API (работает даже для vm.tiktok.com)
    tiktok_id = data.get("id")
    if tiktok_id:
        result.video_source_id = f"tiktok:{tiktok_id}"

    images = data.get("images") or []
    real_images = [img for img in images if isinstance(img, str) and img.startswith("http")]

    if real_images:
        # ===== ФОТО-ПОСТ =====
        result.content_type = "photos"
        result.photo_total = len(real_images)
        result.direct_url = real_images[0]
        result.photo_paths = await _download_image_list(real_images, "tikwm")
        logging.info(f"TikTok slideshow via tikwm: {len(result.photo_paths)}/{len(real_images)} photos")
        return result

    # ===== ВИДЕО-ПОСТ =====
    result.content_type = "video"
    video_url = data.get("hdplay") or data.get("play")
    if video_url and video_url.startswith("http"):
        result.video_url = video_url
        result.direct_url = data.get("play") or video_url
        logging.info("TikTok video direct URL via tikwm (fast path)")
    else:
        logging.debug("tikwm: video post, no play URL — yt-dlp fallback")
    return result


def _pick_twitter_mp4(video: dict) -> str | None:
    """Лучшее mp4 из вариантов твита, которое влезает в лимит Telegram."""
    duration = float(video.get("duration") or 0)
    mp4s = [
        f for f in (video.get("formats") or video.get("variants") or [])
        if isinstance(f, dict) and f.get("url")
        and (f.get("container") == "mp4" or ".mp4" in f["url"])
    ]
    if not mp4s:
        url = video.get("url")
        return url if url and ".mp4" in url else None

    mp4s.sort(key=lambda f: f.get("bitrate") or 0, reverse=True)
    if duration > 0:
        for f in mp4s:
            est_bytes = (f.get("bitrate") or 0) / 8 * duration
            if est_bytes < MAX_UPLOAD_BYTES:
                return f["url"]
        return mp4s[-1]["url"]  # всё большое — берём самое лёгкое
    return mp4s[0]["url"]


async def analyze_twitter_content(url: str) -> TikTokContent:
    """Анализирует твит через публичный API fxtwitter (без авторизации).
    Видео → прямая ссылка на mp4, только фото → скачанные файлы.
    При любой ошибке возвращает "unknown" — тогда скачает yt-dlp."""
    result = TikTokContent()

    m = TWITTER_STATUS_REGEX.search(url)
    if not m:
        return result
    tweet_id = m.group(1)
    result.video_source_id = f"twitter:{tweet_id}"

    session = await get_http_session()
    try:
        timeout = aiohttp.ClientTimeout(total=6)
        headers = {"User-Agent": "TikTokSaveBot/1.0"}
        async with session.get(f"https://api.fxtwitter.com/status/{tweet_id}",
                               headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                logging.warning(f"fxtwitter API HTTP {resp.status} for {url}")
                return result
            data = await resp.json(content_type=None)
    except asyncio.TimeoutError:
        logging.warning(f"fxtwitter API timeout for {url}")
        return result
    except Exception as e:
        logging.debug(f"fxtwitter API failed for {url}: {e}")
        return result

    media = ((data or {}).get("tweet") or {}).get("media") or {}
    videos = media.get("videos") or []
    photos = media.get("photos") or []

    if videos:
        video_url = _pick_twitter_mp4(videos[0])
        result.content_type = "video"
        if video_url:
            result.video_url = video_url
            result.direct_url = video_url
            logging.info("Twitter video direct URL via fxtwitter (fast path)")
        return result

    if photos:
        image_urls = [p["url"] for p in photos if isinstance(p, dict) and p.get("url")]
        downloaded = await _download_image_list(image_urls, "fxtwitter")
        if downloaded:
            result.content_type = "photos"
            result.photo_paths = downloaded
            result.photo_total = len(image_urls)
            result.direct_url = image_urls[0]
            logging.info(f"Twitter photos via fxtwitter: {len(downloaded)} photos")
    return result


# =====================================================================
#  BOT HANDLERS
# =====================================================================

@dp.message(Command("setup"))
async def cmd_setup(message: Message):
    if message.from_user.id != ADMIN_ID:
        return
    active_groups.add(message.chat.id)
    save_data()
    html = (
        "<h2>⚙️ Настройка системы</h2>\n"
        "<p>Робот-помощник успешно <b>активирован</b> в данном чате!</p>\n"
        "<hr/>\n"
        "<p>Я буду автоматически перехватывать ссылки из следующих источников и отправлять медиафайлы напрямую:</p>\n"
        "<ul>\n"
        "  <li><b>TikTok</b> (видео и слайдшоу из фото)</li>\n"
        "  <li><b>YouTube</b> (видеоролики и Shorts)</li>\n"
        "  <li><b>Instagram</b> (Reels и публикации)</li>\n"
        "  <li><b>X / Twitter</b> (видео и фото из твитов)</li>\n"
        "</ul>\n"
        "<p>Используйте команду <code>/disable</code> для отключения автоматической загрузки.</p>"
    )
    await send_rich_message(message.chat.id, html, reply_to_message_id=message.message_id)

@dp.message(Command("disable"))
async def cmd_disable(message: Message):
    if message.from_user.id != ADMIN_ID:
        return
    if message.chat.id in active_groups:
        active_groups.remove(message.chat.id)
        save_data()
        html = (
            "<h2>⚙️ Настройка системы</h2>\n"
            "<p>Автоматическая загрузка медиафайлов <b>деактивирована</b>.</p>\n"
            "<hr/>\n"
            "<p>Я больше не буду обрабатывать входящие ссылки в этом чате. Вы можете активировать меня снова с помощью команды <code>/setup</code>.</p>"
        )
    else:
        html = (
            "<h2>⚠️ Обратите внимание</h2>\n"
            "<p>Бот <b>не был настроен</b> в этом чате.</p>\n"
            "<hr/>\n"
            "<p>Для активации автоматической загрузки отправьте команду <code>/setup</code>.</p>"
        )
    await send_rich_message(message.chat.id, html, reply_to_message_id=message.message_id)

def download_video(url: str, output_path: str, max_retries: int = 3):
    """Скачивает видео с retry-логикой для обхода временных ошибок TikTok."""
    import time
    cookies_path = os.path.abspath("cookies.txt")
    
    ydl_opts = {
        'outtmpl': output_path,
        # Более гибкий выбор формата с ограничением ~50МБ
        'max_filesize': 50 * 1024 * 1024,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        # В твите (и плейлисте) может быть несколько видео — берём первое,
        # иначе они перезапишут друг друга в одном outtmpl
        'playlist_items': '1',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        # YouTube требует JS-рантайм для решения n-challenge (иначе форматы
        # отбрасываются и падаем в "Requested format is not available").
        # По умолчанию yt-dlp ищет только deno — разрешаем ещё node/bun.
        'js_runtimes': {'deno': {}, 'node': {}, 'bun': {}},
        # android_vr не поддерживает cookies и молча пропускается, когда задан
        # cookiefile, поэтому держим cookie-совместимые клиенты первыми.
        'extractor_args': {'youtube': {'player_client': ['default', 'android_vr']}},
        # Фрагментированные потоки (HLS/DASH: X, YouTube) качаем параллельно
        'concurrent_fragment_downloads': 8,
        # Перепаковка в mp4 без перекодирования: секунды вместо минут.
        # Формат выше и так предпочитает mp4/m4a, перекодировать нечего.
        'postprocessors': [{
            'key': 'FFmpegVideoRemuxer',
            'preferedformat': 'mp4',
        }],
    }

    # Подключаем куки, если файл существует (нужны для приватного/18+ контента)
    if os.path.exists(cookies_path):
        ydl_opts['cookiefile'] = cookies_path
    last_error = None
    tried_without_cookies = False
    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return  # Успех
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            err_msg = str(e).lower()

            # Протухшие/разлогиненные куки ломают запрос (Instagram отвечает
            # HTTP 400 на "Downloading video info"), хотя без них публичный
            # контент качается. Один раз пробуем без кук.
            if 'cookiefile' in ydl_opts and not tried_without_cookies:
                tried_without_cookies = True
                logging.warning(f"Download with cookies failed ({e}). Retrying without cookies...")
                ydl_opts.pop('cookiefile')
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    logging.warning("Downloaded without cookies — cookies.txt, похоже, протух, обнови его")
                    return
                except yt_dlp.utils.DownloadError as e2:
                    last_error = e2
                    # Без кук тоже не вышло — возвращаем их и идём по обычной логике
                    ydl_opts['cookiefile'] = cookies_path

            # Retry только на временные ошибки (rehydration, rate limit, etc.)
            is_retryable = any(kw in err_msg for kw in [
                'rehydration', 'unable to extract', 'http error 429',
                'http error 503', 'connection reset', 'timed out',
            ])
            if is_retryable and attempt < max_retries:
                logging.warning(f"Download attempt {attempt}/{max_retries} failed: {e}. Retrying in {attempt * 2}s...")
                time.sleep(attempt * 2)
                continue
            raise  # Не-временная ошибка или исчерпаны попытки
    raise last_error  # На случай если цикл завершился без raise

async def _update_status(chat_id: int, message_id: int, html_text: str):
    """Хелпер для обновления caption статусного сообщения (stub GIF)."""
    try:
        fallback_caption = to_fallback_html(html_text)
        await bot.edit_message_caption(
            caption=fallback_caption,
            chat_id=chat_id,
            message_id=message_id,
            parse_mode="HTML"
        )
    except Exception as e:
        logging.error(f"Failed to update status message: {e}")


def _extract_file_info(result) -> tuple[str | None, str | None]:
    """(file_id, тип) из ответа edit_message_media — тип нужен, чтобы потом
    переотправить медиа по file_id правильным InputMedia*."""
    if not isinstance(result, Message):
        return None, None
    if result.video:
        return result.video.file_id, "video"
    if result.animation:
        return result.animation.file_id, "animation"
    if result.photo:
        return result.photo[-1].file_id, "photo"
    if result.document:
        return result.document.file_id, "document"
    return None, None


def _extract_file_id(result) -> str | None:
    """file_id из ответа edit_message_media.

    Telegram возвращает обновлённое сообщение для не-inline правок. Сохранённый
    file_id позволяет позже перекачать медиа из Telegram, не трогая исходную
    ссылку — она к тому времени может умереть.
    """
    if not isinstance(result, Message):
        return None
    if result.video:
        return result.video.file_id
    if result.animation:
        return result.animation.file_id
    if result.photo:
        return result.photo[-1].file_id
    if result.document:
        return result.document.file_id
    return None


async def _after_send(chat_id: int, rec_id: int, sent_message_id: int | None,
                      media_path: str | None, kind: str = "video",
                      tg_file_id: str | None = None,
                      tg_file_type: str | None = None):
    """Общий хвост после успешной отправки медиа.

    Вызывается ПОСЛЕ того, как видео уже улетело в чат, поэтому на скорость
    доставки не влияет. Файл переносится в очередь и обрабатывается фоном.
    """
    if rec_id <= 0:
        return

    if sent_message_id:
        update_bot_message_id(rec_id, sent_message_id, tg_file_id, tg_file_type)

    if chat_id in embedding_groups and media_path:
        try:
            await embedding.enqueue_media(rec_id, chat_id, media_path, kind)
        except Exception as e:
            logging.error(f"Не удалось поставить видео {rec_id} в очередь: {e}")


async def process_url(message: Message, url: str):
    # Замер по этапам: одна строка в логе на каждую ссылку (см. finally)
    t_start = time.monotonic()
    stages: list[tuple[str, float]] = []
    last_mark = [t_start]

    def mark(name: str):
        now = time.monotonic()
        stages.append((name, now - last_mark[0]))
        last_mark[0] = now

    # Сколько апдейт шёл от Telegram до бота (date — с точностью до секунды)
    try:
        tg_lag = (datetime.now(timezone.utc) - message.date).total_seconds()
    except Exception:
        tg_lag = None

    original_url = url
    video_id = str(uuid.uuid4())
    output_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")

    # Определяем имя пользователя для статистики
    user = message.from_user
    if user:
        display_name = f"@{user.username}" if user.username else user.full_name
    else:
        display_name = "Unknown"

    is_tiktok = bool(TIKTOK_URL_REGEX.match(url))
    is_twitter = bool(TWITTER_STATUS_REGEX.match(url))
    # Короткая ссылка vt./vm.tiktok.com: раскрываем сами (~0.3 с) — так ID
    # известен сразу (дубликат и кэш срабатывают до tikwm), а tikwm
    # по полной ссылке отвечает быстрее.
    # Заглушку отправляем сразу, параллельно с раскрытием ссылки и БД
    stub_html = (
        "<h3>📥 Загрузка медиа</h3>\n"
        "<p>Статус: 🟡 <b>Скачивание...</b></p>\n"
        f"<p>Источник: <a href=\"{html.escape(url)}\">{get_domain_name(url)}</a></p>"
    )
    stub_task = asyncio.create_task(send_stub_animation(
        chat_id=message.chat.id,
        caption=to_fallback_html(stub_html),
        reply_to_message_id=message.message_id,
    ))

    if is_tiktok and not extract_video_source_id(url):
        url = await resolve_tiktok_short_url(url)
        mark("resolve")
    elif is_tiktok:
        # Хвост ?_r=1&u_code=...&share_link_id=... — трекинг шаринга, он не нужен
        url = url.split("?")[0]

    domain = get_domain_name(url)
    escaped_url = html.escape(url)

    # ============================================================
    #  ДЕДУПЛИКАЦИЯ: Проверяем video_source_id до скачивания
    # ============================================================
    video_source_id = extract_video_source_id(url)
    is_duplicate = False
    duplicate_info = None
    if video_source_id:
        duplicate_info = find_duplicate(message.chat.id, video_source_id)
        if duplicate_info:
            is_duplicate = True

    mark("db")

    # Заглушка (1x1 GIF, мгновенно после первой загрузки) — одно сообщение без лишнего уведомления
    try:
        status_msg = await stub_task
    except Exception:
        logging.info(f"⏱ {original_url}: заглушка не отправилась, tg_lag {tg_lag}")
        raise
    status_msg_id = status_msg.message_id
    mark("stub")

    # Обновления статуса шлём фоном — работа не ждёт ответа Telegram.
    # Перед заменой медиа дожидаемся их, иначе запоздавший статус
    # перезапишет подпись готового видео.
    status_tasks: list[asyncio.Task] = []

    def _bg_status(html_text: str):
        status_tasks.append(asyncio.create_task(
            _update_status(message.chat.id, status_msg_id, html_text)
        ))

    async def _flush_status():
        if status_tasks:
            await asyncio.gather(*status_tasks, return_exceptions=True)
            status_tasks.clear()

    photo_files = []  # для TikTok фото — будем чистить в finally
    direct_url_to_save = None  # Прямая ссылка для сохранения в БД

    # Формируем shame-подпись для дубликатов: со ссылкой на первое такое видео в чате
    def _build_shame_caption() -> str | None:
        if not is_duplicate or not duplicate_info:
            return None
        parts = ["Аяйайай бака, скинул повторчик)"]
        dup_username = html.escape(duplicate_info["username"])
        msg_link = get_message_link(
            message.chat.id, duplicate_info.get("bot_message_id"),
            chat_username=message.chat.username,
        )
        when = _human_when(duplicate_info.get("created_at"))
        when_part = f" ({when})" if when else ""
        parts.append(f"Первый кинул: {dup_username}{when_part}")
        if msg_link:
            parts.append(f"<a href=\"{msg_link}\">Первое такое видео</a>")
        return "\n".join(parts)

    try:
        # ============================================================
        #  TikTok (tikwm API) и X/Twitter (fxtwitter API): анализ контента
        #  Определяет фото/видео + даёт прямые ссылки для скачивания
        # ============================================================
        # ============================================================
        #  КЭШ: это видео бот уже отправлял (в любом чате) — шлём по file_id,
        #  ничего не скачивая. Telegram отдаёт его мгновенно.
        # ============================================================
        cached = find_cached_media(video_source_id) if video_source_id else None
        media_cls = {
            "video": InputMediaVideo, "photo": InputMediaPhoto,
            "animation": InputMediaAnimation, "document": InputMediaDocument,
        }.get(cached["file_type"]) if cached else None
        if media_cls:
            try:
                shame_caption = _build_shame_caption()
                await _flush_status()
                edited = await bot.edit_message_media(
                    media=media_cls(
                        media=cached["file_id"],
                        caption=shame_caption,
                        parse_mode="HTML" if shame_caption else None,
                    ),
                    chat_id=message.chat.id,
                    message_id=status_msg_id,
                )
                tg_file_id, tg_file_type = _extract_file_info(edited)
                if user:
                    rec_id = add_video_record(
                        chat_id=message.chat.id,
                        user_id=user.id,
                        username=display_name,
                        url=url,
                        video_source_id=video_source_id,
                        direct_url=cached.get("direct_url"),
                    )
                    # Файла на диске нет — в поисковый индекс эта копия не попадёт
                    await _after_send(
                        message.chat.id, rec_id, status_msg_id, None,
                        kind=cached["file_type"],
                        tg_file_id=tg_file_id or cached["file_id"],
                        tg_file_type=tg_file_type or cached["file_type"],
                    )
                mark("cache_send")
                logging.info(f"Cache hit for {video_source_id}: sent by file_id")
                return
            except Exception as cache_err:
                logging.warning(f"Cached file_id failed for {video_source_id}: {cache_err}")

        if is_tiktok or is_twitter:
            _bg_status((
                    "<h3>📥 Загрузка медиа</h3>\n"
                    "<p>Статус: 🔍 <b>Определение типа контента...</b></p>\n"
                    f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
            ))

            if is_tiktok:
                content = await analyze_tiktok_content(url, output_path)
            else:
                content = await analyze_twitter_content(url)
            direct_url_to_save = content.direct_url
            mark(content.via or ("tikwm" if is_tiktok else "fxtwitter"))

            # Если API дал video_source_id (работает для коротких ссылок vm.tiktok.com)
            if not video_source_id and content.video_source_id:
                video_source_id = content.video_source_id
                if not is_duplicate:
                    duplicate_info = find_duplicate(message.chat.id, video_source_id)
                    if duplicate_info:
                        is_duplicate = True

            # --- ФОТО ---
            if content.content_type == "photos" and content.photo_paths:
                photo_files = content.photo_paths
                shame_caption = _build_shame_caption()
                try:
                    sent_message_id = None
                    tg_file_id = None
                    tg_file_type = None

                    # Не всё скачалось — честно пишем об этом в подписи
                    caption_parts = []
                    if shame_caption:
                        caption_parts.append(shame_caption)
                    if content.photo_total > len(photo_files):
                        caption_parts.append(
                            f"⚠️ Скачалось {len(photo_files)} из {content.photo_total} фото"
                        )
                    photo_caption = "\n\n".join(caption_parts) or None

                    if len(photo_files) == 1:
                        photo = FSInputFile(photo_files[0])
                        await _flush_status()
                        edited = await bot.edit_message_media(
                            media=InputMediaPhoto(
                                media=photo,
                                caption=photo_caption,
                                parse_mode="HTML" if photo_caption else None,
                            ),
                            chat_id=message.chat.id,
                            message_id=status_msg_id,
                        )
                        sent_message_id = status_msg_id
                        tg_file_id, tg_file_type = _extract_file_info(edited)
                        mark("upload")
                    else:
                        # В альбоме Telegram максимум 10 фото — шлём пачками
                        chunks = [photo_files[i:i + 10] for i in range(0, len(photo_files), 10)]
                        # Одиночный хвост в альбом не упаковать — присоединяем к предыдущей пачке
                        if len(chunks) > 1 and len(chunks[-1]) == 1:
                            chunks[-2], chunks[-1] = chunks[-2][:-1], chunks[-2][-1:] + chunks[-1]
                        await _flush_status()
                        for chunk_index, chunk in enumerate(chunks):
                            media_group = [InputMediaPhoto(media=FSInputFile(p)) for p in chunk]
                            if chunk_index == 0 and photo_caption:
                                media_group[0] = InputMediaPhoto(
                                    media=FSInputFile(chunk[0]),
                                    caption=photo_caption,
                                    parse_mode="HTML",
                                )
                            if chunk_index > 0:
                                await asyncio.sleep(1)  # не упираться в флуд-лимит
                            sent_group = await message.reply_media_group(media_group)
                            if chunk_index == 0 and sent_group:
                                sent_message_id = sent_group[0].message_id
                        mark("upload")
                        await bot.delete_message(
                            chat_id=message.chat.id,
                            message_id=status_msg_id,
                        )

                    if user:
                        rec_id = add_video_record(
                            chat_id=message.chat.id,
                            user_id=user.id,
                            username=display_name,
                            url=url,
                            video_source_id=video_source_id,
                            direct_url=direct_url_to_save,
                        )
                        # Первое фото уходит в очередь: enqueue его перенесёт,
                        # и finally ниже уже ничего не найдёт по этому пути.
                        await _after_send(
                            message.chat.id, rec_id, sent_message_id,
                            photo_files[0], kind="image", tg_file_id=tg_file_id, tg_file_type=tg_file_type,
                        )
                    return  # Успешно отправили фото — выходим

                except Exception as photo_send_err:
                    logging.error(f"Failed to send photos: {photo_send_err}")
                    if is_tiktok:
                        # yt-dlp фото-посты TikTok не умеет — фоллбэк бессмысленен
                        raise RuntimeError(f"не удалось отправить фото: {photo_send_err}")
                    # Фото скачались но не отправились — продолжаем к видео

            if is_tiktok and content.content_type == "photos" and not content.photo_paths:
                raise RuntimeError("TikTok отдал слайдшоу, но ни одно фото не скачалось")

            # --- ВИДЕО: прямое скачивание по ссылке из API (без yt-dlp!) ---
            if content.content_type == "video" and (content.video_file or content.video_url):
                _bg_status((
                        "<h3>📥 Загрузка медиа</h3>\n"
                        "<p>Статус: 🟡 <b>Скачивание видео (fast)...</b></p>\n"
                        f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
                ))

                if content.video_file:
                    direct_ok = True  # свой парсер уже скачал (время — в tiktok_page)
                else:
                    direct_ok = await download_video_direct(content.video_url, output_path)
                    mark("download")
                if direct_ok:
                    # Видео скачано напрямую — отправляем
                    shame_caption = _build_shame_caption()
                    video = FSInputFile(output_path)
                    await _flush_status()
                    edited = await bot.edit_message_media(
                        media=InputMediaVideo(
                            media=video,
                            caption=shame_caption,
                            parse_mode="HTML" if shame_caption else None,
                        ),
                        chat_id=message.chat.id,
                        message_id=status_msg_id,
                    )
                    sent_message_id = status_msg_id
                    tg_file_id, tg_file_type = _extract_file_info(edited)
                    mark("upload")

                    if user:
                        rec_id = add_video_record(
                            chat_id=message.chat.id,
                            user_id=user.id,
                            username=display_name,
                            url=url,
                            video_source_id=video_source_id,
                            direct_url=direct_url_to_save,
                        )
                        await _after_send(
                            message.chat.id, rec_id, sent_message_id,
                            output_path, kind="video", tg_file_id=tg_file_id, tg_file_type=tg_file_type,
                        )
                    return  # Успешно — выходим

                else:
                    logging.warning(f"Direct download failed for {url}, falling back to yt-dlp")
                    # Падаем в yt-dlp ниже

        # ============================================================
        #  Скачиваем видео через yt-dlp (YouTube, Instagram, TikTok/X fallback)
        # ============================================================
        _bg_status((
                "<h3>📥 Загрузка медиа</h3>\n"
                "<p>Статус: 🟡 <b>Скачивание видео...</b></p>\n"
                f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
        ))

        await asyncio.wait_for(
            asyncio.to_thread(download_video, url, output_path),
            timeout=180,
        )

        mark("yt-dlp")
        found_files = glob.glob(os.path.join(TEMP_DIR, f"{video_id}*"))
        actual_path = found_files[0] if found_files else output_path

        if os.path.exists(actual_path) and status_msg_id:
            shame_caption = _build_shame_caption()
            video = FSInputFile(actual_path)
            await _flush_status()
            edited = await bot.edit_message_media(
                media=InputMediaVideo(
                    media=video,
                    caption=shame_caption,
                    parse_mode="HTML" if shame_caption else None,
                ),
                chat_id=message.chat.id,
                message_id=status_msg_id,
            )
            sent_message_id = status_msg_id
            tg_file_id, tg_file_type = _extract_file_info(edited)
            mark("upload")

            if user:
                rec_id = add_video_record(
                    chat_id=message.chat.id,
                    user_id=user.id,
                    username=display_name,
                    url=url,
                    video_source_id=video_source_id,
                    direct_url=direct_url_to_save,
                )
                await _after_send(
                    message.chat.id, rec_id, sent_message_id,
                    actual_path, kind="video", tg_file_id=tg_file_id, tg_file_type=tg_file_type,
                )
        else:
            error_html = (
                "<h3>❌ Ошибка загрузки</h3>\n"
                "<p>Не удалось скачать видео по указанной ссылке.</p>\n"
                "<hr/>\n"
                "<p>Детали: <i>Файл не был найден на диске.</i></p>"
            )
            if status_msg_id:
                await _flush_status()
                await _update_status(message.chat.id, status_msg_id, error_html)

    except (asyncio.TimeoutError, yt_dlp.utils.DownloadError, Exception) as e:
        is_timeout = isinstance(e, asyncio.TimeoutError)
        is_download_err = isinstance(e, yt_dlp.utils.DownloadError)

        if is_timeout:
            logging.error(f"Timeout downloading {url}")

        if is_timeout:
            error_html = (
                "<h3>⏱️ Превышен лимит времени</h3>\n"
                "<p>Загрузка файла заняла слишком много времени и была прервана.</p>\n"
                "<hr/>\n"
                "<p>Пожалуйста, попробуйте отправить ссылку ещё раз позже.</p>"
            )
        elif is_download_err and re.search(r'tiktok\.com/@[^/\s]+/photo/', str(e)):
            # tikwm не ответил, а yt-dlp слайдшоу TikTok не поддерживает
            error_html = (
                "<h3>❌ Ошибка загрузки</h3>\n"
                "<p>Это фото-пост TikTok, а сервис для скачивания фото сейчас не отвечает.</p>\n"
                "<hr/>\n"
                "<p>Попробуй отправить ссылку ещё раз через минуту.</p>"
            )
        elif is_download_err:
            # yt-dlp красит текст ANSI-кодами ("\x1b[0;31mERROR:\x1b[0m") — в чате это мусор
            clean_err = re.sub(r'\x1b\[[0-9;]*m', '', str(e))
            clean_err = re.sub(r'^\s*ERROR:\s*', '', clean_err)
            escaped_err = html.escape(clean_err[:150])
            error_html = (
                "<h3>❌ Ошибка загрузки</h3>\n"
                "<p>Не удалось скачать медиа по указанной ссылке.</p>\n"
                "<hr/>\n"
                f"<p>Детали: <code>{escaped_err}</code></p>"
            )
        else:
            escaped_err = html.escape(str(e)[:100])
            error_html = (
                "<h3>❌ Непредвиденная ошибка</h3>\n"
                "<p>Произошла непредвиденная ошибка при обработке медиа.</p>\n"
                "<hr/>\n"
                f"<p>Детали: <code>{escaped_err}</code></p>"
            )

        logging.error(f"Error processing {url}: {e}")
        if status_msg_id:
            await _flush_status()
            await _update_status(message.chat.id, status_msg_id, error_html)

    finally:
        total = time.monotonic() - t_start
        parts = " | ".join(f"{name} {sec:.2f}" for name, sec in stages)
        lag = f"{tg_lag:.1f}s" if tg_lag is not None else "?"
        logging.info(f"⏱ {original_url}: total {total:.2f}s (tg_lag {lag}) | {parts}")
        for f in glob.glob(os.path.join(TEMP_DIR, f"{video_id}*")):
            try:
                os.remove(f)
            except Exception as e:
                logging.error(f"Error removing file {f}: {e}")
        for f in photo_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logging.error(f"Error removing photo file {f}: {e}")


# =====================================================================
#  MEOW — ТРАНСКРИПЦИЯ МЕДИА ЧЕРЕЗ WHISPER
# =====================================================================

@dp.message(F.text.casefold() == "meow")
async def handle_meow_transcribe(message: Message):
    """Обработка 'meow' — транскрипция аудио/видео из replied сообщения."""

    # Должен быть реплай
    reply = message.reply_to_message
    if not reply:
        return  # Просто 'meow' без реплая — игнорируем

    # Провайдер может быть не настроен — говорим об этом прямо, а не молчим
    if not transcription.is_available():
        reason = html.escape(transcription.unavailable_reason() or "")
        hint = html.escape(transcription.setup_hint())
        await send_rich_message(message.chat.id, (
            "<h2>🎙️ Транскрипция аудио</h2>\n"
            "<p>Статус: ⚙️ <b>Функция не настроена</b></p>\n"
            "<hr/>\n"
            f"<p>Причина: {reason}</p>\n"
            f"<p>{hint}</p>\n"
            "<p>После этого бота нужно перезапустить.</p>"
        ), reply_to_message_id=message.message_id)
        return

    # Извлекаем медиа из replied сообщения
    media_info = get_media_from_message(reply)
    if not media_info:
        error_html = (
            "<h2>🎙️ Транскрипция аудио</h2>\n"
            "<p>Статус: ⚠️ <b>Ошибка</b></p>\n"
            "<hr/>\n"
            "<p>Пожалуйста, используйте команду <code>meow</code> в качестве <b>ответа (reply)</b> на сообщение с аудио, видео, голосовым или видео-заметкой.</p>"
        )
        await send_rich_message(message.chat.id, error_html, reply_to_message_id=message.message_id)
        return

    file_id, ext = media_info
    temp_id = str(uuid.uuid4())
    raw_path = os.path.join(TEMP_DIR, f"whisper_{temp_id}{ext}")
    mp3_path = os.path.join(TEMP_DIR, f"whisper_{temp_id}.mp3")

    status_html = (
        "<h2>🎙️ Транскрипция аудио</h2>\n"
        "<p>Статус: 🔄 <b>Распознавание речи...</b></p>\n"
        "<hr/>\n"
        "<p>Пожалуйста, подождите. Это может занять некоторое время в зависимости от длительности аудио.</p>"
    )
    sent_result = await send_rich_message(
        chat_id=message.chat.id,
        html=status_html,
        reply_to_message_id=message.message_id
    )
    status_msg_id = sent_result.get("result", {}).get("message_id") if sent_result.get("ok") else None

    try:
        # 1. Скачиваем файл из Telegram
        try:
            tg_file = await bot.get_file(file_id)
            await bot.download_file(tg_file.file_path, destination=raw_path)
        except Exception as e:
            logging.error(f"Failed to download file from Telegram: {e}")
            error_html = (
                "<h2>🎙️ Транскрипция аудио</h2>\n"
                "<p>Статус: ❌ <b>Не удалось загрузить файл</b></p>\n"
                "<hr/>\n"
                "<p>Возможно, размер файла превышает лимит Telegram для ботов (20 MB).</p>"
            )
            if status_msg_id:
                await edit_rich_message(chat_id=message.chat.id, message_id=status_msg_id, html=error_html)
            return

        if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
            error_html = (
                "<h2>🎙️ Транскрипция аудио</h2>\n"
                "<p>Статус: ❌ <b>Файл пуст</b></p>\n"
                "<hr/>\n"
                "<p>Скачанный файл имеет размер 0 байт.</p>"
            )
            if status_msg_id:
                await edit_rich_message(chat_id=message.chat.id, message_id=status_msg_id, html=error_html)
            return

        # 2. Конвертируем в mp3
        needs_conversion = ext.lower() not in (".mp3",)
        target_path = raw_path

        if needs_conversion:
            success = await asyncio.to_thread(extract_audio_to_mp3, raw_path, mp3_path)
            if not success:
                error_html = (
                    "<h2>🎙️ Транскрипция аудио</h2>\n"
                    "<p>Статус: ❌ <b>Ошибка обработки</b></p>\n"
                    "<hr/>\n"
                    "<p>Произошла ошибка при конвертации аудиодорожки в формат MP3 (ffmpeg).</p>"
                )
                if status_msg_id:
                    await edit_rich_message(chat_id=message.chat.id, message_id=status_msg_id, html=error_html)
                return
            target_path = mp3_path
        else:
            target_path = raw_path

        # 3. Отправляем в Whisper API
        text = await asyncio.to_thread(transcribe_audio, target_path)

        if not text:
            error_html = (
                "<h2>🎙️ Транскрипция аудио</h2>\n"
                "<p>Статус: ❌ <b>Речь не распознана</b></p>\n"
                "<hr/>\n"
                "<p>Не удалось распознать речь. Возможно, аудиофайл не содержит голоса или произошел сбой на стороне API.</p>"
            )
            if status_msg_id:
                await edit_rich_message(chat_id=message.chat.id, message_id=status_msg_id, html=error_html)
            return

        # 4. Отправляем результат
        # Telegram ограничивает сообщение 4096 символами
        escaped_text = html.escape(text)
        if len(escaped_text) > 4000:
            # Отправляем частями
            if status_msg_id:
                await bot.delete_message(
                    chat_id=message.chat.id,
                    message_id=status_msg_id,
                )
            chunks = [escaped_text[i:i+4000] for i in range(0, len(escaped_text), 4000)]
            for i, chunk in enumerate(chunks):
                header = f"<h2>📝 Результат транскрипции (Часть {i+1}/{len(chunks)})</h2>" if len(chunks) > 1 else "<h2>📝 Результат транскрипции</h2>"
                html_msg = (
                    f"{header}\n"
                    "<hr/>\n"
                    f"<blockquote>{chunk}</blockquote>"
                )
                await send_rich_message(message.chat.id, html_msg, reply_to_message_id=message.message_id)
        else:
            html_msg = (
                "<h2>📝 Результат транскрипции</h2>\n"
                "<hr/>\n"
                f"<blockquote>{escaped_text}</blockquote>"
            )
            if status_msg_id:
                await edit_rich_message(chat_id=message.chat.id, message_id=status_msg_id, html=html_msg)
            else:
                await send_rich_message(message.chat.id, html_msg, reply_to_message_id=message.message_id)

    except Exception as e:
        logging.error(f"Unexpected error in meow transcription: {e}")
        escaped_err = html.escape(str(e)[:100])
        error_html = (
            "<h2>🎙️ Транскрипция аудио</h2>\n"
            "<p>Статус: ❌ <b>Непредвиденная ошибка</b></p>\n"
            "<hr/>\n"
            f"<p>Детали: <code>{escaped_err}</code></p>"
        )
        if status_msg_id:
            try:
                await edit_rich_message(chat_id=message.chat.id, message_id=status_msg_id, html=error_html)
            except Exception:
                pass

    finally:
        # Чистим временные файлы
        for path in (raw_path, mp3_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logging.error(f"Error removing temp file {path}: {e}")


# --- Обработчик реакций ---
@dp.message_reaction()
async def handle_reaction(event: MessageReactionUpdated):
    """Реакции на сообщения бота.

    Событие приходит на ОДНОГО человека и содержит его полный набор реакций
    на это сообщение. Пустой набор — штатная ситуация: реакцию сняли.
    Поэтому обновляем строго строки этого пользователя и никого больше.
    """
    # Аноним («от имени чата») приходит без user — берём actor_chat.
    # Если нет ни того, ни другого, привязать реакцию не к кому.
    actor = event.user or event.actor_chat
    if actor is None:
        return

    emojis = []
    for reaction in (event.new_reaction or []):
        emoji = getattr(reaction, "emoji", None)
        if emoji:
            emojis.append(emoji)
            continue
        custom = getattr(reaction, "custom_emoji_id", None)
        if custom:
            emojis.append(f"custom:{custom}")
            continue
        # Платные звёзды и любые будущие типы: факт реакции сохраняем,
        # даже если конкретное изображение нам неизвестно
        kind = getattr(reaction, "type", None)
        if kind:
            emojis.append(f"type:{kind}")

    sync_user_reactions(
        event.chat.id, event.message_id, actor.id, emojis, event.date,
    )


# --- Команда /stats ---
MEDAL_EMOJIS = ["🥇", "🥈", "🥉"]

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    chat_id = message.chat.id
    stats = get_chat_stats(chat_id)

    if not stats:
        empty_html = (
            "<h2>📊 Статистика активности</h2>\n"
            "<hr/>\n"
            "<p>Статистика пуста — ещё никто не отправлял видео в этот чат!</p>"
        )
        await send_rich_message(message.chat.id, empty_html, reply_to_message_id=message.message_id)
        return

    total_videos = sum(s["video_count"] for s in stats)
    total_reactions = sum(s["reaction_count"] for s in stats)

    async def get_user_display_name(s):
        try:
            member = await bot.get_chat_member(chat_id=chat_id, user_id=s["user_id"])
            user_obj = member.user
            return f"@{user_obj.username}" if user_obj.username else user_obj.full_name
        except Exception as e:
            logging.warning(f"Failed to get chat member {s['user_id']}: {e}")
            return s["username"]

    tasks = [get_user_display_name(s) for s in stats]
    names = await asyncio.gather(*tasks)

    table_rows = []
    for i, s in enumerate(stats):
        medal = MEDAL_EMOJIS[i] if i < 3 else f"<b>{i + 1}</b>"
        name = names[i]
        escaped_name = html.escape(name)
        vids = s["video_count"]
        reacts = s["reaction_count"]
        table_rows.append(
            f"  <tr>\n"
            f"    <td>{medal}</td>\n"
            f"    <td><b>{escaped_name}</b></td>\n"
            f"    <td>{vids}</td>\n"
            f"    <td>{reacts}</td>\n"
            f"  </tr>"
        )

    rows_str = "\n".join(table_rows)
    stats_html = (
        "<table>\n"
        "  <tr>\n"
        "    <th></th>\n"
        "    <th>Пользователь</th>\n"
        "    <th>🎬</th>\n"
        "    <th>❤️</th>\n"
        "  </tr>\n"
        f"{rows_str}\n"
        "</table>\n"
        "<hr/>\n"
        f"<p>Всего отправлено видео: <b>{total_videos}</b> | Всего реакций: <b>{total_reactions}</b></p>"
    )

    await send_rich_message(message.chat.id, stats_html, reply_to_message_id=message.message_id)


# =====================================================================
#  INFO — КАРТОЧКА ВИДЕО ИЗ БД В ОТВЕТ НА СООБЩЕНИЕ
# =====================================================================

def _human_when(dt) -> str | None:
    """Дату из БД в человеческий вид: «только что», «25 мин назад», «вчера, 18:04»."""
    if not dt:
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    delta = now - dt
    minutes = delta.total_seconds() / 60
    if minutes < 1:
        return "только что"
    if minutes < 60:
        n = int(minutes)
        return f"{n} мин назад"
    hours = minutes / 60
    if hours < 24:
        n = int(hours)
        return f"{n} ч назад"
    if delta.days == 1:
        return f"вчера, {dt.strftime('%H:%M')}"
    if delta.days < 30:
        return f"{delta.days} дн назад"
    return dt.strftime("%d.%m.%Y")


# Человеческие названия платформ по префиксу video_source_id
_PLATFORM_NAMES = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "instagram": "Instagram",
    "twitter": "X (Twitter)",
}

# Человеческие названия статусов индексации
_EMBED_STATUS = {
    "done": "✅ Готово",
    "pending": "⏳ В очереди",
    "processing": "🔄 Обрабатывается",
    "failed": "❌ Ошибка",
    "skipped": "⏭ Пропущено",
}


@dp.message(Command("info"))
async def cmd_info(message: Message):
    """Карточка видео из базы: кто и когда скинул, реакции, статус индексации и описание."""
    reply = message.reply_to_message
    if not reply:
        await send_rich_message(message.chat.id, (
            "<h2>🎬 Информация о видео</h2>\n"
            "<p>Используй <code>/info</code> в <b>ответ</b> на видео, отправленное ботом.</p>"
        ), reply_to_message_id=message.message_id)
        return

    record = get_video_record_by_message(message.chat.id, reply.message_id)
    if not record:
        await send_rich_message(message.chat.id, (
            "<h2>🎬 Информация о видео</h2>\n"
            "<p>Это сообщение не найдено в базе — возможно, видео отправлено "
            "до подключения бота.</p>"
        ), reply_to_message_id=message.message_id)
        return

    # Платформа: из video_source_id, иначе по домену ссылки
    source_id = record.get("video_source_id") or ""
    platform = _PLATFORM_NAMES.get(source_id.split(":")[0]) or get_domain_name(record.get("url") or "")

    reactions = get_message_reaction_count(message.chat.id, reply.message_id)
    embed_status = _EMBED_STATUS.get(record.get("embed_status")) or "— не индексировалось"

    rows = [
        ("📺 Источник", f"<b>{platform}</b>"),
        ("👤 Отправил", f"<b>{html.escape(record.get('username') or 'кто-то')}</b>"),
        ("📅 Когда", f"<b>{_human_when(record.get('created_at')) or 'неизвестно'}</b>"),
        ("❤️ Реакций", f"<b>{reactions}</b>"),
        ("🧠 Индексация", f"<b>{embed_status}</b>"),
    ]
    rows_str = "\n".join(
        f"  <tr>\n    <td>{label}</td>\n    <td>{value}</td>\n  </tr>"
        for label, value in rows
    )

    card_html = (
        "<h2>🎬 Информация о видео</h2>\n"
        "<table>\n"
        f"{rows_str}\n"
        "</table>"
    )

    # Описание — то, что бот распознал из видео (описание кадров / транскрипт)
    visual_caption = (record.get("visual_caption") or "").strip()
    transcript = (record.get("transcript") or "").strip()

    if visual_caption or transcript:
        card_html += "\n<hr/>"
        if visual_caption:
            escaped_desc = html.escape(visual_caption[:2500])
            if len(visual_caption) > 2500:
                escaped_desc += "…"
            card_html += (
                "\n<details>\n"
                "  <summary>Описание</summary>\n"
                f"  <p>{escaped_desc.replace(chr(10), '<br/>')}</p>\n"
                "</details>"
            )
        if transcript:
            escaped_text = html.escape(transcript[:2500])
            if len(transcript) > 2500:
                escaped_text += "…"
            card_html += (
                "\n<details>\n"
                "  <summary>Транскрипция</summary>\n"
                f"  <p>{escaped_text.replace(chr(10), '<br/>')}</p>\n"
                "</details>"
            )
    else:
        card_html += (
            "\n<hr/>\n"
            "<p><i>Описание недоступно — видео ещё не проиндексировано "
            "или семантический поиск выключен.</i></p>"
        )

    # Ссылка на источник — в сворачиваемом блоке, чтобы не мусорить в карточке
    url = record.get("url")
    if url:
        card_html += (
            "\n<details>\n"
            "  <summary>Ссылка на источник</summary>\n"
            f"  <p><a href=\"{html.escape(url)}\">{html.escape(url)}</a></p>\n"
            "</details>"
        )

    await send_rich_message(message.chat.id, card_html, reply_to_message_id=message.message_id)


# =====================================================================
#  СЕМАНТИЧЕСКИЙ ПОИСК
# =====================================================================

def _embedding_disabled_html() -> str:
    return (
        "<h2>🔍 Поиск по видео</h2>\n"
        "<p>В этом чате <b>семантический поиск не включён</b>.</p>\n"
        "<hr/>\n"
        "<p>Создатель бота может включить его командой <code>/embed_on</code>. "
        "После этого в поиск будут попадать все новые видео — уже отправленные "
        "ранее останутся неиндексированными.</p>"
    )


def _format_results(query_title: str, results: list[tuple[int, float]],
                    chat_id: int) -> str:
    """Собирает выдачу: процент, автор, ссылка на сообщение, кусок текста."""
    details = get_embedding_details([vid for vid, _ in results])

    items = []
    for position, (video_id, score) in enumerate(results, 1):
        info = details.get(video_id)
        if not info:
            continue

        percent = round(score * 100)
        author = html.escape(info.get("username") or "кто-то")
        link = get_message_link(chat_id, info.get("bot_message_id"))

        if link:
            head = f'<b>{position}.</b> <a href="{link}">{percent}% — от {author}</a>'
        else:
            # Ссылку не построить (личка или потерянный message_id) — даём исходный URL
            source = html.escape(info.get("url") or "")
            head = (f'<b>{position}.</b> {percent}% — от {author} '
                    f'(<a href="{source}">источник</a>)' if source
                    else f'<b>{position}.</b> {percent}% — от {author}')

        snippet = (info.get("visual_caption") or info.get("transcript") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > 160:
            snippet = snippet[:160].rstrip() + "…"

        items.append(f"<li>{head}<br/><i>{html.escape(snippet)}</i></li>"
                     if snippet else f"<li>{head}</li>")

    if not items:
        return (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Ничего похожего не нашлось.</p>"
        )

    rows = "\n".join(items)
    return (
        f"<h2>🔍 {query_title}</h2>\n"
        f"<ul>\n{rows}\n</ul>"
    )


@dp.message(Command("embed_on"))
async def cmd_embed_on(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    if not embedding.is_configured():
        reason = html.escape(embedding.unavailable_reason() or "неизвестная причина")
        await send_rich_message(message.chat.id, (
            "<h2>⚠️ Поиск недоступен</h2>\n"
            f"<p>Причина: <b>{reason}</b></p>\n"
            "<hr/>\n"
            "<p>Что сделать:</p>\n"
            "<ul>\n"
            "  <li>Зависимости: <code>pip install -r requirements.txt</code></li>\n"
            "  <li>Ключ: <a href=\"https://openrouter.ai/keys\">openrouter.ai/keys</a> "
            "→ в <code>.env</code> как <code>OPENROUTER_API_KEY</code></li>\n"
            "</ul>\n"
            "<p>После этого перезапусти бота.</p>"
        ), reply_to_message_id=message.message_id)
        return

    if message.chat.id in embedding_groups:
        body = (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Уже <b>включён</b> в этом чате.</p>\n"
            "<hr/>\n"
            "<p>Состояние — <code>/embed_status</code>.</p>"
        )
    else:
        embedding_groups.add(message.chat.id)
        save_data()
        body = (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Семантический поиск <b>включён</b>.</p>\n"
            "<hr/>\n"
            "<p>Новые видео после отправки в чат будут распознаваться и попадать "
            "в индекс. На скорость скачивания это не влияет — обработка идёт фоном.</p>\n"
            "<p>Ранее отправленные видео <b>не индексируются</b>.</p>\n"
            "<ul>\n"
            "  <li><code>/find текст</code> — найти видео по смыслу</li>\n"
            "  <li><code>/similar</code> в ответ на видео — найти похожие</li>\n"
            "  <li><code>/embed_status</code> — состояние и расход</li>\n"
            "</ul>"
        )
    await send_rich_message(message.chat.id, body, reply_to_message_id=message.message_id)


@dp.message(Command("embed_off"))
async def cmd_embed_off(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    if message.chat.id in embedding_groups:
        embedding_groups.discard(message.chat.id)
        save_data()
        body = (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Индексация <b>выключена</b>.</p>\n"
            "<hr/>\n"
            "<p>Новые видео в индекс больше не попадают. Уже построенный индекс "
            "сохраняется — <code>/find</code> продолжит искать по нему.</p>"
        )
    else:
        body = (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Индексация и так <b>не была включена</b> в этом чате.</p>"
        )
    await send_rich_message(message.chat.id, body, reply_to_message_id=message.message_id)


@dp.message(Command("find"))
async def cmd_find(message: Message):
    if message.chat.id not in embedding_groups:
        await send_rich_message(message.chat.id, _embedding_disabled_html(),
                                reply_to_message_id=message.message_id)
        return

    query = re.sub(r"^/find(@\w+)?\s*", "", message.text or "", flags=re.IGNORECASE).strip()
    if not query:
        await send_rich_message(message.chat.id, (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Добавь текст запроса после команды.</p>\n"
            "<hr/>\n"
            "<p>Например: <code>/find кот играет на пианино</code></p>"
        ), reply_to_message_id=message.message_id)
        return

    if embedding.index_size(message.chat.id) == 0:
        await send_rich_message(message.chat.id, (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Индекс пока <b>пуст</b>.</p>\n"
            "<hr/>\n"
            "<p>Видео попадают в него после отправки в чат. Скинь пару роликов "
            "и попробуй снова.</p>"
        ), reply_to_message_id=message.message_id)
        return

    try:
        results = await embedding.search_by_text(message.chat.id, query)
    except Exception as e:
        logging.error(f"Ошибка поиска: {e}")
        await send_rich_message(message.chat.id, (
            "<h2>🔍 Поиск по видео</h2>\n"
            "<p>Не удалось выполнить поиск — сервис эмбеддингов недоступен.</p>\n"
            "<hr/>\n"
            f"<p>Детали: <code>{html.escape(str(e)[:120])}</code></p>"
        ), reply_to_message_id=message.message_id)
        return

    title = f"Поиск: «{html.escape(query[:80])}»"
    await send_rich_message(
        message.chat.id,
        _format_results(title, results, message.chat.id),
        reply_to_message_id=message.message_id,
    )


@dp.message(Command("similar"))
async def cmd_similar(message: Message):
    if message.chat.id not in embedding_groups:
        await send_rich_message(message.chat.id, _embedding_disabled_html(),
                                reply_to_message_id=message.message_id)
        return

    reply = message.reply_to_message
    if not reply:
        await send_rich_message(message.chat.id, (
            "<h2>🔍 Похожие видео</h2>\n"
            "<p>Используй <code>/similar</code> в <b>ответ</b> на сообщение с видео от бота.</p>"
        ), reply_to_message_id=message.message_id)
        return

    video_id = get_video_id_by_message(message.chat.id, reply.message_id)
    if not video_id:
        await send_rich_message(message.chat.id, (
            "<h2>🔍 Похожие видео</h2>\n"
            "<p>Это сообщение не найдено в базе — возможно, видео отправлено "
            "до подключения бота.</p>"
        ), reply_to_message_id=message.message_id)
        return

    results = await embedding.search_by_video(message.chat.id, video_id)
    if results is None:
        await send_rich_message(message.chat.id, (
            "<h2>🔍 Похожие видео</h2>\n"
            "<p>Это видео ещё не проиндексировано.</p>\n"
            "<hr/>\n"
            "<p>Либо обработка не завершилась, либо оно отправлено до включения "
            "поиска. Состояние очереди — <code>/embed_status</code>.</p>"
        ), reply_to_message_id=message.message_id)
        return

    await send_rich_message(
        message.chat.id,
        _format_results("Похожие видео", results, message.chat.id),
        reply_to_message_id=message.message_id,
    )


@dp.message(Command("embed_status"))
async def cmd_embed_status(message: Message):
    report = embedding.status_report(message.chat.id)

    enabled = message.chat.id in embedding_groups
    total = report.get("total_videos", 0)
    done = report.get("done", 0)
    not_embedded = max(0, total - done)
    percent = round(done / total * 100) if total else 0

    limit = report.get("month_limit") or 0
    spent = report.get("month_spend", 0.0)
    left = report.get("month_left")

    if limit > 0:
        left_percent = round((left or 0) / limit * 100)
        budget_row = (
            f"<p>Лимит на месяц: <b>${spent:.2f}</b> из <b>${limit:.2f}</b> "
            f"— осталось <b>${(left or 0):.2f}</b> ({left_percent}%)</p>"
        )
    else:
        budget_row = f"<p>Потрачено за месяц: <b>${spent:.2f}</b> (лимит не задан)</p>"

    queue_pending = report.get("pending", 0) + report.get("processing", 0)

    body = (
        "<h2>🔍 Состояние поиска</h2>\n"
        f"<p>В этом чате: <b>{'включён' if enabled else 'выключен'}</b></p>\n"
        + (f"<p>⚠️ Подсистема не работает: <b>"
           f"{html.escape(report.get('unavailable_reason') or '')}</b></p>\n"
           if not report.get("configured") else "")
        +
        "<hr/>\n"
        "<table>\n"
        "  <tr><th>Показатель</th><th>Значение</th></tr>\n"
        f"  <tr><td>Всего видео в чате</td><td>{total}</td></tr>\n"
        f"  <tr><td>Проиндексировано</td><td>{done} ({percent}%)</td></tr>\n"
        f"  <tr><td>Без индекса</td><td>{not_embedded}</td></tr>\n"
        f"  <tr><td>В очереди</td><td>{queue_pending}</td></tr>\n"
        f"  <tr><td>Пропущено</td><td>{report.get('skipped', 0)}</td></tr>\n"
        f"  <tr><td>Ошибок</td><td>{report.get('failed', 0)}</td></tr>\n"
        "</table>\n"
        "<hr/>\n"
        f"{budget_row}\n"
        f"<p>Файлов в очереди на диске: <b>{report.get('queue_mb', 0):.1f} МБ</b></p>\n"
        "<details>\n"
        "  <summary>Модели и настройки</summary>\n"
        "  <ul>\n"
        f"    <li>Эмбеддинги: <code>{report.get('embed_model')}</code></li>\n"
        f"    <li>Центрирование векторов: "
        f"{'включено' if report.get('centering') else '<b>ВЫКЛЮЧЕНО</b> (нет файла константы!)'}</li>\n"
        f"    <li>Распознавание речи: <code>{report.get('asr_model')}</code></li>\n"
        f"    <li>Описание кадров: <code>{report.get('vlm_model')}</code> "
        f"(режим: {report.get('vlm_policy')})</li>\n"
        f"    <li>Порог автоподсказки «похоже»: {report.get('similar_threshold')}</li>\n"
        "  </ul>\n"
        "</details>"
    )
    await send_rich_message(message.chat.id, body, reply_to_message_id=message.message_id)


@dp.message(Command("embed_retry"))
async def cmd_embed_retry(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    count = retry_failed_jobs(message.chat.id)
    embedding.wake_worker()

    if count:
        body = (
            "<h2>🔁 Переиндексация</h2>\n"
            f"<p>Возвращено в очередь: <b>{count}</b>.</p>\n"
            "<hr/>\n"
            "<p>Задачи, у которых не сохранился файл, повторить нельзя — "
            "они останутся в ошибках.</p>"
        )
    else:
        body = (
            "<h2>🔁 Переиндексация</h2>\n"
            "<p>Повторять нечего — упавших задач с сохранённым файлом нет.</p>"
        )
    await send_rich_message(message.chat.id, body, reply_to_message_id=message.message_id)


# =====================================================================
#  /update — git pull + перезапуск (только админ, только в личке)
# =====================================================================
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str], timeout: int = 120) -> tuple[bool, str]:
    try:
        out = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
        return out.returncode == 0, ((out.stdout or "") + (out.stderr or "")).strip()
    except Exception as e:
        return False, str(e)


def _do_update() -> dict:
    """git pull --ff-only; при изменении requirements.txt — pip install;
    перед перезапуском проверяет, что код компилируется."""
    ok, old = _run(["git", "rev-parse", "HEAD"])
    if not ok:
        return {"ok": False, "stage": "git rev-parse", "out": old}

    ok, pull_out = _run(["git", "pull", "--ff-only"])
    if not ok:
        return {"ok": False, "stage": "git pull", "out": pull_out}

    _, new = _run(["git", "rev-parse", "HEAD"])
    if new == old:
        return {"ok": True, "changed": False}

    _, log = _run(["git", "log", "--oneline", "--no-decorate", "-n", "15", f"{old}..{new}"])
    _, files = _run(["git", "diff", "--name-only", old, new])
    changed_files = files.splitlines()

    pip_out = None
    if "requirements.txt" in changed_files:
        ok, pip_out = _run([sys.executable, "-m", "pip", "install", "-q",
                            "--disable-pip-version-check", "-r", "requirements.txt"], timeout=900)
        if not ok:
            return {"ok": False, "stage": "pip install", "out": pip_out, "log": log}

    # Сломанный код после рестарта не поднимется — проверяем заранее
    py_files = [f for f in os.listdir(REPO_DIR) if f.endswith(".py")]
    ok, compile_out = _run([sys.executable, "-m", "py_compile", *py_files])
    if not ok:
        return {"ok": False, "stage": "py_compile", "out": compile_out, "log": log}

    return {"ok": True, "changed": True, "log": log,
            "old": old[:7], "new": new[:7], "pip": pip_out is not None}


@dp.message(Command("update"))
async def cmd_update(message: Message):
    if not message.from_user or message.from_user.id != ADMIN_ID:
        return
    if message.chat.type != "private":
        return

    force = "force" in (message.text or "").lower()
    status = await message.answer("⏳ Обновляюсь: <code>git pull</code>...", parse_mode="HTML")
    res = await asyncio.to_thread(_do_update)

    if not res["ok"]:
        text = (f"❌ Обновление остановлено на шаге <b>{res['stage']}</b>, "
                f"бот работает на старой версии.\n"
                f"<pre>{html.escape(res['out'][-3000:])}</pre>")
        if res["stage"] in ("pip install", "py_compile"):
            text += "\n⚠️ Новый код уже на диске — до исправления не перезапускай бота руками."
        await status.edit_text(text, parse_mode="HTML")
        return

    if not res["changed"] and not force:
        await status.edit_text(
            "✅ Уже последняя версия, перезапуск не нужен.\n"
            "Перезапустить всё равно: <code>/update force</code>", parse_mode="HTML")
        return

    if res["changed"]:
        version = res["new"]
        text = (f"✅ Обновлено <code>{res['old']}</code> → <code>{res['new']}</code>"
                + (" (+ зависимости)" if res["pip"] else "") + "\n"
                f"<pre>{html.escape(res['log'][-3000:])}</pre>\n")
    else:
        version = None
        text = "✅ Изменений нет, перезапускаю по запросу.\n"
    busy = f" (жду {_active_downloads} загрузок)" if _active_downloads else ""
    await status.edit_text(text + f"🔄 Перезапускаюсь{busy}...", parse_mode="HTML")

    # Новый процесс прочитает это из окружения и отчитается, что поднялся
    os.environ["BOT_RESTART_NOTICE"] = (
        f"✅ Бот перезапущен, версия {version}." if version else "✅ Бот перезапущен."
    )
    await graceful_restart()


# =====================================================================
#  /settings — переключатели вместо правки .env (только админ, в личке)
# =====================================================================
_this_module = sys.modules[__name__]

settings_store = SettingsStore([
    Setting("TIKTOK_SCRAPER", "TikTok: свой парсер", _this_module, "TIKTOK_SCRAPER",
            [(True, "✅ вкл"), (False, "❌ выкл (tikwm)")],
            "свой разбор страницы ~1.5 с; выкл — всегда через tikwm (5-9 с)"),
    Setting("YTDLP_UPDATE_HOURS", "Автообновление yt-dlp", _this_module, "YTDLP_UPDATE_HOURS",
            [(24.0, "раз в 24 ч"), (12.0, "раз в 12 ч"), (6.0, "раз в 6 ч"),
             (48.0, "раз в 48 ч"), (0.0, "❌ выкл")],
            "pip install -U и перезапуск, если вышла новая версия"),
    Setting("TRANSCRIBE_PROVIDER", "Распознавание речи", transcription, "TRANSCRIBE_PROVIDER",
            [("auto", "авто"), ("deepinfra", "DeepInfra"), ("openrouter", "OpenRouter"),
             ("off", "❌ выкл")],
            "команда meow и транскрипт для поиска"),
    Setting("VLM_POLICY", "Описание кадров (VLM)", embedding, "VLM_POLICY",
            [("always", "всегда"), ("no_speech", "только без речи"), ("never", "❌ никогда")],
            "главный рычаг стоимости поиска; «всегда» — точнее всего"),
    Setting("VLM_FRAMES", "Кадров в VLM", embedding, "VLM_FRAMES",
            [(4, "4"), (6, "6"), (8, "8"), (2, "2")],
            "больше кадров — точнее описание, но дороже"),
    Setting("EMBED_MONTHLY_LIMIT_USD", "Лимит на месяц", embedding, "MONTHLY_LIMIT_USD",
            [(5.0, "$5"), (10.0, "$10"), (20.0, "$20"), (1.0, "$1"), (2.0, "$2"),
             (0.0, "без лимита")],
            "при исчерпании индексация встаёт, видео качаются как обычно"),
    Setting("EMBED_SIMILAR_THRESHOLD", "Подсказка «похоже»", embedding, "SIMILAR_NOTIFY_THRESHOLD",
            [(1.0, "❌ выкл"), (0.9, "от 90%"), (0.85, "от 85%"), (0.8, "от 80%")],
            "бот сам подписывает ссылку на похожий ролик"),
    Setting("EMBED_FIND_MIN_SCORE", "Порог выдачи /find", embedding, "FIND_MIN_SCORE",
            [(0.2, "0.20"), (0.25, "0.25"), (0.3, "0.30"), (0.1, "0.10"), (0.15, "0.15")],
            "результаты ниже порога не показываются"),
])


def _settings_text() -> str:
    lines = ["<b>⚙️ Настройки бота</b>",
             "Нажми на кнопку — значение переключится на следующее и сразу применится.",
             "<i>* — изменено здесь, без звёздочки — значение из .env</i>", ""]
    for st in settings_store.settings:
        mark = " *" if settings_store.is_overridden(st.key) else ""
        lines.append(f"<b>{html.escape(st.title)}</b>: {html.escape(st.label(st.current()))}{mark}")
        if st.hint:
            lines.append(f"  <i>{html.escape(st.hint)}</i>")
    return "\n".join(lines)


def _settings_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for i, st in enumerate(settings_store.settings):
        mark = " *" if settings_store.is_overridden(st.key) else ""
        rows.append([InlineKeyboardButton(
            text=f"{st.title}: {st.label(st.current())}{mark}",
            callback_data=f"set:{i}",
        )])
    rows.append([
        InlineKeyboardButton(text="↩️ Сбросить к .env", callback_data="set:reset"),
        InlineKeyboardButton(text="✖️ Закрыть", callback_data="set:close"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(Command("settings"))
async def cmd_settings(message: Message):
    if not message.from_user or message.from_user.id != ADMIN_ID:
        return
    if message.chat.type != "private":
        return
    await message.answer(_settings_text(), parse_mode="HTML",
                         reply_markup=_settings_keyboard())


@dp.callback_query(F.data.startswith("set:"))
async def on_settings_button(callback: CallbackQuery):
    if not callback.from_user or callback.from_user.id != ADMIN_ID:
        await callback.answer("Только для админа", show_alert=True)
        return

    action = callback.data.split(":", 1)[1]
    if action == "close":
        await callback.message.delete()
        await callback.answer()
        return

    if action == "reset":
        settings_store.reset()
        note = "Все настройки сброшены к .env"
    else:
        try:
            st = settings_store.settings[int(action)]
        except (ValueError, IndexError):
            await callback.answer("Устаревшая кнопка — открой /settings заново")
            return
        value = settings_store.cycle(st.key)
        note = f"{st.title}: {st.label(value)}"
        logging.info(f"Админ изменил настройку {st.key} = {value!r}")

    try:
        await callback.message.edit_text(_settings_text(), parse_mode="HTML",
                                         reply_markup=_settings_keyboard())
    except Exception as e:
        logging.debug(f"settings edit: {e}")  # «message is not modified» и т.п.
    await callback.answer(note)


@dp.message(F.text)
async def handle_message(message: Message):
    # Если это группа, и она не инициализирована — игнорируем
    if message.chat.type in ["group", "supergroup"]:
        if message.chat.id not in active_groups:
            return

    # Ищем все ссылки на поддерживаемые площадки
    urls = URL_REGEX.findall(message.text)
    if not urls:
        return

    # Запускаем обработку каждой темы параллельно, как фоновые задачи
    for url in urls:
        asyncio.create_task(_tracked_download(process_url(message, url)))


# =====================================================================
#  АВТООБНОВЛЕНИЕ YT-DLP
# =====================================================================
# Площадки (особенно Instagram) регулярно ломают экстракторы, а фиксы
# выходят в новых версиях yt-dlp. Раз в YTDLP_UPDATE_HOURS бот ставит
# свежую версию через pip и, если она поменялась, перезапускается —
# предварительно дождавшись конца текущих загрузок. 0 — выключено.

_active_downloads = 0
_restart_requested = False


async def _tracked_download(coro):
    """Считает активные загрузки, чтобы не перезапускаться посреди скачивания."""
    global _active_downloads
    _active_downloads += 1
    try:
        await coro
    finally:
        _active_downloads -= 1


def _installed_ytdlp_version() -> str | None:
    """Версия yt-dlp на диске (в отдельном процессе: загруженный модуль — старый)."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import yt_dlp; print(yt_dlp.version.__version__)"],
            capture_output=True, text=True, timeout=60,
        )
        return out.stdout.strip() or None
    except Exception as e:
        logging.error(f"Не удалось узнать версию yt-dlp: {e}")
        return None


def _pip_upgrade_ytdlp() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "-q",
             "--disable-pip-version-check", "yt-dlp[default]"],
            capture_output=True, text=True, timeout=600,
        )
        return out.returncode == 0, (out.stderr or out.stdout)[-500:]
    except Exception as e:
        return False, str(e)


async def graceful_restart(notice: str | None = None):
    """Дожидается конца текущих загрузок (не дольше 10 минут), пишет админу
    и останавливает polling. Сам перезапуск делает _restart_process() после
    выхода из main()."""
    global _restart_requested
    if _restart_requested:
        return
    _restart_requested = True
    for _ in range(120):
        if _active_downloads == 0:
            break
        await asyncio.sleep(5)
    if notice:
        try:
            await bot.send_message(ADMIN_ID, notice)
        except Exception:
            pass
    try:
        await dp.stop_polling()
    except RuntimeError as e:
        # polling ещё не стартовал — перезапуск всё равно произойдёт после выхода
        logging.warning(f"stop_polling: {e}")


async def ytdlp_auto_updater():
    """Работает всегда; интервал (и выключение — 0) берётся из настройки
    на каждом круге, поэтому меняется из /settings без перезапуска."""
    running_version = yt_dlp.version.__version__
    last_check = None
    while True:
        hours = YTDLP_UPDATE_HOURS
        loop_time = asyncio.get_running_loop().time()
        if hours <= 0 or (last_check is not None and loop_time - last_check < hours * 3600):
            await asyncio.sleep(60)
            continue
        last_check = loop_time
        ok, details = await asyncio.to_thread(_pip_upgrade_ytdlp)
        if not ok:
            logging.error(f"Автообновление yt-dlp не удалось: {details}")
        else:
            new_version = await asyncio.to_thread(_installed_ytdlp_version)
            if new_version and new_version != running_version:
                logging.warning(f"yt-dlp обновлён: {running_version} → {new_version}. Перезапуск...")
                await graceful_restart(
                    f"🔄 yt-dlp обновлён: {running_version} → {new_version}. Бот перезапускается."
                )
                return
            logging.info(f"yt-dlp актуален: {running_version}")

async def _handle_similar_found(chat_id: int, new_video_id: int,
                                similar_video_id: int, score: float):
    """Автоподсказка «похоже на это» при отправке нового видео.

    Молчит, пока EMBED_SIMILAR_THRESHOLD не опустят ниже 1.0.
    """
    details = get_embedding_details([new_video_id, similar_video_id])
    new_info = details.get(new_video_id)
    old_info = details.get(similar_video_id)
    if not new_info or not old_info:
        return

    link = get_message_link(chat_id, old_info.get("bot_message_id"))
    if not link:
        return

    author = html.escape(old_info.get("username") or "кто-то")
    body = (
        f"<p>🤔 Похоже на <a href=\"{link}\">это видео от {author}</a> "
        f"({round(score * 100)}%)</p>"
    )
    await send_rich_message(
        chat_id, body,
        reply_to_message_id=new_info.get("bot_message_id"),
    )


async def main():
    global _http_session
    logging.basicConfig(level=logging.INFO)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)
    load_data()
    settings_store.load()
    init_db()

    # Поисковый индекс держим в памяти: пара тысяч векторов — единицы мегабайт
    indexed = embedding.load_index()
    logging.info(
        f"Эмбеддинги: {indexed} векторов, "
        f"включены в {len(embedding_groups)} чатах"
    )

    # Pre-create the shared HTTP session so the first request doesn't pay the connection cost
    _http_session = aiohttp.ClientSession()

    worker_task = None
    if embedding.is_configured():
        embedding.similar_callback = _handle_similar_found
        worker_task = asyncio.create_task(embedding.worker(bot, ADMIN_ID))
    elif embedding_groups:
        logging.warning(
            "Поиск включён в %s чатах, но не работает: %s. "
            "Видео качаются и отправляются как обычно.",
            len(embedding_groups), embedding.unavailable_reason(),
        )

    updater_task = asyncio.create_task(ytdlp_auto_updater())

    print("Бот запущен!")
    try:
        # Запускаем polling, игнорируем накопившиеся апдейты.
        # После автоперезапуска не игнорируем: это сообщения за последние секунды.
        auto_restarted = os.environ.pop("BOT_AUTO_RESTARTED", None) == "1"
        await bot.delete_webhook(drop_pending_updates=not auto_restarted)
        restart_notice = os.environ.pop("BOT_RESTART_NOTICE", None)
        if restart_notice:
            try:
                await bot.send_message(ADMIN_ID, restart_notice)
            except Exception as e:
                logging.warning(f"Не удалось отправить уведомление о перезапуске: {e}")
        await dp.start_polling(bot, allowed_updates=["message", "message_reaction", "callback_query"])
    finally:
        if updater_task and not updater_task.done():
            updater_task.cancel()
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        if _http_session and not _http_session.closed:
            await _http_session.close()

def _restart_process():
    """Перезапускает бота тем же интерпретатором с теми же аргументами."""
    os.environ["BOT_AUTO_RESTARTED"] = "1"
    args = [sys.executable] + sys.argv
    if os.name == "nt":
        # os.execv на Windows не заменяет процесс, а отвязывает новый от консоли
        subprocess.Popen(args)
        sys.exit(0)
    os.execv(sys.executable, args)


if __name__ == "__main__":
    asyncio.run(main())
    if _restart_requested:
        _restart_process()
