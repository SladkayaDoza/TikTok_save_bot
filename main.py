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
from datetime import datetime
from dataclasses import dataclass, field
import subprocess
import requests as http_requests
import aiohttp
from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile, InputMediaVideo, InputMediaPhoto,
    MessageReactionUpdated,
)
import yt_dlp
import dotenv

dotenv.load_dotenv()

from database import (
    init_db, add_video_record, update_bot_message_id,
    sync_user_reactions, get_chat_stats, find_duplicate,
    get_embedding_details, get_video_id_by_message, retry_failed_jobs,
    get_video_record_by_message, get_message_reaction_count,
)
import embedding
import transcription

# --- CONFIGURATION ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID"))
DATA_FILE = "active_groups.json"
TEMP_DIR = "temporary_videos"
MEDIA_DIR = "media"

# Распознавание речи живёт в transcription.py: провайдер (DeepInfra / OpenRouter /
# выключено) выбирается там, клиент не создаётся на старте. Бот запускается
# и работает даже без единого ключа — просто без этих функций.

bot = Bot(token=TOKEN)
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

# Регулярное выражение для поиска ссылок YouTube (в т.ч. Shorts), TikTok и Instagram
URL_REGEX = re.compile(
    r'(https?://(?:www\.)?(?:youtube\.com|youtu\.be|tiktok\.com|vm\.tiktok\.com|vt\.tiktok\.com|instagram\.com)[^\s]+)'
)

TIKTOK_URL_REGEX = re.compile(
    r'https?://(?:www\.)?(?:tiktok\.com|vm\.tiktok\.com|vt\.tiktok\.com)'
)


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

    return None


def get_message_link(chat_id: int, message_id: int) -> str | None:
    """Constructs a Telegram deep link to a specific message in a supergroup."""
    if not message_id:
        return None
    if chat_id < 0:
        # Supergroup: chat_id is -100XXXXXXXXXX → need XXXXXXXXXX
        clean_id = str(abs(chat_id))
        if clean_id.startswith("100"):
            clean_id = clean_id[3:]
        return f"https://t.me/c/{clean_id}/{message_id}"
    return None


async def download_video_direct(url: str, output_path: str) -> bool:
    """Скачивает видео напрямую через aiohttp (без yt-dlp).
    Используется для TikTok видео когда tikwm API даёт прямую ссылку."""
    session = await get_http_session()
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logging.warning(f"Direct download failed: HTTP {resp.status}")
                return False
            with open(output_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(65536):
                    f.write(chunk)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
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

async def _download_image_list(image_urls: list[str], source_name: str) -> list[str]:
    """Скачивает список изображений по URL. Возвращает пути к файлам."""
    session = await get_http_session()
    downloaded = []
    for i, img_url in enumerate(image_urls[:10]):
        try:
            async with session.get(img_url) as img_resp:
                if img_resp.status != 200:
                    continue
                content = await img_resp.read()
                if not content or len(content) < 100:
                    continue
                ct = img_resp.headers.get("Content-Type", "")
                ext = ".jpg"
                if "png" in ct:
                    ext = ".png"
                elif "webp" in ct:
                    ext = ".webp"
                photo_path = os.path.join(TEMP_DIR, f"photo_{uuid.uuid4()}{ext}")
                with open(photo_path, "wb") as f:
                    f.write(content)
                if os.path.getsize(photo_path) > 0:
                    downloaded.append(photo_path)
        except Exception as e:
            logging.error(f"Error downloading photo {i} via {source_name}: {e}")
    return downloaded


async def analyze_tiktok_content(url: str) -> TikTokContent:
    """Анализирует TikTok URL через tikwm API (быстро, с таймаутами).
    Определяет тип контента (фото/видео), извлекает прямые ссылки.
    Для фото — скачивает файлы. Для видео — возвращает direct URL.
    НЕ использует yt-dlp для анализа — только лёгкие HTTP API."""

    result = TikTokContent()

    if not TIKTOK_URL_REGEX.match(url):
        return result

    session = await get_http_session()

    # --- Метод 1: tikwm.com API (основной) — таймаут 4 сек ---
    try:
        api_url = "https://www.tikwm.com/api/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        timeout = aiohttp.ClientTimeout(total=4)
        async with session.post(api_url, data={"url": url, "hd": 1}, headers=headers, timeout=timeout) as resp:
            if resp.status == 200:
                api_data = await resp.json()
                if api_data.get("code") == 0:
                    data = api_data.get("data", {})

                    # Извлекаем video_source_id из API (работает даже для vm.tiktok.com)
                    tiktok_id = data.get("id")
                    if tiktok_id:
                        result.video_source_id = f"tiktok:{tiktok_id}"

                    images = data.get("images", [])

                    if images and isinstance(images, list) and len(images) > 0:
                        # ===== ФОТО-ПОСТ =====
                        real_images = [img for img in images if isinstance(img, str) and img.startswith("http")]
                        if real_images:
                            downloaded = await _download_image_list(real_images, "tikwm")
                            if downloaded:
                                result.content_type = "photos"
                                result.photo_paths = downloaded
                                result.direct_url = real_images[0]
                                logging.info(f"TikTok photo slideshow via tikwm: {len(downloaded)} photos")
                                return result
                    else:
                        # ===== ВИДЕО-ПОСТ =====
                        video_url = data.get("hdplay") or data.get("play")
                        if video_url and video_url.startswith("http"):
                            result.content_type = "video"
                            result.video_url = video_url
                            result.direct_url = data.get("play") or video_url
                            logging.info(f"TikTok video direct URL via tikwm (fast path)")
                            return result
                        else:
                            # tikwm определил что видео, но не дал ссылку — yt-dlp скачает
                            result.content_type = "video"
                            logging.debug(f"tikwm: video post, no play URL — yt-dlp fallback")
                            return result
    except asyncio.TimeoutError:
        logging.warning(f"tikwm API timeout (4s) for {url}")
    except Exception as e:
        logging.debug(f"tikwm API failed for {url}: {e}")

    # --- Метод 2: tikcdn.io API (резервный, только фото) — таймаут 3 сек ---
    try:
        api_url = "https://tikcdn.io/ssstik/video"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        timeout = aiohttp.ClientTimeout(total=3)
        async with session.post(api_url, data={"id": url, "locale": "en", "tt": "1"}, headers=headers, timeout=timeout) as resp:
            if resp.status == 200:
                text = await resp.text()
                img_matches = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)[^\s"\'<>]*', text)
                if img_matches:
                    downloaded = await _download_image_list(img_matches, "tikcdn")
                    if downloaded:
                        result.content_type = "photos"
                        result.photo_paths = downloaded
                        return result
    except asyncio.TimeoutError:
        logging.warning(f"tikcdn API timeout (3s) for {url}")
    except Exception as e:
        logging.debug(f"tikcdn API failed for {url}: {e}")

    # Оба API не сработали — возвращаем "unknown", process_url скачает через yt-dlp
    logging.debug(f"All TikTok APIs failed for {url}, falling back to yt-dlp download")
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
        'quiet': True,
        'no_warnings': True,
        # YouTube требует JS-рантайм для решения n-challenge (иначе форматы
        # отбрасываются и падаем в "Requested format is not available").
        # По умолчанию yt-dlp ищет только deno — разрешаем ещё node/bun.
        'js_runtimes': {'deno': {}, 'node': {}, 'bun': {}},
        # android_vr не поддерживает cookies и молча пропускается, когда задан
        # cookiefile, поэтому держим cookie-совместимые клиенты первыми.
        'extractor_args': {'youtube': {'player_client': ['default', 'android_vr']}},
        # Конвертация в mp4 через ffmpeg если нужно
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    # Подключаем куки, если файл существует (критично для Instagram)
    if os.path.exists(cookies_path):
        ydl_opts['cookiefile'] = cookies_path
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return  # Успех
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            err_msg = str(e).lower()
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
                      tg_file_id: str | None = None):
    """Общий хвост после успешной отправки медиа.

    Вызывается ПОСЛЕ того, как видео уже улетело в чат, поэтому на скорость
    доставки не влияет. Файл переносится в очередь и обрабатывается фоном.
    """
    if rec_id <= 0:
        return

    if sent_message_id:
        update_bot_message_id(rec_id, sent_message_id, tg_file_id)

    if chat_id in embedding_groups and media_path:
        try:
            await embedding.enqueue_media(rec_id, chat_id, media_path, kind)
        except Exception as e:
            logging.error(f"Не удалось поставить видео {rec_id} в очередь: {e}")


async def process_url(message: Message, url: str):
    video_id = str(uuid.uuid4())
    output_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")

    # Определяем имя пользователя для статистики
    user = message.from_user
    if user:
        display_name = f"@{user.username}" if user.username else user.full_name
    else:
        display_name = "Unknown"

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

    status_html = (
        "<h3>📥 Загрузка медиа</h3>\n"
        "<p>Статус: 🟡 <b>Скачивание...</b></p>\n"
        f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
    )

    # Send stub GIF (instant after first upload) — single message, no extra notification
    fallback_caption = to_fallback_html(status_html)
    status_msg = await send_stub_animation(
        chat_id=message.chat.id,
        caption=fallback_caption,
        reply_to_message_id=message.message_id,
    )
    status_msg_id = status_msg.message_id

    photo_files = []  # для TikTok фото — будем чистить в finally
    direct_url_to_save = None  # Прямая ссылка для сохранения в БД

    # Формируем shame-подпись для дубликатов
    def _build_shame_caption() -> str | None:
        if not is_duplicate or not duplicate_info:
            return None
        parts = ["Аяйайай бака, скинул повторчик)"]
        dup_username = html.escape(duplicate_info["username"])
        msg_link = get_message_link(message.chat.id, duplicate_info.get("bot_message_id"))
        if msg_link:
            parts.append(f"Первый кинул: <a href=\"{msg_link}\">{dup_username}</a>")
        else:
            parts.append(f"Первый кинул: {dup_username}")
        return "\n".join(parts)

    try:
        # ============================================================
        #  TikTok: анализ контента через tikwm API
        #  Определяет фото/видео + даёт прямые ссылки для скачивания
        # ============================================================
        if TIKTOK_URL_REGEX.match(url):
            try:
                await _update_status(message.chat.id, status_msg_id, (
                    "<h3>📥 Загрузка медиа</h3>\n"
                    "<p>Статус: 🔍 <b>Определение типа контента...</b></p>\n"
                    f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
                ))
            except Exception:
                pass

            content = await analyze_tiktok_content(url)
            direct_url_to_save = content.direct_url

            # Если tikwm дал video_source_id (работает для коротких ссылок vm.tiktok.com)
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

                    if len(photo_files) == 1:
                        photo = FSInputFile(photo_files[0])
                        edited = await bot.edit_message_media(
                            media=InputMediaPhoto(
                                media=photo,
                                caption=shame_caption,
                                parse_mode="HTML" if shame_caption else None,
                            ),
                            chat_id=message.chat.id,
                            message_id=status_msg_id,
                        )
                        sent_message_id = status_msg_id
                        tg_file_id = _extract_file_id(edited)
                    else:
                        media_group = [
                            InputMediaPhoto(media=FSInputFile(p))
                            for p in photo_files[:10]
                        ]
                        # Добавляем shame-подпись к первому фото
                        if shame_caption:
                            media_group[0] = InputMediaPhoto(
                                media=FSInputFile(photo_files[0]),
                                caption=shame_caption,
                                parse_mode="HTML",
                            )
                        sent_group = await message.reply_media_group(media_group)
                        if sent_group:
                            sent_message_id = sent_group[0].message_id
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
                            photo_files[0], kind="image", tg_file_id=tg_file_id,
                        )
                    return  # Успешно отправили фото — выходим

                except Exception as photo_send_err:
                    logging.error(f"Failed to send TikTok photos: {photo_send_err}")
                    # Фото скачались но не отправились — продолжаем к видео

            # --- TikTok ВИДЕО: прямое скачивание через tikwm (без yt-dlp!) ---
            if content.content_type == "video" and content.video_url:
                try:
                    await _update_status(message.chat.id, status_msg_id, (
                        "<h3>📥 Загрузка медиа</h3>\n"
                        "<p>Статус: 🟡 <b>Скачивание видео (fast)...</b></p>\n"
                        f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
                    ))
                except Exception:
                    pass

                direct_ok = await download_video_direct(content.video_url, output_path)
                if direct_ok:
                    # Видео скачано напрямую — отправляем
                    shame_caption = _build_shame_caption()
                    video = FSInputFile(output_path)
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
                    tg_file_id = _extract_file_id(edited)

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
                            output_path, kind="video", tg_file_id=tg_file_id,
                        )
                    return  # Успешно — выходим

                else:
                    logging.warning(f"Direct download failed for {url}, falling back to yt-dlp")
                    # Падаем в yt-dlp ниже

        # ============================================================
        #  Скачиваем видео через yt-dlp (YouTube, Instagram, TikTok fallback)
        # ============================================================
        try:
            await _update_status(message.chat.id, status_msg_id, (
                "<h3>📥 Загрузка медиа</h3>\n"
                "<p>Статус: 🟡 <b>Скачивание видео...</b></p>\n"
                f"<p>Источник: <a href=\"{escaped_url}\">{domain}</a></p>"
            ))
        except Exception:
            pass

        await asyncio.wait_for(
            asyncio.to_thread(download_video, url, output_path),
            timeout=180,
        )

        found_files = glob.glob(os.path.join(TEMP_DIR, f"{video_id}*"))
        actual_path = found_files[0] if found_files else output_path

        if os.path.exists(actual_path) and status_msg_id:
            shame_caption = _build_shame_caption()
            video = FSInputFile(actual_path)
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
            tg_file_id = _extract_file_id(edited)

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
                    actual_path, kind="video", tg_file_id=tg_file_id,
                )
        else:
            error_html = (
                "<h3>❌ Ошибка загрузки</h3>\n"
                "<p>Не удалось скачать видео по указанной ссылке.</p>\n"
                "<hr/>\n"
                "<p>Детали: <i>Файл не был найден на диске.</i></p>"
            )
            if status_msg_id:
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
        elif is_download_err:
            escaped_err = html.escape(str(e)[:100])
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
            await _update_status(message.chat.id, status_msg_id, error_html)

    finally:
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
    now = datetime.utcnow()
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
        asyncio.create_task(process_url(message, url))

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

    print("Бот запущен!")
    try:
        # Запускаем polling, игнорируем накопившиеся апдейты
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=["message", "message_reaction"])
    finally:
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        if _http_session and not _http_session.closed:
            await _http_session.close()

if __name__ == "__main__":
    asyncio.run(main())
