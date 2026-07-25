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
from dataclasses import dataclass, field
import subprocess
import requests as http_requests
import aiohttp
from openai import OpenAI
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
    sync_reactions, get_chat_stats, find_duplicate,
)

# --- CONFIGURATION ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID"))
DATA_FILE = "active_groups.json"
TEMP_DIR = "temporary_videos"
MEDIA_DIR = "media"

# --- DeepInfra Whisper ---
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  # Замените на ваш ключ DeepInfra
whisper_client = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)
# ---------------------

bot = Bot(token=TOKEN)
dp = Dispatcher()
active_groups = set()

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
    global active_groups
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                active_groups = set(json.load(f))
        except Exception as e:
            logging.error(f"Failed to load active groups: {e}")

def save_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(list(active_groups), f)
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
    """Отправляет mp3 файл в DeepInfra Whisper API (OpenAI SDK).
    Возвращает текст транскрипции или None при ошибке."""
    try:
        with open(mp3_path, "rb") as f:
            transcript = whisper_client.audio.transcriptions.create(
                model="openai/whisper-large-v3",
                file=f,
                language="ru"
            )
        text = transcript.text.strip() if transcript.text else ""
        return text if text else None

    except Exception as e:
        logging.error(f"Whisper API error: {e}")
        return None


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
        parts = ["Аяйайай бака 🫵😹, видео других не смотрит а скинул повторчик)"]
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
                    if len(photo_files) == 1:
                        photo = FSInputFile(photo_files[0])
                        await bot.edit_message_media(
                            media=InputMediaPhoto(
                                media=photo,
                                caption=shame_caption,
                                parse_mode="HTML" if shame_caption else None,
                            ),
                            chat_id=message.chat.id,
                            message_id=status_msg_id,
                        )
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
                        await message.reply_media_group(media_group)
                        await bot.delete_message(
                            chat_id=message.chat.id,
                            message_id=status_msg_id,
                        )

                    if user:
                        add_video_record(
                            chat_id=message.chat.id,
                            user_id=user.id,
                            username=display_name,
                            url=url,
                            video_source_id=video_source_id,
                            direct_url=direct_url_to_save,
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
                    await bot.edit_message_media(
                        media=InputMediaVideo(
                            media=video,
                            caption=shame_caption,
                            parse_mode="HTML" if shame_caption else None,
                        ),
                        chat_id=message.chat.id,
                        message_id=status_msg_id,
                    )
                    sent_message_id = status_msg_id

                    # Получаем telegram_file_id из обновлённого сообщения
                    tg_file_id = None
                    try:
                        # Сообщение уже было edit_message_media → видео загружено
                        # file_id можно получить из ответа, но edit_message_media
                        # в aiogram не всегда возвращает обновлённое сообщение.
                        # Пока оставляем None — заполним в будущем если понадобится.
                        pass
                    except Exception:
                        pass

                    if user:
                        rec_id = add_video_record(
                            chat_id=message.chat.id,
                            user_id=user.id,
                            username=display_name,
                            url=url,
                            video_source_id=video_source_id,
                            direct_url=direct_url_to_save,
                        )
                        if rec_id > 0 and sent_message_id:
                            update_bot_message_id(rec_id, sent_message_id, tg_file_id)
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
            timeout=120,
        )

        found_files = glob.glob(os.path.join(TEMP_DIR, f"{video_id}*"))
        actual_path = found_files[0] if found_files else output_path

        if os.path.exists(actual_path) and status_msg_id:
            shame_caption = _build_shame_caption()
            video = FSInputFile(actual_path)
            await bot.edit_message_media(
                media=InputMediaVideo(
                    media=video,
                    caption=shame_caption,
                    parse_mode="HTML" if shame_caption else None,
                ),
                chat_id=message.chat.id,
                message_id=status_msg_id,
            )
            sent_message_id = status_msg_id

            if user:
                rec_id = add_video_record(
                    chat_id=message.chat.id,
                    user_id=user.id,
                    username=display_name,
                    url=url,
                    video_source_id=video_source_id,
                    direct_url=direct_url_to_save,
                )
                if rec_id > 0 and sent_message_id:
                    update_bot_message_id(rec_id, sent_message_id)
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
    """Track reactions added/removed on bot messages."""
    chat_id = event.chat.id
    message_id = event.message_id

    new_reactions = []
    for reaction in (event.new_reaction or []):
        emoji = None
        if hasattr(reaction, "emoji"):
            emoji = reaction.emoji
        elif hasattr(reaction, "custom_emoji_id"):
            emoji = f"custom:{reaction.custom_emoji_id}"
        if emoji:
            new_reactions.append({
                "user_id": event.user.id if event.user else 0,
                "emoji": emoji,
            })

    sync_reactions(chat_id, message_id, new_reactions)


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

async def main():
    global _http_session
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)
    load_data()
    init_db()
    logging.basicConfig(level=logging.INFO)
    # Pre-create the shared HTTP session so the first request doesn't pay the connection cost
    _http_session = aiohttp.ClientSession()
    print("Бот запущен!")
    try:
        # Запускаем polling, игнорируем накопившиеся апдейты
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=["message", "message_reaction"])
    finally:
        if _http_session and not _http_session.closed:
            await _http_session.close()

if __name__ == "__main__":
    asyncio.run(main())
