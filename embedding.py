"""Семантический поиск по видео: очередь, обогащение текстом, эмбеддинги, индекс.

Как это устроено
----------------
1. process_url отправил видео в чат -> файл ПЕРЕНОСИТСЯ в embed_queue/ и
   ставится задача в БД. Путь отправки при этом не задерживается ни на миллисекунду.
2. Фоновый воркер (один, чтобы не отбирать CPU у скачивания) разбирает очередь:
   ffmpeg -> VAD -> Whisper -> при необходимости VLM -> эмбеддинг -> БД.
3. Векторы лежат в БД как BLOB и держатся в памяти numpy-матрицей по чатам.
   Для пары тысяч роликов это единицы мегабайт, векторная БД не нужна.

Очередь живёт в БД, а не в памяти: рестарт бота не теряет задачи.
Все API идут через OpenRouter — один ключ на транскрипцию, описание и эмбеддинги.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime

import requests

# numpy нужен только для поиска. Импорт мягкий: если на сервере его нет,
# бот должен продолжать качать и отправлять видео как раньше, а не падать
# при старте. git pull не имеет права сломать работающего бота.
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:  # pragma: no cover
    np = None
    _NUMPY_OK = False

import transcription
from database import (
    enqueue_embedding, get_embedding_stats, get_embedding_vector,
    get_month_spend, load_embeddings, mark_job_failed, mark_job_skipped,
    next_pending_job, reset_stuck_jobs, save_embedding_result,
)

log = logging.getLogger(__name__)

# =====================================================================
#  КОНФИГУРАЦИЯ
# =====================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBED_MODEL = os.getenv("EMBED_MODEL", "google/gemini-embedding-2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))       # Матрёшка: 128..3072
VLM_MODEL = os.getenv("VLM_MODEL", "google/gemini-3.5-flash-lite")
VLM_FRAMES = int(os.getenv("VLM_FRAMES", "4"))
FRAME_MAX_EDGE = 768

# 300 токенов не хватало на карточку из шести полей, а модели с обязательным
# reasoning съедали лимит незаметно и отдавали текст, оборванный на полуслове.
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "2000"))
_VLM_SEND_REASONING = os.getenv("VLM_REASONING", "1") not in ("0", "false", "no")

# Описание кадров даёт больший прирост точности поиска, чем любой другой шаг,
# поэтому по умолчанию считаем его для всех роликов. no_speech экономит около
# половины расхода ценой примерно 15 п.п. recall@1.
VLM_POLICY = os.getenv("VLM_POLICY", "always")       # always | no_speech | never

MIN_SPEECH_SEC = 1.0

QUEUE_DIR = "embed_queue"
MAX_QUEUE_MB = int(os.getenv("EMBED_MAX_QUEUE_MB", "2000"))

MONTHLY_LIMIT_USD = float(os.getenv("EMBED_MONTHLY_LIMIT_USD", "5.0"))

# Порог автоматического уведомления о похожем видео при отправке.
# 1.0 = практически никогда (косинус до единицы не дотягивает даже у копий).
# Понизь до ~0.85, чтобы бот начал сам подписывать «похоже на это».
SIMILAR_NOTIFY_THRESHOLD = float(os.getenv("EMBED_SIMILAR_THRESHOLD", "1.0"))

FIND_TOP_N = 5

# Порог подобран замером на 327 запросах ПОСЛЕ снятия общей компоненты (см.
# _postprocess): 0.20 глушит 9 бессмысленных запросов из 20 ценой 0.3%
# правильных ответов. Честного «ничего не найдено» на этой шкале не существует —
# классы правильных и мусорных ответов перекрываются, — так что это отсечка
# заведомого мусора, а не гарантия.
FIND_MIN_SCORE = float(os.getenv("EMBED_FIND_MIN_SCORE", "0.20"))

# Константа модели для снятия общей (анизотропной) компоненты векторов.
# Файл лежит рядом с этим модулем; без него поиск работает, но заметно хуже.
MU_FILENAME = "isotropy_mu_768.npy"

# Цена текстовых токенов gemini-embedding-2, $/токен — на случай, если
# OpenRouter не вернул фактическую стоимость в usage.
EMBED_PRICE_PER_TOKEN = 0.20 / 1_000_000

# Сколько неудач подряд до личного сообщения админу
FAILURE_ALERT_THRESHOLD = 3

_NO_WINDOW = {}
if hasattr(subprocess, "CREATE_NO_WINDOW"):
    _NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW}


def is_configured() -> bool:
    """Готова ли подсистема к работе. False -> бот работает как до обновления."""
    return bool(OPENROUTER_API_KEY) and _NUMPY_OK


def unavailable_reason() -> str | None:
    """Человекочитаемая причина, почему поиск не работает."""
    if not _NUMPY_OK:
        return "не установлен numpy (pip install numpy)"
    if not OPENROUTER_API_KEY:
        return "не задан OPENROUTER_API_KEY"
    return None


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "TikTok Save Bot",
    }


# =====================================================================
#  FFMPEG
# =====================================================================

def _run(cmd: list[str], timeout: int = 180):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, **_NO_WINDOW)
    except FileNotFoundError:
        log.error("Не найден %s — нужен ffmpeg в PATH", cmd[0])
    except subprocess.TimeoutExpired:
        log.error("Таймаут ffmpeg: %s", " ".join(cmd[:3]))
    except Exception as e:
        log.error("Ошибка ffmpeg: %s", e)
    return None


def _probe(path: str) -> dict:
    res = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ], timeout=60)

    info = {"duration": None, "has_audio": False}
    if not res or res.returncode != 0:
        return info
    try:
        data = json.loads(res.stdout.decode("utf-8", errors="replace"))
    except Exception:
        return info

    dur = data.get("format", {}).get("duration")
    if dur:
        try:
            info["duration"] = float(dur)
        except ValueError:
            pass
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            info["has_audio"] = True
            break
    return info


def _extract_audio(src: str, dst: str) -> bool:
    """16 kHz mono mp3 — оптимальный вход для Whisper."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    res = _run([
        "ffmpeg", "-y", "-i", src,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "64k",
        dst,
    ])
    return bool(res and res.returncode == 0
                and os.path.exists(dst) and os.path.getsize(dst) > 0)


def _extract_frames(src: str, out_dir: str, count: int, duration: float | None) -> list[str]:
    """Кадры на 10%..90% длительности — мимо чёрного первого кадра и заставки."""
    os.makedirs(out_dir, exist_ok=True)

    if not duration or duration <= 0:
        timestamps = [0.0]
    elif count == 1:
        timestamps = [duration * 0.5]
    else:
        step = 0.8 / (count - 1)
        timestamps = [duration * (0.1 + step * i) for i in range(count)]

    scale = f"scale='min({FRAME_MAX_EDGE},iw)':-2"
    frames = []
    for i, ts in enumerate(timestamps):
        dst = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        res = _run([
            "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", src,
            "-frames:v", "1", "-vf", scale, "-q:v", "3", dst,
        ], timeout=60)
        if res and res.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            frames.append(dst)
    return frames


def _prepare_image(src: str, dst: str) -> str | None:
    res = _run([
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale='min({FRAME_MAX_EDGE},iw)':-2", "-q:v", "3", dst,
    ], timeout=60)
    if res and res.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    return None


# =====================================================================
#  VAD — есть ли речь (локально, бесплатно)
# =====================================================================

def _speech_seconds(audio_path: str) -> float:
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        log.warning("faster-whisper не установлен — VAD пропущен")
        return 999.0

    try:
        audio = decode_audio(audio_path, sampling_rate=16000)
        segments = get_speech_timestamps(audio, VadOptions(
            threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=500,
        ))
        return sum(s["end"] - s["start"] for s in segments) / 16000.0
    except Exception as e:
        log.warning("VAD упал (%s) — считаю что речь есть", e)
        return 999.0


# =====================================================================
#  OPENROUTER
# =====================================================================

_send_dimensions = True


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


_MU = None
_MU_CHECKED = False


def _get_mu():
    """Ленивая загрузка константы центрирования.

    Именно ленивая и с мягким откатом: main.py импортирует этот модуль
    безусловно, и np.load на уровне модуля уронил бы бота при отсутствии
    numpy или файла — ровно то, от чего защищает мягкий импорт numpy выше.
    """
    global _MU, _MU_CHECKED
    if _MU_CHECKED:
        return _MU
    _MU_CHECKED = True

    if not _NUMPY_OK:
        return None

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MU_FILENAME)
    try:
        mu = np.load(path).astype(np.float32)
    except Exception as e:
        log.warning("Не удалось загрузить %s (%s). Поиск будет работать заметно "
                    "хуже — верни файл из репозитория.", MU_FILENAME, e)
        return None

    if mu.ndim != 1 or mu.shape[0] != EMBED_DIM:
        log.warning("%s рассчитан на другую размерность (%s против EMBED_DIM=%s) — "
                    "центрирование отключено", MU_FILENAME, mu.shape, EMBED_DIM)
        return None

    _MU = mu
    return _MU


def _postprocess(vector: np.ndarray) -> np.ndarray:
    """Нормализовать -> снять общую компоненту -> нормализовать снова.

    Векторы gemini-embedding-2 анизотропны: у любых двух документов косинус
    около 0.48 просто потому, что все они смотрят в одну сторону, и различать
    приходится на третьем знаке. Вычитание средней компоненты возвращает
    пространству разрешающую способность: несвязанные пары падают до ~0.24,
    тематические поднимаются наверх. Стоит ноль вызовов API.

    Применяется ОДИНАКОВО к документам и запросам — иначе они окажутся
    в разных пространствах и поиск сломается.
    """
    vector = _l2_normalize(vector)
    mu = _get_mu()
    if mu is None:
        return vector
    return _l2_normalize(vector - mu)


def vector_recipe() -> str:
    """Как именно получен вектор — пишется в БД рядом с ним.

    Векторы, посчитанные по разным рецептам, между собой несравнимы, поэтому
    индекс обязан их различать: иначе поиск молча выдаёт мусор.
    """
    base = f"{EMBED_MODEL}@{EMBED_DIM}"
    return f"{base}+mu1" if _get_mu() is not None else base


def embed_texts(texts: list[str]) -> tuple[list[np.ndarray], float]:
    """Список текстов -> (список векторов, суммарная стоимость).

    OpenRouter принимает массив строк одним запросом, и батчевый вектор
    побитово совпадает с одиночным — так что несколько текстов дешевле и
    быстрее слать вместе, чем по одному.
    """
    global _send_dimensions

    cleaned = [(t or "").strip() for t in texts]
    if not cleaned or any(not t for t in cleaned):
        raise ValueError("Пустой текст — нечего эмбеддить")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY не задан")

    url = f"{OPENROUTER_BASE_URL}/embeddings"
    last_error = None

    for attempt in range(4):
        body = {
            "model": EMBED_MODEL,
            "input": cleaned if len(cleaned) > 1 else cleaned[0],
            "encoding_format": "float",
        }
        if _send_dimensions:
            body["dimensions"] = EMBED_DIM

        try:
            response = requests.post(url, headers=_headers(), json=body, timeout=90)
        except requests.RequestException as e:
            last_error = e
            _sleep_backoff(attempt)
            continue

        if response.status_code == 200:
            payload = response.json()
            # Порядок в ответе не гарантирован — раскладываем по полю index
            items = sorted(payload["data"], key=lambda d: d.get("index", 0))
            if len(items) != len(cleaned):
                raise RuntimeError(
                    f"OpenRouter вернул {len(items)} векторов на {len(cleaned)} текстов"
                )

            vectors = []
            for item in items:
                vector = np.asarray(item["embedding"], dtype=np.float32)
                # gemini-embedding-2 обучен по Матрёшке — лишние измерения
                # можно просто отрезать, вектор потом перенормируется.
                if vector.shape[0] > EMBED_DIM:
                    vector = vector[:EMBED_DIM]
                vectors.append(_postprocess(vector))

            usage = payload.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            if not cost:
                tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
                cost = tokens * EMBED_PRICE_PER_TOKEN

            return vectors, cost

        if response.status_code == 400 and _send_dimensions:
            log.info("OpenRouter не принял dimensions — обрезаю вектор на своей стороне")
            _send_dimensions = False
            continue

        if response.status_code in (408, 429, 500, 502, 503):
            last_error = RuntimeError(f"HTTP {response.status_code}")
            _sleep_backoff(attempt)
            continue

        raise RuntimeError(f"OpenRouter embeddings HTTP {response.status_code}: {response.text[:200]}")

    raise RuntimeError(f"OpenRouter embeddings недоступен: {last_error}")


def embed_text(text: str) -> tuple[np.ndarray, float]:
    """Один текст -> (вектор float32, стоимость в USD)."""
    vectors, cost = embed_texts([text])
    return vectors[0], cost


def _sleep_backoff(attempt: int) -> None:
    import time
    time.sleep(2 ** attempt)


def transcribe(audio_path: str) -> tuple[str, float]:
    """Распознавание речи через выбранного провайдера (см. transcription.py).

    Отдельная обёртка нужна, чтобы воркер не знал, кто именно распознаёт,
    и чтобы её было удобно подменять в тестах.
    """
    return transcription.transcribe(audio_path)


# Промпт подобран эмпирически: на независимом наборе запросов он даёт
# +11 п.п. recall@1 против свободного описания абзацем. Ключевое — жёсткая
# структура из шести полей и прямой запрет на описание обстановки: старый
# промпт тратил половину ответа на стены, пол и «атмосферу», по которым
# никто не ищет, и делал все описания похожими друг на друга.
_VLM_PROMPT = """Это кадры короткого видео из чата с друзьями. Составь по ним поисковую карточку: по ней человек через месяц должен узнать ролик и найти его поиском.

Ответь строго этими шестью строками, без вступлений и без пустой болтовни:

Кто/что: люди, животные, предметы, персонажи — обычными словами, как назвал бы их обычный человек. Если у объекта есть и бытовое, и точное название, дай оба через запятую («велосипед, шоссейник»).
Действие: что происходит в кадре. Одно предложение.
Суть: о чём ролик, в чём шутка, новость или соль. Одно предложение.
Текст на экране: дословно все надписи, подписи, титры, реплики, цифры и элементы интерфейса. Если надписей нет — напиши «нет».
Названия: узнаваемые игры, фильмы, сериалы, соцсети, бренды, места, известные люди. Если ничего не узнаётся — «нет».
Теги: 10-14 коротких поисковых слов через запятую, включая общую категорию (например: рыбалка, ремонт, кофе, путешествия, музыка, авто).

Запрещено писать про: освещение, стены, пол, интерьер, качество съёмки, «атмосферу», «настроение», «создавая ощущение». Запрещены обороты «на кадрах», «на видео», «изображён», «запечатлён»."""

# Поля карточки — по ним же проверяем, что ответ не оборвался на полуслове
_CARD_FIELDS = ("Кто/что:", "Действие:", "Суть:", "Текст на экране:", "Названия:", "Теги:")


def _caption_looks_truncated(text: str, finish_reason: str | None) -> bool:
    """Оборвался ли ответ VLM.

    Проверяем ровно то, что действительно сигнализирует об обрыве: явный
    finish_reason='length' и отсутствие полей карточки. Признак usage.cost==0
    для этого не годится — при недоступности API стоимость тоже нулевая,
    и повтор ушёл бы в цикл на лежащем провайдере.
    """
    if finish_reason == "length":
        return True
    if not text:
        return False
    return sum(1 for field in _CARD_FIELDS if field in text) < 5


def caption_frames(frames: list[str]) -> tuple[str, float]:
    """Описание кадров через VLM. При недоступности возвращает ('', 0)."""
    if not frames:
        return "", 0.0

    content = [{"type": "text", "text": _VLM_PROMPT}]
    for frame in frames[:6]:
        try:
            with open(frame, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    if len(content) == 1:
        return "", 0.0

    total_cost = 0.0
    send_reasoning = _VLM_SEND_REASONING

    # Максимум одна повторная попытка: обрыв ответа бывает, но зацикливаться
    # на недоступном провайдере нельзя — воркер один и очередь встанет.
    for attempt in range(2):
        body = {
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": VLM_MAX_TOKENS,
            "temperature": 0.2,
            "usage": {"include": True},
        }
        if send_reasoning:
            # Некоторые модели (gemini-3.5-flash-lite) рассуждают всегда и
            # съедают лимит токенов молча. Просим минимум и не возвращать его.
            body["reasoning"] = {"effort": "low", "exclude": True}

        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=_headers(), json=body, timeout=180,
            )
        except requests.RequestException as e:
            log.warning("VLM недоступна: %s", e)
            return "", total_cost

        if response.status_code == 400 and send_reasoning:
            # Модель не принимает поле reasoning — повторяем без него
            log.info("VLM не поддерживает reasoning, повторяю без него")
            send_reasoning = False
            continue

        if response.status_code != 200:
            log.warning("VLM HTTP %s: %s", response.status_code, response.text[:200])
            return "", total_cost

        payload = response.json()
        total_cost += float((payload.get("usage") or {}).get("cost") or 0.0)

        try:
            choice = payload["choices"][0]
            text = (choice["message"]["content"] or "").strip()
            finish = choice.get("finish_reason")
        except (KeyError, IndexError):
            return "", total_cost

        if not _caption_looks_truncated(text, finish):
            return text, total_cost

        if attempt == 0:
            log.info("Описание оборвано (finish_reason=%s, %s симв.) — переспрашиваю",
                     finish, len(text))
        else:
            log.warning("Описание оборвано повторно, беру как есть")
            return text, total_cost

    return "", total_cost


# =====================================================================
#  СБОРКА ТЕКСТА
# =====================================================================

def build_doc(kind: str, transcript: str, caption: str, extra: dict | None = None) -> str:
    """Всё известное о ролике в один размеченный текст — его и эмбеддим."""
    lines = [f"Тип: {'видео' if kind == 'video' else 'изображение'}"]
    for key, value in (extra or {}).items():
        if value:
            lines.append(f"{key}: {value}")
    if caption:
        lines.append(f"На видео: {caption}")
    if transcript:
        lines.append(f"Речь: {transcript}")
    return "\n".join(lines)


# =====================================================================
#  ИНДЕКС В ПАМЯТИ
# =====================================================================

# chat_id -> {"ids": [video_id, ...], "matrix": np.ndarray (N, D)}
_index: dict[int, dict] = {}


def load_index() -> int:
    """Читает готовые эмбеддинги из БД в память. Вызывается при старте."""
    _index.clear()
    if not _NUMPY_OK:
        return 0

    rows = load_embeddings()
    current = vector_recipe()

    buckets: dict[int, list] = {}
    stale = 0
    for row in rows:
        # Векторы, посчитанные по другому рецепту (иная модель, размерность
        # или отсутствие центрирования), лежат в другом пространстве. Считать
        # по ним косинус — молча выдавать мусор, поэтому пропускаем.
        if (row.get("recipe") or "") != current:
            stale += 1
            continue
        try:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
        except Exception:
            continue
        if vector.size == 0:
            continue
        buckets.setdefault(row["chat_id"], []).append((row["video_id"], vector))

    if stale:
        log.warning(
            "Пропущено %s векторов с устаревшим рецептом (текущий: %s). "
            "Эти ролики не будут находиться, пока их не переиндексировать.",
            stale, current,
        )

    total = 0
    for chat_id, items in buckets.items():
        # Векторы разной длины (сменилась модель) в одном индексе бессмысленны —
        # оставляем самую распространённую размерность.
        dims = {}
        for _, vector in items:
            dims[vector.shape[0]] = dims.get(vector.shape[0], 0) + 1
        main_dim = max(dims, key=dims.get)
        if len(dims) > 1:
            log.warning("Чат %s: в индексе размерности %s, оставляю %sd",
                        chat_id, dims, main_dim)

        items = [(vid, vec) for vid, vec in items if vec.shape[0] == main_dim]
        if not items:
            continue

        _index[chat_id] = {
            "ids": [vid for vid, _ in items],
            "matrix": np.vstack([vec for _, vec in items]),
        }
        total += len(items)

    log.info("Поисковый индекс: %s векторов в %s чатах", total, len(_index))
    return total


def index_add(chat_id: int, video_id: int, vector: np.ndarray) -> None:
    bucket = _index.get(chat_id)
    if bucket is None:
        _index[chat_id] = {"ids": [video_id], "matrix": vector.reshape(1, -1)}
        return

    if bucket["matrix"].shape[1] != vector.shape[0]:
        log.warning("Размерность вектора не совпала с индексом чата %s — пропускаю", chat_id)
        return

    if video_id in bucket["ids"]:
        return

    bucket["ids"].append(video_id)
    bucket["matrix"] = np.vstack([bucket["matrix"], vector.reshape(1, -1)])


def index_size(chat_id: int) -> int:
    bucket = _index.get(chat_id)
    return len(bucket["ids"]) if bucket else 0


def _search_vector(chat_id: int, query: np.ndarray, top_n: int,
                   exclude_video_id: int = None) -> list[tuple[int, float]]:
    if not _NUMPY_OK:
        return []
    bucket = _index.get(chat_id)
    if not bucket or bucket["matrix"].shape[1] != query.shape[0]:
        return []

    # Векторы нормализованы при сохранении => косинус == скалярное произведение
    scores = bucket["matrix"] @ query
    order = np.argsort(-scores)

    results = []
    for i in order:
        video_id = bucket["ids"][int(i)]
        if exclude_video_id is not None and video_id == exclude_video_id:
            continue
        score = float(scores[int(i)])
        if score < FIND_MIN_SCORE:
            break
        results.append((video_id, score))
        if len(results) >= top_n:
            break
    return results


async def search_by_text(chat_id: int, query: str, top_n: int = FIND_TOP_N) -> list[tuple[int, float]]:
    vector, _cost = await asyncio.to_thread(embed_text, query)
    return _search_vector(chat_id, vector, top_n)


async def search_by_video(chat_id: int, video_id: int,
                          top_n: int = FIND_TOP_N) -> list[tuple[int, float]] | None:
    blob = await asyncio.to_thread(get_embedding_vector, video_id)
    if not blob:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    return _search_vector(chat_id, vector, top_n, exclude_video_id=video_id)


# =====================================================================
#  ОЧЕРЕДЬ
# =====================================================================

_wake = asyncio.Event()
_consecutive_failures = 0
_budget_alerted = False

# Колбэк для автоподсказки «похоже на это». Ставится из main.py, чтобы этот
# модуль не занимался форматированием сообщений и ссылками Telegram.
# Сигнатура: async (chat_id, new_video_id, similar_video_id, score) -> None
similar_callback = None


def wake_worker() -> None:
    """Будит воркер, не дожидаясь периодического опроса (после /embed_retry)."""
    _wake.set()


def _queue_size_mb() -> float:
    if not os.path.isdir(QUEUE_DIR):
        return 0.0
    total = 0
    for name in os.listdir(QUEUE_DIR):
        path = os.path.join(QUEUE_DIR, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total / (1024 * 1024)


def _enqueue_blocking(video_id: int, chat_id: int, src_path: str, kind: str) -> bool:
    if _queue_size_mb() > MAX_QUEUE_MB:
        log.warning("Очередь эмбеддингов больше %s МБ — пропускаю видео %s",
                    MAX_QUEUE_MB, video_id)
        return False

    os.makedirs(QUEUE_DIR, exist_ok=True)
    ext = os.path.splitext(src_path)[1] or ".mp4"
    dst_path = os.path.join(QUEUE_DIR, f"{video_id}_{uuid.uuid4().hex[:8]}{ext}")

    try:
        os.replace(src_path, dst_path)          # быстрый путь: тот же диск
    except OSError:
        try:
            shutil.move(src_path, dst_path)     # разные диски
        except Exception as e:
            log.error("Не удалось перенести файл в очередь: %s", e)
            return False

    if not enqueue_embedding(video_id, chat_id, dst_path, kind):
        _cleanup(dst_path)
        return False
    return True


async def enqueue_media(video_id: int, chat_id: int, src_path: str,
                        kind: str = "video") -> bool:
    """Переносит уже отправленный файл в очередь и ставит задачу.

    Именно ПЕРЕНОС, а не копия: файл уходит из temporary_videos/ раньше, чем
    его удалит finally в process_url, и не занимает диск дважды.

    Файловые операции уходят в поток, чтобы не задерживать event loop, а
    пробуждение воркера делается уже в цикле (asyncio.Event не потокобезопасен).
    """
    if not is_configured():
        return False
    if not src_path or not os.path.exists(src_path):
        return False

    ok = await asyncio.to_thread(_enqueue_blocking, video_id, chat_id, src_path, kind)
    if ok:
        _wake.set()
    return ok


def _cleanup(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        log.debug("Не удалось удалить %s: %s", path, e)


def _process_job_blocking(job: dict) -> dict:
    """Вся тяжёлая работа по одной задаче. Выполняется в отдельном потоке."""
    media_path = job["media_path"]
    kind = job.get("kind") or "video"
    work_dir = os.path.join(QUEUE_DIR, f"work_{job['id']}")
    cost = 0.0
    transcript, caption = "", ""

    try:
        if not media_path or not os.path.exists(media_path):
            return {"skip": "файл не найден", "cost": 0.0}

        os.makedirs(work_dir, exist_ok=True)

        if kind == "video":
            info = _probe(media_path)
            has_speech = False

            # Если распознавание не настроено — не тратим ffmpeg на аудио,
            # has_speech остаётся False и ролик уходит на описание кадров.
            if info["has_audio"] and transcription.is_available():
                audio_path = os.path.join(work_dir, "audio.mp3")
                if _extract_audio(media_path, audio_path):
                    speech = _speech_seconds(audio_path)
                    if speech >= MIN_SPEECH_SEC:
                        has_speech = True
                        transcript, asr_cost = transcribe(audio_path)
                        cost += asr_cost

            # VLM зовём когда речи нет или транскрипт вышел пустым
            want_vlm = (
                VLM_POLICY == "always"
                or (VLM_POLICY == "no_speech" and (not has_speech or not transcript))
            )
            if want_vlm and VLM_POLICY != "never":
                frames = _extract_frames(media_path, work_dir, VLM_FRAMES, info["duration"])
                if frames:
                    caption, vlm_cost = caption_frames(frames)
                    cost += vlm_cost
        else:
            if VLM_POLICY != "never":
                prepared = _prepare_image(media_path, os.path.join(work_dir, "image.jpg"))
                if prepared:
                    caption, vlm_cost = caption_frames([prepared])
                    cost += vlm_cost

        if not transcript and not caption:
            return {"skip": "не удалось извлечь ни речь, ни описание", "cost": cost}

        doc_text = build_doc(kind, transcript, caption)
        vector, embed_cost = embed_text(doc_text)
        cost += embed_cost

        return {
            "transcript": transcript,
            "caption": caption,
            "doc_text": doc_text,
            "vector": vector,
            "cost": cost,
        }
    except Exception as e:
        # Ошибку возвращаем, а не бросаем: иначе потерялась бы стоимость уже
        # оплаченных вызовов (Whisper/VLM могли отработать до сбоя на эмбеддинге),
        # и месячный лимит считался бы неверно.
        log.error("Обработка видео %s упала: %s", job.get("video_id"), e)
        return {"error": str(e), "cost": cost}
    finally:
        # Исходник НЕ трогаем: при временной ошибке (API лёг) задача вернётся
        # в очередь, и файл ещё понадобится. Его удаляет воркер — после успеха,
        # пропуска или явного отказа.
        shutil.rmtree(work_dir, ignore_errors=True)


async def _notify_admin(bot, admin_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id=admin_id, text=text, parse_mode="HTML")
    except Exception as e:
        log.error("Не удалось уведомить админа: %s", e)


async def worker(bot, admin_id: int) -> None:
    """Единственный фоновый воркер. Один — чтобы не отбирать CPU у скачивания."""
    global _consecutive_failures, _budget_alerted

    if not is_configured():
        log.warning("OPENROUTER_API_KEY не задан — воркер эмбеддингов не запущен")
        return

    requeued = await asyncio.to_thread(reset_stuck_jobs)
    if requeued:
        log.info("Возвращено в очередь после рестарта: %s задач", requeued)

    log.info("Воркер эмбеддингов запущен (модель %s, речь: %s)",
             EMBED_MODEL, transcription.provider_title())

    if not transcription.is_available() and VLM_POLICY == "never":
        log.warning(
            "Ни распознавание речи, ни описание кадров не включены — "
            "текста для эмбеддинга взять неоткуда, все видео будут пропущены"
        )

    while True:
        job = None
        try:
            # Лимит проверяем ДО того, как взять задачу: иначе она бы жгла
            # попытки и навсегда уходила в failed, хотя причина временная.
            # Видео при этом качаются и отправляются как обычно — встаёт
            # только обогащение, а очередь ждёт следующего месяца.
            if MONTHLY_LIMIT_USD > 0:
                spent = await asyncio.to_thread(get_month_spend)
                if spent >= MONTHLY_LIMIT_USD:
                    if not _budget_alerted:
                        _budget_alerted = True
                        await _notify_admin(bot, admin_id, (
                            f"⚠️ <b>Лимит на эмбеддинги исчерпан</b>\n"
                            f"Потрачено ${spent:.2f} из ${MONTHLY_LIMIT_USD:.2f}.\n"
                            f"Видео качаются и отправляются как обычно, "
                            f"но в поиск больше не попадают.\n"
                            f"Очередь не теряется — обработается, когда лимит "
                            f"обновится или вырастет (EMBED_MONTHLY_LIMIT_USD)."
                        ))
                    await asyncio.sleep(300)
                    continue

            job = await asyncio.to_thread(next_pending_job)

            if job is None:
                _wake.clear()
                try:
                    await asyncio.wait_for(_wake.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
                continue

            result = await asyncio.to_thread(_process_job_blocking, job)

            if "error" in result:
                exhausted = await asyncio.to_thread(
                    mark_job_failed, job["id"], result["error"], result["cost"]
                )
                if exhausted:
                    log.warning("Видео %s: попытки исчерпаны, нужен /embed_retry",
                                job["video_id"])
                _consecutive_failures += 1
                if _consecutive_failures == FAILURE_ALERT_THRESHOLD:
                    await _notify_admin(bot, admin_id, (
                        f"⚠️ <b>Эмбеддинги падают</b>\n"
                        f"{FAILURE_ALERT_THRESHOLD} неудачи подряд.\n"
                        f"Последняя ошибка: <code>{result['error'][:200]}</code>\n"
                        f"Проверь баланс OpenRouter и доступность моделей."
                    ))
                continue

            if "skip" in result:
                await asyncio.to_thread(
                    mark_job_skipped, job["id"], result["skip"], result["cost"]
                )
                _cleanup(job["media_path"])
                log.info("Видео %s пропущено: %s", job["video_id"], result["skip"])
                _consecutive_failures = 0
                continue

            vector = result["vector"]

            # Ищем похожее ДО добавления в индекс, чтобы не найти самого себя.
            # Порог 1.0 (по умолчанию) = автоподсказка выключена полностью.
            # Понизь EMBED_SIMILAR_THRESHOLD до ~0.85, чтобы включить.
            if similar_callback is not None and SIMILAR_NOTIFY_THRESHOLD < 1.0:
                try:
                    nearest = _search_vector(job["chat_id"], vector, 1)
                    if nearest and nearest[0][1] >= SIMILAR_NOTIFY_THRESHOLD:
                        await similar_callback(
                            job["chat_id"], job["video_id"], nearest[0][0], nearest[0][1]
                        )
                except Exception as e:
                    log.error("Автоподсказка «похоже» упала: %s", e)

            await asyncio.to_thread(
                save_embedding_result,
                job["id"], result["transcript"], result["caption"],
                result["doc_text"], vector.astype(np.float32).tobytes(),
                int(vector.shape[0]), EMBED_MODEL, result["cost"],
                vector_recipe(),
            )
            index_add(job["chat_id"], job["video_id"], vector)
            _cleanup(job["media_path"])

            _consecutive_failures = 0
            _budget_alerted = False
            log.info("Видео %s в индексе (стоимость $%.5f)",
                     job["video_id"], result["cost"])

        except asyncio.CancelledError:
            log.info("Воркер эмбеддингов остановлен")
            raise
        except Exception as e:
            log.error("Ошибка воркера: %s", e, exc_info=True)
            try:
                if job:
                    await asyncio.to_thread(mark_job_failed, job["id"], str(e))
            except Exception:
                pass

            _consecutive_failures += 1
            if _consecutive_failures == FAILURE_ALERT_THRESHOLD:
                await _notify_admin(bot, admin_id, (
                    f"⚠️ <b>Эмбеддинги падают</b>\n"
                    f"{FAILURE_ALERT_THRESHOLD} неудачи подряд.\n"
                    f"Последняя ошибка: <code>{str(e)[:200]}</code>\n"
                    f"Проверь баланс OpenRouter и доступность моделей."
                ))
            await asyncio.sleep(10)


# =====================================================================
#  СТАТУС
# =====================================================================

def status_report(chat_id: int) -> dict:
    stats = get_embedding_stats(chat_id)
    spent = get_month_spend()
    stats.update({
        "month_spend": spent,
        "month_limit": MONTHLY_LIMIT_USD,
        "month_left": max(0.0, MONTHLY_LIMIT_USD - spent) if MONTHLY_LIMIT_USD > 0 else None,
        "index_size": index_size(chat_id),
        "queue_mb": _queue_size_mb(),
        "embed_model": EMBED_MODEL,
        "vector_recipe": vector_recipe(),
        "centering": _get_mu() is not None,
        "asr_model": (f"{transcription.ASR_MODEL} ({transcription.provider_title()})"
                      if transcription.is_available() else "не настроено"),
        "vlm_model": VLM_MODEL if VLM_POLICY != "never" else "выключена",
        "vlm_policy": VLM_POLICY,
        "similar_threshold": SIMILAR_NOTIFY_THRESHOLD,
        "configured": is_configured(),
        "unavailable_reason": unavailable_reason(),
    })
    return stats
