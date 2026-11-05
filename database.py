import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, String, DateTime, Float,
    LargeBinary, Text, func
)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = "video_stats.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class VideoRecord(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, nullable=False, index=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    username = Column(String, nullable=False)
    url = Column(String, nullable=False)
    bot_message_id = Column(BigInteger, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Deduplication & analytics fields
    video_source_id = Column(String, nullable=True, index=True)   # "tiktok:7234567890"
    direct_url = Column(String, nullable=True)                     # Прямая ссылка на медиа
    telegram_file_id = Column(String, nullable=True, index=True)  # file_id от Telegram


class ReactionRecord(Base):
    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, nullable=False, index=True)
    message_id = Column(BigInteger, nullable=False, index=True)
    user_id = Column(BigInteger, nullable=False)
    emoji = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class VideoEmbedding(Base):
    """Эмбеддинг видео + очередь обработки в одной таблице.

    Очередь живёт в БД, а не в памяти: рестарт бота не теряет задачи.
    """
    __tablename__ = "video_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, nullable=False, unique=True, index=True)
    chat_id = Column(BigInteger, nullable=False, index=True)

    # pending | processing | done | failed | skipped
    status = Column(String, nullable=False, default="pending", index=True)

    media_path = Column(String, nullable=True)   # файл в embed_queue/, удаляется после обработки
    kind = Column(String, nullable=True)         # video | image

    transcript = Column(Text, nullable=True)
    visual_caption = Column(Text, nullable=True)
    doc_text = Column(Text, nullable=True)

    embedding = Column(LargeBinary, nullable=True)   # float32.tobytes()
    embedding_dim = Column(Integer, nullable=True)
    embed_model = Column(String, nullable=True)
    # Как получен вектор: модель, размерность и постобработка
    # ("google/gemini-embedding-2@768+mu1"). Векторы с разными рецептами
    # несравнимы, поэтому индекс грузит только совпадающие с текущим.
    vector_recipe = Column(String, nullable=True, index=True)

    attempts = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    cost_usd = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)


def _backup_once():
    """Разовый бэкап БД перед первой миграцией под эмбеддинги.

    Делается только если таблицы video_embeddings ещё нет — то есть ровно
    один раз, при обновлении бота.
    """
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='video_embeddings'"
        ).fetchone()
    finally:
        conn.close()

    if exists:
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{DB_PATH}.bak-{stamp}"
    try:
        shutil.copy2(DB_PATH, backup_path)
        logging.info(f"Бэкап базы перед миграцией: {backup_path}")
    except Exception as e:
        logging.error(f"Не удалось сделать бэкап базы: {e}")


def _migrate_db():
    """Add new columns to existing tables if they don't exist (safe upgrade)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(videos)")
    existing = {row[1] for row in cursor.fetchall()}

    migrations = {
        "video_source_id": "TEXT",
        "direct_url": "TEXT",
        "telegram_file_id": "TEXT",
    }

    for col, col_type in migrations.items():
        if col not in existing:
            cursor.execute(f"ALTER TABLE videos ADD COLUMN {col} {col_type}")
            logging.info(f"Migration: added column '{col}' to videos table")

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS ix_videos_video_source_id ON videos (video_source_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS ix_videos_telegram_file_id ON videos (telegram_file_id)"
    )

    # Реакции обновляются по одному пользователю за раз — под этот запрос
    # нужен составной индекс, иначе на каждой реакции идёт скан таблицы.
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS ix_reactions_msg_user "
        "ON reactions (chat_id, message_id, user_id)"
    )

    # Эмбеддинги: таблицу создаёт create_all(), здесь только составной индекс
    # под основной запрос воркера (очередь) и под загрузку индекса по чату.
    cursor.execute("PRAGMA table_info(video_embeddings)")
    vemb_cols = {row[1] for row in cursor.fetchall()}
    if vemb_cols:
        if "vector_recipe" not in vemb_cols:
            cursor.execute("ALTER TABLE video_embeddings ADD COLUMN vector_recipe TEXT")
            logging.info("Migration: added column 'vector_recipe' to video_embeddings")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_vemb_recipe "
            "ON video_embeddings (chat_id, vector_recipe)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_vemb_status_created "
            "ON video_embeddings (status, created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_vemb_chat_status "
            "ON video_embeddings (chat_id, status)"
        )

    conn.commit()
    conn.close()


def init_db():
    """Create all tables if they don't exist."""
    _backup_once()
    Base.metadata.create_all(engine)
    _migrate_db()
    logging.info("Database initialized.")


def add_video_record(chat_id: int, user_id: int, username: str, url: str,
                     video_source_id: str = None, direct_url: str = None) -> int:
    """Add a video record and return its ID."""
    session = Session()
    try:
        record = VideoRecord(
            chat_id=chat_id,
            user_id=user_id,
            username=username,
            url=url,
            video_source_id=video_source_id,
            direct_url=direct_url,
        )
        session.add(record)
        session.commit()
        return record.id
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to add video record: {e}")
        return -1
    finally:
        session.close()


def update_bot_message_id(record_id: int, bot_message_id: int,
                          telegram_file_id: str = None):
    """Update the bot_message_id (and optionally telegram_file_id) after the video is sent."""
    session = Session()
    try:
        record = session.query(VideoRecord).filter_by(id=record_id).first()
        if record:
            record.bot_message_id = bot_message_id
            if telegram_file_id:
                record.telegram_file_id = telegram_file_id
            session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to update bot_message_id: {e}")
    finally:
        session.close()


def find_duplicate(chat_id: int, video_source_id: str) -> dict | None:
    """Find the first record with this video_source_id in the chat.
    Returns dict with user info and message link data, or None."""
    if not video_source_id:
        return None
    session = Session()
    try:
        base = (
            session.query(VideoRecord)
            .filter_by(chat_id=chat_id, video_source_id=video_source_id)
            .order_by(VideoRecord.created_at.asc(), VideoRecord.id.asc())
        )
        # Первое видео, на которое можно сослаться (сообщение бота известно);
        # если таких нет — просто самая ранняя запись
        record = (
            base.filter(VideoRecord.bot_message_id.isnot(None)).first()
            or base.first()
        )
        if record:
            return {
                "user_id": record.user_id,
                "username": record.username,
                "bot_message_id": record.bot_message_id,
                "url": record.url,
                "created_at": record.created_at,
            }
        return None
    except Exception as e:
        logging.error(f"Failed to find duplicate: {e}")
        return None
    finally:
        session.close()


def sync_user_reactions(chat_id: int, message_id: int, user_id: int,
                        emojis: list[str], reacted_at: datetime = None):
    """Обновить реакции ОДНОГО пользователя на одном сообщении.

    Telegram присылает MessageReactionUpdated отдельно на каждого человека:
    в событии лежит только его личный набор реакций. Поэтому трогать здесь
    можно исключительно его строки — иначе реакция одного стирает реакции
    всех остальных.

    Пустой emojis — это штатная ситуация: человек снял реакцию. Его строки
    надо удалить и ничего не вставлять.
    """
    session = Session()
    try:
        session.query(ReactionRecord).filter_by(
            chat_id=chat_id, message_id=message_id, user_id=user_id
        ).delete()

        # Telegram отдаёт время с таймзоной, остальная база живёт в наивном
        # UTC. Без приведения сравнения дат молча ломаются.
        stamp = reacted_at or datetime.utcnow()
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)

        for emoji in emojis:
            session.add(ReactionRecord(
                chat_id=chat_id,
                message_id=message_id,
                user_id=user_id,
                emoji=emoji,
                created_at=stamp,
            ))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to sync reactions for user {user_id}: {e}")
    finally:
        session.close()


def get_chat_stats(chat_id: int) -> list[dict]:
    """
    Get aggregated stats for a chat.
    Returns list of dicts:
      {"user_id", "username", "video_count", "reaction_count"}
    sorted by video_count descending.
    """
    session = Session()
    try:
        # Sub-query 1: count videos per user (one row per user_id)
        video_sub = (
            session.query(
                VideoRecord.user_id.label("uid"),
                func.max(VideoRecord.username).label("uname"),
                func.count(VideoRecord.id).label("vcnt"),
            )
            .filter(VideoRecord.chat_id == chat_id)
            .group_by(VideoRecord.user_id)
            .subquery()
        )

        # Sub-query 2: count reactions per video owner (one row per user_id)
        # Join reactions to videos via (chat_id, message_id) -> (chat_id, bot_message_id)
        reaction_sub = (
            session.query(
                VideoRecord.user_id.label("owner_id"),
                func.count(ReactionRecord.id).label("rcnt"),
            )
            .join(
                ReactionRecord,
                (ReactionRecord.chat_id == VideoRecord.chat_id)
                & (ReactionRecord.message_id == VideoRecord.bot_message_id),
            )
            .filter(VideoRecord.chat_id == chat_id)
            .group_by(VideoRecord.user_id)
            .subquery()
        )

        # Main query: join the two subqueries (no row multiplication)
        rows = (
            session.query(
                video_sub.c.uid.label("user_id"),
                video_sub.c.uname.label("username"),
                video_sub.c.vcnt.label("video_count"),
                func.coalesce(reaction_sub.c.rcnt, 0).label("reaction_count"),
            )
            .outerjoin(reaction_sub, reaction_sub.c.owner_id == video_sub.c.uid)
            .order_by(video_sub.c.vcnt.desc())
            .all()
        )

        return [
            {
                "user_id": row.user_id,
                "username": row.username,
                "video_count": row.video_count,
                "reaction_count": row.reaction_count,
            }
            for row in rows
        ]
    except Exception as e:
        logging.error(f"Failed to get chat stats: {e}")
        return []
    finally:
        session.close()


# =====================================================================
#  ЭМБЕДДИНГИ: очередь, результаты, статистика
# =====================================================================

MAX_EMBED_ATTEMPTS = 3


def enqueue_embedding(video_id: int, chat_id: int, media_path: str, kind: str) -> bool:
    """Ставит видео в очередь на эмбеддинг. Повторная постановка игнорируется."""
    session = Session()
    try:
        existing = session.query(VideoEmbedding).filter_by(video_id=video_id).first()
        if existing:
            return False
        session.add(VideoEmbedding(
            video_id=video_id,
            chat_id=chat_id,
            media_path=media_path,
            kind=kind,
            status="pending",
        ))
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to enqueue embedding for video {video_id}: {e}")
        return False
    finally:
        session.close()


def reset_stuck_jobs() -> int:
    """Возвращает в очередь задачи, зависшие в processing после падения бота."""
    session = Session()
    try:
        count = (
            session.query(VideoEmbedding)
            .filter_by(status="processing")
            .update({"status": "pending"}, synchronize_session=False)
        )
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to reset stuck jobs: {e}")
        return 0
    finally:
        session.close()


def next_pending_job() -> dict | None:
    """Берёт следующую задачу и помечает её processing (воркер один, гонок нет)."""
    session = Session()
    try:
        job = (
            session.query(VideoEmbedding)
            .filter_by(status="pending")
            .order_by(VideoEmbedding.created_at.asc())
            .first()
        )
        if not job:
            return None

        job.status = "processing"
        job.attempts = (job.attempts or 0) + 1
        session.commit()

        return {
            "id": job.id,
            "video_id": job.video_id,
            "chat_id": job.chat_id,
            "media_path": job.media_path,
            "kind": job.kind,
            "attempts": job.attempts,
        }
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to fetch next job: {e}")
        return None
    finally:
        session.close()


def save_embedding_result(job_id: int, transcript: str, visual_caption: str,
                          doc_text: str, embedding: bytes, dim: int,
                          model: str, cost_usd: float,
                          vector_recipe: str = None) -> None:
    session = Session()
    try:
        job = session.query(VideoEmbedding).filter_by(id=job_id).first()
        if not job:
            return
        job.status = "done"
        job.transcript = transcript or None
        job.visual_caption = visual_caption or None
        job.doc_text = doc_text
        job.embedding = embedding
        job.embedding_dim = dim
        job.embed_model = model
        job.vector_recipe = vector_recipe
        job.cost_usd = (job.cost_usd or 0.0) + (cost_usd or 0.0)
        job.media_path = None
        job.error = None
        job.processed_at = datetime.utcnow()
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to save embedding result for job {job_id}: {e}")
    finally:
        session.close()


def mark_job_failed(job_id: int, error: str, cost_usd: float = 0.0) -> bool:
    """Помечает задачу неудачной. Возвращает True, если попытки исчерпаны."""
    session = Session()
    try:
        job = session.query(VideoEmbedding).filter_by(id=job_id).first()
        if not job:
            return False

        job.error = (error or "")[:500]
        job.cost_usd = (job.cost_usd or 0.0) + (cost_usd or 0.0)
        exhausted = (job.attempts or 0) >= MAX_EMBED_ATTEMPTS
        job.status = "failed" if exhausted else "pending"
        if exhausted:
            job.processed_at = datetime.utcnow()
        session.commit()
        return exhausted
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to mark job {job_id} failed: {e}")
        return False
    finally:
        session.close()


def mark_job_skipped(job_id: int, reason: str, cost_usd: float = 0.0) -> None:
    """Задача пропущена штатно (нет файла, нет текста) — повторять не нужно.

    Стоимость всё равно записываем: VLM мог отработать и вернуть пустоту,
    но деньги за него уже списаны и должны попасть в месячный лимит.
    """
    session = Session()
    try:
        job = session.query(VideoEmbedding).filter_by(id=job_id).first()
        if not job:
            return
        job.status = "skipped"
        job.error = (reason or "")[:500]
        job.cost_usd = (job.cost_usd or 0.0) + (cost_usd or 0.0)
        job.media_path = None
        job.processed_at = datetime.utcnow()
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to mark job {job_id} skipped: {e}")
    finally:
        session.close()


def load_embeddings(chat_id: int = None) -> list[dict]:
    """Все готовые эмбеддинги для загрузки поискового индекса в память."""
    session = Session()
    try:
        query = session.query(
            VideoEmbedding.video_id,
            VideoEmbedding.chat_id,
            VideoEmbedding.embedding,
            VideoEmbedding.embedding_dim,
            VideoEmbedding.vector_recipe,
        ).filter(VideoEmbedding.status == "done", VideoEmbedding.embedding.isnot(None))

        if chat_id is not None:
            query = query.filter(VideoEmbedding.chat_id == chat_id)

        return [
            {
                "video_id": row.video_id,
                "chat_id": row.chat_id,
                "embedding": row.embedding,
                "dim": row.embedding_dim,
                "recipe": row.vector_recipe,
            }
            for row in query.all()
        ]
    except Exception as e:
        logging.error(f"Failed to load embeddings: {e}")
        return []
    finally:
        session.close()


def get_embedding_details(video_ids: list[int]) -> dict[int, dict]:
    """Тексты и метаданные найденных видео — для выдачи /find."""
    if not video_ids:
        return {}
    session = Session()
    try:
        rows = (
            session.query(
                VideoEmbedding.video_id,
                VideoEmbedding.transcript,
                VideoEmbedding.visual_caption,
                VideoRecord.url,
                VideoRecord.username,
                VideoRecord.bot_message_id,
                VideoRecord.created_at,
            )
            .join(VideoRecord, VideoRecord.id == VideoEmbedding.video_id)
            .filter(VideoEmbedding.video_id.in_(video_ids))
            .all()
        )
        return {
            row.video_id: {
                "transcript": row.transcript,
                "visual_caption": row.visual_caption,
                "url": row.url,
                "username": row.username,
                "bot_message_id": row.bot_message_id,
                "created_at": row.created_at,
            }
            for row in rows
        }
    except Exception as e:
        logging.error(f"Failed to get embedding details: {e}")
        return {}
    finally:
        session.close()


def get_video_record_by_message(chat_id: int, bot_message_id: int) -> dict | None:
    """Запись о видео по сообщению бота (вместе со статусом индексации) — для /info."""
    session = Session()
    try:
        row = (
            session.query(VideoRecord, VideoEmbedding)
            .outerjoin(VideoEmbedding, VideoEmbedding.video_id == VideoRecord.id)
            .filter(
                VideoRecord.chat_id == chat_id,
                VideoRecord.bot_message_id == bot_message_id,
            )
            .first()
        )
        if not row:
            return None
        record, job = row
        return {
            "id": record.id,
            "url": record.url,
            "video_source_id": record.video_source_id,
            "direct_url": record.direct_url,
            "username": record.username,
            "created_at": record.created_at,
            "embed_status": job.status if job else None,
            "transcript": job.transcript if job else None,
            "visual_caption": job.visual_caption if job else None,
        }
    except Exception as e:
        logging.error(f"Failed to get video record by message: {e}")
        return None
    finally:
        session.close()


def get_message_reaction_count(chat_id: int, message_id: int) -> int:
    """Сколько реакций собрало конкретное сообщение бота."""
    session = Session()
    try:
        return (
            session.query(func.count(ReactionRecord.id))
            .filter(
                ReactionRecord.chat_id == chat_id,
                ReactionRecord.message_id == message_id,
            )
            .scalar() or 0
        )
    except Exception as e:
        logging.error(f"Failed to count reactions for message {message_id}: {e}")
        return 0
    finally:
        session.close()


def get_video_id_by_message(chat_id: int, bot_message_id: int) -> int | None:
    """video_id по сообщению бота — для /similar в ответ на видео."""
    session = Session()
    try:
        record = (
            session.query(VideoRecord)
            .filter_by(chat_id=chat_id, bot_message_id=bot_message_id)
            .first()
        )
        return record.id if record else None
    except Exception as e:
        logging.error(f"Failed to resolve video by message: {e}")
        return None
    finally:
        session.close()


def get_embedding_vector(video_id: int) -> bytes | None:
    session = Session()
    try:
        row = (
            session.query(VideoEmbedding.embedding)
            .filter_by(video_id=video_id, status="done")
            .first()
        )
        return row.embedding if row else None
    except Exception as e:
        logging.error(f"Failed to get embedding vector: {e}")
        return None
    finally:
        session.close()


def get_embedding_stats(chat_id: int) -> dict:
    """Сводка для /embed_status: сколько видео, сколько из них в индексе."""
    session = Session()
    try:
        total_videos = (
            session.query(func.count(VideoRecord.id))
            .filter(VideoRecord.chat_id == chat_id).scalar() or 0
        )

        by_status = dict(
            session.query(VideoEmbedding.status, func.count(VideoEmbedding.id))
            .filter(VideoEmbedding.chat_id == chat_id)
            .group_by(VideoEmbedding.status).all()
        )

        chat_cost = (
            session.query(func.sum(VideoEmbedding.cost_usd))
            .filter(VideoEmbedding.chat_id == chat_id).scalar() or 0.0
        )

        return {
            "total_videos": total_videos,
            "done": by_status.get("done", 0),
            "pending": by_status.get("pending", 0),
            "processing": by_status.get("processing", 0),
            "failed": by_status.get("failed", 0),
            "skipped": by_status.get("skipped", 0),
            "chat_cost_usd": float(chat_cost),
        }
    except Exception as e:
        logging.error(f"Failed to get embedding stats: {e}")
        return {}
    finally:
        session.close()


def get_month_spend() -> float:
    """Потрачено на эмбеддинги с начала текущего месяца (по всем чатам)."""
    session = Session()
    try:
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)
        total = (
            session.query(func.sum(VideoEmbedding.cost_usd))
            .filter(VideoEmbedding.created_at >= month_start).scalar()
        )
        return float(total or 0.0)
    except Exception as e:
        logging.error(f"Failed to get month spend: {e}")
        return 0.0
    finally:
        session.close()


def retry_failed_jobs(chat_id: int = None) -> int:
    """Возвращает упавшие задачи в очередь и обнуляет счётчик попыток."""
    session = Session()
    try:
        query = session.query(VideoEmbedding).filter(VideoEmbedding.status == "failed")
        if chat_id is not None:
            query = query.filter(VideoEmbedding.chat_id == chat_id)

        # Без файла повторять нечего — такие задачи оставляем в failed
        query = query.filter(VideoEmbedding.media_path.isnot(None))

        count = query.update(
            {"status": "pending", "attempts": 0, "error": None},
            synchronize_session=False,
        )
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to retry jobs: {e}")
        return 0
    finally:
        session.close()
