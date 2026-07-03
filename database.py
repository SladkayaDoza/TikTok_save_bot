import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, String, DateTime, func
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


class ReactionRecord(Base):
    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, nullable=False, index=True)
    message_id = Column(BigInteger, nullable=False, index=True)
    user_id = Column(BigInteger, nullable=False)
    emoji = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    logging.info("Database initialized.")


def add_video_record(chat_id: int, user_id: int, username: str, url: str) -> int:
    """Add a video record and return its ID."""
    session = Session()
    try:
        record = VideoRecord(
            chat_id=chat_id,
            user_id=user_id,
            username=username,
            url=url,
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


def update_bot_message_id(record_id: int, bot_message_id: int):
    """Update the bot_message_id after the video is sent."""
    session = Session()
    try:
        record = session.query(VideoRecord).filter_by(id=record_id).first()
        if record:
            record.bot_message_id = bot_message_id
            session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to update bot_message_id: {e}")
    finally:
        session.close()


def sync_reactions(chat_id: int, message_id: int, reactions: list[dict]):
    """
    Sync reactions for a bot message.
    reactions: list of {"user_id": int, "emoji": str}
    """
    session = Session()
    try:
        # Delete old reactions for this message
        session.query(ReactionRecord).filter_by(
            chat_id=chat_id, message_id=message_id
        ).delete()

        # Insert new reactions
        for r in reactions:
            session.add(ReactionRecord(
                chat_id=chat_id,
                message_id=message_id,
                user_id=r["user_id"],
                emoji=r["emoji"],
            ))
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to sync reactions: {e}")
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
