from __future__ import annotations

import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, String, Text, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


class Thing(Base):
    __tablename__ = "things2"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)

    locations: Mapped[list[ThingLocation]] = relationship(
        back_populates="thing",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "things2_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "things2_metadata_idx",
            "metadata",
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:
        return f"<Thing id={self.id!r}>"


class ThingLocation(Base):
    __tablename__ = "things2_location"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    location: Mapped[str] = mapped_column(String)
    thing_id: Mapped[str] = mapped_column(ForeignKey("things2.id", ondelete="CASCADE"))

    thing: Mapped[Thing] = relationship(back_populates="locations")

    def __repr__(self) -> str:
        return f"<ThingLocation thing_id={self.thing_id!r} location={self.location!r}>"


class PGVectorCollection:
    """Synchronous wrapper around the pgvector-backed Things table.

    Handles upsert, similarity search, and location-based filtering.
    All methods open and close their own session so callers don't need to
    manage session lifetimes.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self.Session = sessionmaker(engine)

        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        Base.metadata.create_all(engine)

    # Write operations ======================================================

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert things and their locations.

        If a Thing with the given id already exists it is updated in-place;
        otherwise a new row is inserted. The location is similarly upserted
        on the companion ThingLocation row.
        """
        with self.Session() as session:
            for i, e, m in zip(ids, embeddings, metadatas):
                thing = session.get(Thing, i)
                if thing is None:
                    thing = Thing(id=i, embedding=e, metadata_=m)
                    session.add(thing)
                else:
                    thing.embedding = e
                    thing.metadata_ = m

                location_value = m.get("location", "unknown")
                existing_loc = session.scalars(
                    select(ThingLocation).where(ThingLocation.thing_id == i)
                ).first()
                if existing_loc:
                    existing_loc.location = location_value
                else:
                    session.add(
                        ThingLocation(
                            id=str(uuid.uuid4()),
                            location=location_value,
                            thing_id=i,
                        )
                    )

            session.commit()

    def update_metadata(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Replace the metadata JSONB blob for each given id."""
        with self.Session() as session:
            for i, m in zip(ids, metadatas):
                thing = session.get(Thing, i)
                if thing:
                    thing.metadata_ = m
            session.commit()

    def update_location(self, ids: list[str], locations: list[str]) -> None:
        """Update the location string on the ThingLocation row."""
        with self.Session() as session:
            for i, loc in zip(ids, locations):
                existing_loc = session.scalars(
                    select(ThingLocation).where(ThingLocation.thing_id == i)
                ).first()
                if existing_loc:
                    existing_loc.location = loc
            session.commit()

    def clear(self) -> None:
        """Truncate all rows from the things table (cascades to locations)."""
        with self.engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE things2 CASCADE"))

    # Read operations ========================================================

    def count(self) -> int:
        """Return the total number of registered things."""
        with self.Session() as session:
            return session.scalar(select(func.count()).select_from(Thing)) or 0

    def get(self) -> dict[str, list]:
        """Return all things as parallel id and metadata lists."""
        with self.Session() as session:
            things = session.scalars(select(Thing)).all()
            return {
                "ids": [t.id for t in things],
                "metadatas": [t.metadata_ for t in things],
            }

    def query_loc(self, location: str) -> dict[str, list]:
        """Return all things whose location matches *location* exactly."""
        with self.Session() as session:
            rows = session.execute(
                select(Thing.id, Thing.metadata_, ThingLocation.location)
                .join(ThingLocation, Thing.id == ThingLocation.thing_id)
                .where(ThingLocation.location == location)
            ).all()

            return {
                "ids": [[r.id for r in rows]],
                "metadatas": [[r.metadata_ for r in rows]],
                "locations": [[r.location for r in rows]],
            }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
    ) -> dict[str, list]:
        """Cosine-similarity search against the embedding column.

        Returns the top *n_results* nearest neighbours in the same
        ChromaDB-style nested-list format used throughout the project.
        """
        query_vec = query_embeddings[0]
        with self.Session() as session:
            distance = Thing.embedding.cosine_distance(query_vec)
            rows = session.execute(
                select(Thing.id, Thing.metadata_, distance.label("distance"))
                .order_by(distance)
                .limit(n_results)
            ).all()

            return {
                "ids": [[r.id for r in rows]],
                "metadatas": [[r.metadata_ for r in rows]],
                "distances": [[float(r.distance) for r in rows]],
            }