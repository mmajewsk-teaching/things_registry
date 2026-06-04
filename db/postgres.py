from sqlalchemy import ( Text, String, Index, ForeignKey,select,text)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker,DeclarativeBase,Mapped,mapped_column,relationship
import uuid

class Base(DeclarativeBase):
    pass

class Thing(Base):
    __tablename__ = "things2"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding: Mapped[list] = mapped_column(Vector(384))
    metadata_: Mapped[dict] = mapped_column(JSONB)

    locations: Mapped[list["ThingLocation"]] = relationship(
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
            "metadata_",
            postgresql_using="gin",
        ),
    )

class ThingLocation(Base):
    __tablename__ = "things2_location"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    location: Mapped[str] = mapped_column(String)

    thing_id: Mapped[str] = mapped_column(
        ForeignKey("things2.id", ondelete="CASCADE")
    )

    thing: Mapped[Thing] = relationship(back_populates="locations")

class PGVectorCollection:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        Base.metadata.create_all(engine)

    def add(self, ids, embeddings, metadatas):
        with self.Session() as session:
            for i, e, m in zip(ids, embeddings, metadatas):
                thing = session.get(Thing, i)

                if thing is None:
                    thing = Thing(id=i,embedding=e,metadata_=m)
                    session.add(thing)
                else:
                    thing.embedding = e
                    thing.metadata_ = m

                location = m.get("location", "unknown")

                existing_location = (session.query(ThingLocation).filter(ThingLocation.thing_id == i).first())

                if existing_location:
                    existing_location.location = location
                else:
                    session.add( ThingLocation(id=str(uuid.uuid4()),location=location,thing_id=i))

            session.commit()
    def count(self):
        with self.Session() as session:
            return session.query(Thing).count()

    def get(self):
        with self.Session() as session:
            things = session.scalars(select(Thing)).all()
            return {
                "ids": [t.id for t in things],
                "metadatas": [t.metadata_ for t in things],
            }

    def update_metadata(self, ids, metadatas):
        with self.Session() as session:
            for i, m in zip(ids, metadatas):
                thing = session.get(Thing, i)
                if thing:
                    thing.metadata_ = m
            session.commit()

    def update_location(self, ids, locations):
        with self.Session() as session:

            for i, loc in zip(ids, locations):

                location = (
                    session.query(ThingLocation)
                    .filter(ThingLocation.thing_id == i)
                    .first()
                )

                if location:
                    location.location = loc

            session.commit()

    def query_loc(self, location):
        with self.Session() as session:
            result = (session.query(Thing.id,Thing.metadata_,ThingLocation.location)
                .join(ThingLocation)
                .filter(ThingLocation.location == location)
                .all()
            )

            return {
                "ids": [[r.id for r in result]],
                "metadatas": [[r.metadata_ for r in result]],
                "locations": [[r.location for r in result]],
            }

    def query(self, query_embeddings, n_results=5):
        query_vec = query_embeddings[0]

        with self.Session() as session:
            distance = Thing.embedding.cosine_distance(query_vec)
            result = (
                session.query( Thing.id, Thing.metadata_, distance.label("distance"))
                .order_by(distance)
                .limit(n_results)
                .all()
            )

            return {
                "ids": [[r.id for r in result]],
                "metadatas": [[r.metadata_ for r in result]],
                "distances": [[float(r.distance) for r in result]],
            }
    def clear(self):
        with self.engine.begin() as conn:
            conn.execute( text("TRUNCATE TABLE things2 CASCADE") )