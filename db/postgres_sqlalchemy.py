from sqlalchemy import (MetaData, Table, Column, Text, String, Index, ForeignKey, update,select,func, text)
from sqlalchemy.dialects.postgresql import (JSONB,insert)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker
import uuid

class PGVectorCollection:
    def __init__(self, engine, name, dim=384):
        self.engine = engine
        self.name = name
        self.dim = dim

        self.metadata = MetaData()
        self.Session = sessionmaker(bind=engine)

        self.table = Table(
            name,
            self.metadata,
            Column("id", Text, primary_key=True),
            Column("embedding", Vector(dim)),
            Column("metadata_", JSONB),
            Index(f"{name}_embedding_idx","embedding",postgresql_using="hnsw",postgresql_ops={"embedding": "vector_cosine_ops"}),
            Index(f"{name}_metadata_idx","metadata_",postgresql_using="gin"),
        )

        self.location_table = Table(
            f"{name}_location",
            self.metadata,
            Column("id", Text, primary_key=True),
            Column("location", String),
            Column("thing_id", Text, ForeignKey(f"{name}.id", ondelete="CASCADE")),
        )

        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.metadata.create_all(conn)

    def add(self, ids, embeddings, metadatas):
        with self.Session() as session:
            for i, e, m in zip(ids, embeddings, metadatas):
                stmt = insert(self.table).values(id=i,embedding=e,metadata_=m)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "embedding": stmt.excluded.embedding,
                        "metadata_": stmt.excluded.metadata_,
                    }
                )
                session.execute(stmt)

                location = m.get("location", "unknown")
                loc_stmt = insert(self.location_table).values(id=str(uuid.uuid4()),location=location,thing_id=i)

                loc_stmt = loc_stmt.on_conflict_do_update(index_elements=["id"],set_={"location": loc_stmt.excluded.location})
                session.execute(loc_stmt)

            session.commit()

    def update_metadata(self, ids, metadatas):
        with self.Session() as session:
            for i, m in zip(ids, metadatas):
                stmt = (update(self.table).where(self.table.c.id == i).values(metadata_=m))
                session.execute(stmt)
            session.commit()

    def update_location(self, ids, locations):
        with self.Session() as session:
            for i, loc in zip(ids, locations):
                stmt = (update(self.location_table).where(self.location_table.c.thing_id == i).values(location=loc))
                session.execute(stmt)
            session.commit()

    def count(self):
        with self.Session() as session:
            stmt = select(func.count()).select_from(self.table)
            return session.execute(stmt).scalar()

    def query(self, query_embeddings, n_results=5):
        query_vec = query_embeddings[0]

        with self.Session() as session:
            stmt = (
                select(self.table.c.id, self.table.c.metadata_,(self.table.c.embedding.cosine_distance(query_vec)).label("distance"))
                .order_by(self.table.c.embedding.cosine_distance(query_vec))
                .limit(n_results)
            )
            result = session.execute(stmt).all()
            return {
                "ids": [[r.id for r in result]],
                "metadatas": [[r.metadata_ for r in result]],
                "distances": [[float(r.distance) for r in result]],
            }

    def query_loc(self, location):
        with self.Session() as session:
            stmt = (select(self.table.c.id, self.table.c.metadata_, self.location_table.c.location)
                .join(self.location_table, self.table.c.id == self.location_table.c.thing_id)
                .where(self.location_table.c.location == location)
            )
            result = session.execute(stmt).all()

            return {
                "ids": [[r.id for r in result]],
                "metadatas": [[r.metadata_ for r in result]],
                "locations": [[r.location for r in result]],
            }

    def get(self):
        with self.Session() as session:
            stmt = select(self.table.c.id,self.table.c.metadata_)
            result = session.execute(stmt).all()
            return {
                "ids": [r.id for r in result],
                "metadatas": [r.metadata_ for r in result],
            }

    def clear(self):
        with self.engine.begin() as conn:
            conn.execute(
                text(f"TRUNCATE TABLE {self.name} CASCADE")
            )