
import json
from sqlalchemy import text
import uuid

DATABASE_URL = "postgresql://postgres:haslo@localhost:5432/testdb"

class PGVectorCollection:
    def __init__(self, engine, name, dim=384):
        self.engine = engine
        self.name = name
        self.dim = dim

        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({dim}),
                    metadata JSONB
                )
            """))

            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {name}_location (
                    id TEXT PRIMARY KEY,
                    location TEXT,
                    thing_id TEXT,
                    CONSTRAINT fk_{name}_location
                    FOREIGN KEY (thing_id)
                    REFERENCES {name}(id)
                    ON DELETE CASCADE
                )
            """))

            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{name}_location_location
                ON {name}_location(location)
            """))
           
             # indeks vector (cosine)
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {name}_embedding_idx
                ON {name}
                USING hnsw (embedding vector_cosine_ops)
            """))

            # indeks metadata
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {name}_metadata_idx
                ON {name} USING GIN (metadata)
            """))

            conn.commit()

    def add(self, ids, embeddings, metadatas):
        with self.engine.connect() as conn:
            for i, e, m in zip(ids, embeddings, metadatas):
                conn.execute(text(f"""
                    INSERT INTO {self.name} (id, embedding, metadata)
                    VALUES (:id, :embedding, :metadata)
                    ON CONFLICT (id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """), {
                    "id": i,
                    "embedding": e,
                    "metadata": json.dumps(m)
                })
                location = m.get('location', 'unknown')
                location_id = str(uuid.uuid4())
                conn.execute(text(f"""
                    INSERT INTO {self.name}_location (id, location, thing_id)
                    VALUES (:location_id, :location, :thing_id)
                    ON CONFLICT (id) DO UPDATE
                    SET location = EXCLUDED.location
                """), {
                    "location_id": location_id,
                    "location": location,
                    "thing_id": i
                })
            conn.commit()

    def update_metadata(self, ids, metadatas):
        with self.engine.connect() as conn:
            for i, m in zip(ids, metadatas):
                conn.execute(text(f"""
                    UPDATE {self.name}
                    SET metadata = :metadata
                    WHERE id = :id
                """), {
                    "id": i,
                    "metadata": json.dumps(m)
                })
            conn.commit()

    def update_location(self, ids, locations):
        with self.engine.connect() as conn:
            for i, loc in zip(ids, locations):
                conn.execute(text(f"""
                    UPDATE {self.name}_location
                    SET location = :location
                    WHERE thing_id = :thing_id
                """), {
                    "thing_id": i,
                    "location": loc
                })
            conn.commit()

    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(
                text(f"SELECT COUNT(*) FROM {self.name}")
            ).scalar()

    def query(self, query_embeddings, n_results=5, include=None):
        query_vec = query_embeddings[0]

        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT id, metadata, embedding <=> :query AS distance
                FROM {self.name}
                ORDER BY embedding <=> :query
                LIMIT :k
            """), {
                "query": str(query_vec),
                "k": n_results
            }).fetchall()

        return {
            "ids": [[r[0] for r in result]],
            "metadatas": [[r[1] for r in result]],
            "distances": [[float(r[2]) for r in result]],
        }

    def query_loc(self, location):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT t.id, t.metadata, l.location
                FROM {self.name} t
                JOIN {self.name}_location l ON t.id = l.thing_id
                WHERE l.location = :location
            """), {
                "location": location
            }).fetchall()

        return {
            "ids": [[r[0] for r in result]],
            "metadatas": [[r[1] for r in result]],
            "locations": [[r[2] for r in result]],
        }

    def get(self, include=None):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT id, metadata FROM {self.name}
            """)).fetchall()

        return {
            "ids": [r[0] for r in result],
            "metadatas": [r[1] for r in result],
        }

    def clear(self):
        with self.engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.name} CASCADE"))
            conn.commit()

class VocabularyCollection:
    def __init__(self, engine, name="vocabulary", dim=512):
        self.engine = engine
        self.name = name
        self.dim = dim

        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    label TEXT PRIMARY KEY,
                    embedding vector({dim})
                )
            """))

            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {name}_embedding_idx
                ON {name}
                USING hnsw (embedding vector_cosine_ops)
            """))

            conn.commit()

    def add(self, label, embedding):
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                   INSERT INTO {self.name} (label, embedding)
                   VALUES (:label, :embedding)
                   ON CONFLICT (label) DO UPDATE
                   SET embedding = EXCLUDED.embedding
               """), {
                "label": label,
                "embedding": embedding.tolist()
            })

            conn.commit()

    def add_many(self, labels, embeddings):
        with self.engine.connect() as conn:
            for label, emb in zip(labels, embeddings):
                conn.execute(text(f"""
                    INSERT INTO {self.name} (label, embedding)
                    VALUES (:label, :embedding)
                    ON CONFLICT (label) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                """), {
                    "label": label,
                    "embedding": emb.tolist()
                })

            conn.commit()

    def get_all(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                   SELECT label, embedding
                   FROM {self.name}
               """)).fetchall()

        return result

    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(
                text(f"SELECT COUNT(*) FROM {self.name}")
            ).scalar()

    def clear(self):
        with self.engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.name}"))
            conn.commit()


# def create_things():
#     return PGVectorCollection(engine, name="things2", dim=1024)
#
# def create_vocab():
#         return VocabularyCollection(engine, name="vocabulary", dim=512)

# def create_engine():
#     return sa_create_engine(
#         DATABASE_URL,
#         pool_pre_ping=True,
#         pool_size=5,
#         max_overflow=10,
#     )
#
# def create_things(engine):
#     return PGVectorCollection(
#         engine=engine,
#         name="things2",
#         dim=1024
#     )