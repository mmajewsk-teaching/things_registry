## Setup Steps

```bash
# Step 0: [OPTIONAL] run conda
# Step 1: pip install -r requirements.txt

# Step 2: Run PostgreSQL with pgvector
docker run --name pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=haslo \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Step 3: Start backend
uvicorn api:app --reload

# Step 4: Start frontend
# Open index.html in your browser