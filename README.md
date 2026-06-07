## Things Registry

Visual object registry backed by PostgreSQL + pgvector. Upload a photo of an object to find matches, check locations, or update where something lives.

---

## Setup

```bash
# 0. (Optional) Create and activate the conda environment
source ~/miniconda3/bin/activate
conda create -n ai python=3.10 -y
conda activate ai

# 1. Install dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL with pgvector
docker run --name pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=haslo \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# If the container already exists (stopped):
docker start pgvector

# 3. Start the backend
uvicorn api:app --reload

# 4. In a second terminal, serve the frontend on port 5500
# (required: CORS is configured to allow requests only from this port)

cd frontend

# Use whichever command is available on your system:
python -m http.server 5500
# or
python3 -m http.server 5500
```

Open **http://localhost:5500** in your browser.

> **Note:** Do not open `index.html` directly as `file://` - API calls will be blocked by CORS.

---

## Usage

| Card | What it does |
|---|---|
| **Initialize Things** | Runs the ingestion pipeline - embeds all images in `DLWC_AA/things_photos/` and stores them in the DB |
| **Get Things Count** | Shows how many embeddings are stored |
| **Clear Things** | Deletes all embeddings from the DB |
| **Query Path** | Upload an image → finds the top 5 visually similar objects and their locations |
| **Change Location** | Upload an image + type a new location → updates the location of the best matching object |

Results are shown as image cards (with similarity % and location) plus raw JSON below.

---

## Stack

- **FastAPI** - REST API + static file serving
- **pgvector / PostgreSQL** - vector similarity search
- **DINOv2** - image embeddings (runs on CPU by default)
- **Vanilla JS + CSS** - frontend, no build step
