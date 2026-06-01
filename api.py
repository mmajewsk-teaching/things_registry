from fastapi import Request, FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile, shutil
from datetime import datetime
from sqlalchemy import create_engine
from db.postgres_sqlalchemy import PGVectorCollection
from pipelines.detection_pipeline import process_image
from pipelines.ingestion_pipeline import run_ingestion_pipeline
from pipelines.retrieval_pipeline import respond_to_query,respond_to_query_location
from services.crop_service import register_crop
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
#this value is to change is only for testing purposes, to see the effect of metadata update on retrieval results
SAME_OBJECT_THRESHOLD = 0.70
#class for receiving change location requests
class ChangeLoc(BaseModel):
    path: str
    location: str

#Class for receiving query requests
class QueryRequest(BaseModel):
    path: str

class LocationRequest(BaseModel):
    location: str
#[Previous Instruction]
### run database container
    ### docker run -d \
    #   --name pgvector \
    #   -e POSTGRES_USER=myuser \
    #   -e POSTGRES_PASSWORD=mypass \
    #   -e POSTGRES_DB=mydb \
    #   -p 5432:5432 \
    #   pgvector/pgvector:pg16
    #### conda to run project 
    #### source ~/miniconda3/bin/activate
    #### conda create -n ai python=3.10
    #### pip install -r requirements.txt
    #### python3 main.py
#[End Previous instruction]

#[New Instruction]
    #Step0: [OPTIONAL] run conda
    #Step1: pip install -r requirements.txt
    #Step2: docker run --name pgvector -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=haslo -e POSTGRES_DB=testdb -p 5432:5432 pgvector/pgvector:pg16
    #Step3: uvicorn api:app --reload
    #Step4: run index.html

#[End New Instruction]

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500","http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    app.state.engine = create_engine("postgresql://postgres:haslo@localhost:5432/testdb")
    app.state.things = PGVectorCollection(engine=app.state.engine,name="things2",dim=384)


@app.get("/things/count")
def get_things_count(request: Request):
    things = request.app.state.things
    return {
        "count": int(things.count()),
        "collection": things.name
    }

@app.post("/things/clear")
def clear_things(request: Request):
    things = request.app.state.things
    things.clear()
    things.initialized = False
    return {
        "status": "ok",
        "message": "things collection cleared"
    }

# @app.post("/query/path")
# def query_from_path(path: str, request: Request):
#     things = request.app.state.things

#     crops, _ = process_image(Path(path))
#     qcrop = crops[0]

#     return respond_to_query(qcrop, things,top_k=5)

@app.post("/add/path")
async def add_thing(request: Request, file: UploadFile = File(...), location: str = Form(...)):
    things = request.app.state.things
    uploads_dir = Path("./uploads")
    uploads_dir.mkdir(exist_ok=True)

    # Generate permanent file path
    suffix = Path(file.filename).suffix or ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = uploads_dir / filename

    # Save file permanently
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        crops, _ = process_image(file_path)
        qcrop = crops[0]

        # First pass: find best match to decide whether to update metadata
        if things.count() > 0:  # Only query if there are already things in the registry
            initial_res = respond_to_query(qcrop, things, top_k=5)
            top_score = initial_res['matches'][0]['similarity']
        
            if top_score >= SAME_OBJECT_THRESHOLD:
                return {
                "status": "ok",
                "message": "Object already exists with similar location. Do not add duplicate."
                }
        for c in crops:
            register_crop(c, location=location, things=things)

        return {
            "status": "ok",
            "message": "Object added successfully.",
            "file_path": str(file_path)
            }
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise

@app.post("/query/path")
async def query_from_path(request: Request, file: UploadFile = File(...)):
    things = request.app.state.things

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        crops, _ = process_image(tmp_path)
        qcrop = crops[0]
        return respond_to_query(qcrop, things, top_k=5)
    finally:
        tmp_path.unlink(missing_ok=True)

@app.post("/query/getperlocation")
async def query_per_location(request: Request, location: LocationRequest):
    things = request.app.state.things
    things_in_location = respond_to_query_location(location.location, things)
    return things_in_location

    

#EXAMPLE CALL:
#curl -X POST http://localhost:8000/query/changeloc -H 
#"Content-Type: application/json" 
#-d '{"path":"/home/alicja/CUDA/project/RepoCuda/DLWC_AA/test/mlotek_0.png","location":"garden"}'
@app.post("/query/changeloc")
async def query_change_loc(request: Request, file: UploadFile = File(...), location: str = Form(...)):
    things = request.app.state.things

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        crops, _ = process_image(tmp_path)
        qcrop = crops[0]

        # First pass: find best match to decide whether to update metadata
        initial_res = respond_to_query(qcrop, things, top_k=5)
        top_score = initial_res['matches'][0]['similarity']
        original_metadata = initial_res['matches'][0]['metadata']
        updated = False
        if top_score >= SAME_OBJECT_THRESHOLD:
            top_id = initial_res['matches'][0]['id']
            original_metadata['location'] = location
            things.update_metadata(ids=[top_id], metadatas=[original_metadata])
            things.update_location(ids=[top_id], locations=[location])
            updated = True

        # Second pass: refresh results after possible metadata update
        final_res = respond_to_query(qcrop, things, top_k=5)
        if updated:
            final_res.setdefault('message', '')
            final_res['message'] += f" Location updated to '{location}'."

        return final_res
    finally:
        tmp_path.unlink(missing_ok=True)

@app.post("/things/initialize")
def initialize_things(request: Request):
    things = request.app.state.things

    result = run_ingestion_pipeline(things=things)

    return {
        "status": "ok",
        **result,
    }
