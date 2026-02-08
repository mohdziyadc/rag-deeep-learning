from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from db.metadata_store import metadata_store
from core.vectorstore.parsed_index import indexer
from core.ingestion.ingest import IngestionService
from core.ingestion.sources.gdrive import GDriveIngestor
import json


ingestor = IngestionService()
# gdrive_ingestor = GDriveIngestor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await metadata_store.db_init()
    await indexer.connect()
    await indexer.create_index()
    
    yield

    await indexer.close()

app = FastAPI(title='Doc Parser', lifespan=lifespan)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    result = await ingestor.ingest(
        file.filename, 
        data, 
        {"source": "local", "mime_type": file.content_type}
    )
    return result

@app.post('/ingest/gdrive/{folder_id}')
async def ingest_gdrive(folder_id:str):
    with open("service_account.json", "r", encoding="utf-8") as f:
        service_account_creds = json.load(f)
    gdrive = GDriveIngestor(service_account_json=service_account_creds, ingestion=ingestor)
    return await gdrive.ingest_folder(folder_id)
    


@app.get("/")
def root():
    return {"message": "PHASE 2 - RAG Doc Parser API is running"}