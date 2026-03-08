from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes.graphrag import vector_store
from api.routes import graphrag
@asynccontextmanager
async def lifespan(app: FastAPI):
    await vector_store.create_index()
    yield
    await vector_store.close()


app = FastAPI(title="Phase - 4 - Knowledge Graph")

app.include_router(graphrag.router, prefix='/api', tags=['graphrag'])



@app.get('/')
def hello():
    return {"message": "Hello from Phase-4"}

