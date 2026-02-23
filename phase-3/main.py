from fastapi import FastAPI
from api.routes import chat



app = FastAPI(title="Phase 3 - Generator")

app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
def hello():
    return {"message": "Hello from Phase 3 - Generator"}


