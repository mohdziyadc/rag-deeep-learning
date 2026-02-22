from fastapi import FastAPI



app = FastAPI(title="Phase 3 - Generator")


@app.get("/")
def hello():
    return {"message": "Hello from Phase 3 - Generator"}