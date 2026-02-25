from fastapi import FastAPI



app = FastAPI(title="Phase - 4 - Knowledge Graph")

@app.get('/')
def hello():
    return {"message": "Hello from Phase-4"}

