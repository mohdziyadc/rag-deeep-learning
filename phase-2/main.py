from fastapi import FastAPI


app = FastAPI(title='Doc Parser')

@app.get("/")
def root():
    return {"message": "RAG Doc Parser API is running"}