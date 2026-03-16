from fastapi import FastAPI


app = FastAPI(title='Phase 5 - Memory')


@app.get('/')
def main():
    return {"msg": "Hello from phase-5!"}

