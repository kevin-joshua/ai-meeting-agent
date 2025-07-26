from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def root():
  return {"msg" : "Welcome to AI Meeting Agent"}

@app.get("/health")
def health():
  return {"status" : "ok"}