# backend/api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum  # ASGI adapter for serverless

app = FastAPI()

# (optional) your middlewares, routers, etc.
@app.get("/")
def read_root():
    return {"message": "NoteMate AI backend running on Vercel!"}

# wrap in Mangum for AWS Lambda (which Vercel uses)
handler = Mangum(app)
