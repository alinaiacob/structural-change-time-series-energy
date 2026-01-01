from fastapi import FastAPI
from routes.main import router
import uvicorn

app = FastAPI(title="Structural Change API")

app.include_router(router, prefix="/api")

if __name__=="main":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True)

