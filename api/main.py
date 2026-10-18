from dotenv import load_dotenv
load_dotenv()

import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs('uploads', exist_ok=True)
    yield

app = FastAPI(title='Adaptive Study Planner API', description='Backend API for the LangGraph-powered Adaptive Study Planner.', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(router, prefix='/api')

@app.get('/health', tags=['Health'])
async def health_check():
    return {'status': 'healthy'}
if __name__ == '__main__':
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=True)