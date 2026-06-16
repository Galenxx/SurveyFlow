"""
FastAPI application entry point.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.routes import survey, health, streaming

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Survey Agent API starting up...")
    yield
    print("Survey Agent API shutting down...")

app = FastAPI(
    title="Survey Agent API",
    description="Agent-Driven Research Survey Writing System API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(survey.router)
app.include_router(streaming.router)


@app.get("/")
async def root():
    return {
        "name": "Survey Agent API",
        "version": "0.1.0",
        "docs": "/docs",
    }
