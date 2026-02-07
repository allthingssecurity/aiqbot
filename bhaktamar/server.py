"""
Bhaktambar Voice Bot — FastAPI backend.

Runs on port 8081 (alongside AIQNEX bot on 8080).
"""

import asyncio
import os
import ssl
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import aiohttp
import certifi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

load_dotenv()

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_URL = "https://api.daily.co/v1"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

active_bots: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Bhaktambar voice-bot server starting ...")
    yield
    logger.info("Shutting down — cancelling active bots ...")
    for room_name, task in active_bots.items():
        task.cancel()


app = FastAPI(title="Bhaktambar Voice Bot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RoomResponse(BaseModel):
    room_url: str
    room_name: str
    token: str


class HealthResponse(BaseModel):
    status: str
    service: str
    daily_configured: bool
    active_rooms: int


async def create_daily_room(room_name: Optional[str] = None) -> dict:
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
        "Content-Type": "application/json",
    }
    room_config = {
        "properties": {
            "exp": int(datetime.now().timestamp()) + 3600,
            "enable_chat": False,
            "enable_screenshare": False,
            "start_video_off": True,
            "start_audio_off": False,
            "enable_knocking": False,
            "enable_prejoin_ui": False,
        }
    }
    if room_name:
        room_config["name"] = room_name

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=SSL_CTX)) as session:
        async with session.post(
            f"{DAILY_API_URL}/rooms", headers=headers, json=room_config
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Daily API error: {body}")
            return await resp.json()


async def get_daily_token(
    room_name: str, is_owner: bool = False, user_name: Optional[str] = None
) -> str:
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
        "Content-Type": "application/json",
    }
    token_config = {
        "properties": {
            "room_name": room_name,
            "is_owner": is_owner,
            "exp": int(datetime.now().timestamp()) + 3600,
            "enable_screenshare": False,
            "start_video_off": True,
            "start_audio_off": False,
        }
    }
    if user_name:
        token_config["properties"]["user_name"] = user_name

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=SSL_CTX)) as session:
        async with session.post(
            f"{DAILY_API_URL}/meeting-tokens", headers=headers, json=token_config
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Daily token error: {body}")
            data = await resp.json()
            return data["token"]


async def spawn_bot(room_url: str, token: str, room_name: str):
    try:
        from voice_agent import run_bot
        logger.info(f"Spawning bot for room: {room_name}")
        await run_bot(room_url, token)
    except asyncio.CancelledError:
        logger.info(f"Bot cancelled for room: {room_name}")
    except Exception as e:
        logger.error(f"Bot error for room {room_name}: {e}")
    finally:
        active_bots.pop(room_name, None)


@app.get("/")
async def root():
    return {"service": "Bhaktambar Voice Bot", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="bhaktambar-voice-bot",
        daily_configured=bool(DAILY_API_KEY),
        active_rooms=len(active_bots),
    )


@app.post("/room", response_model=RoomResponse)
async def create_room(room_name: Optional[str] = None):
    if not DAILY_API_KEY:
        raise HTTPException(status_code=500, detail="DAILY_API_KEY not configured")

    room = await create_daily_room(room_name)
    room_url = room["url"]
    actual_room_name = room["name"]

    bot_token = await get_daily_token(actual_room_name, is_owner=True, user_name="Bhaktambar Guide")
    user_token = await get_daily_token(actual_room_name, is_owner=False, user_name="User")

    task = asyncio.create_task(spawn_bot(room_url, bot_token, actual_room_name))
    active_bots[actual_room_name] = task

    logger.info(f"Room created: {actual_room_name} → {room_url}")
    return RoomResponse(room_url=room_url, room_name=actual_room_name, token=user_token)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8081"))
    logger.info(f"Starting Bhaktambar voice-bot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
