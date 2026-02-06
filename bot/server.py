"""
AIQNEX Voice Bot — FastAPI backend.

Endpoints:
  POST /room   → create Daily.co room, spawn bot, return room URL + user token
  GET  /health → health check
  GET  /       → service info
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import ssl

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

# ---------------------------------------------------------------------------
# Active bot tracking
# ---------------------------------------------------------------------------
active_bots: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AIQNEX voice-bot server starting …")
    yield
    logger.info("Shutting down — cancelling active bots …")
    for room_name, task in active_bots.items():
        task.cancel()
        logger.info(f"Cancelled bot for room: {room_name}")


app = FastAPI(title="AIQNEX Voice Bot", lifespan=lifespan)

# CORS — allow CF Worker and local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RoomResponse(BaseModel):
    room_url: str
    room_name: str
    token: str


class HealthResponse(BaseModel):
    status: str
    service: str
    daily_configured: bool
    active_rooms: int


# ---------------------------------------------------------------------------
# Daily.co helpers
# ---------------------------------------------------------------------------
async def create_daily_room(room_name: Optional[str] = None) -> dict:
    """Create a new Daily.co room."""
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
    """Get a meeting token for a Daily.co room."""
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


# ---------------------------------------------------------------------------
# Bot spawning
# ---------------------------------------------------------------------------
async def spawn_bot(room_url: str, token: str, room_name: str):
    """Run the voice agent bot for a room."""
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "AIQNEX Voice Bot",
        "version": "1.0.0",
        "endpoints": {
            "POST /room": "Create room and spawn bot",
            "GET /health": "Health check",
            "GET /rooms": "List active rooms",
            "DELETE /room/{room_name}": "Stop bot and delete room",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="aiqnex-voice-bot",
        daily_configured=bool(DAILY_API_KEY),
        active_rooms=len(active_bots),
    )


@app.post("/room", response_model=RoomResponse)
async def create_room(room_name: Optional[str] = None):
    """Create a new Daily.co room and spawn an AIQNEX voice bot."""
    if not DAILY_API_KEY:
        raise HTTPException(status_code=500, detail="DAILY_API_KEY not configured")

    room = await create_daily_room(room_name)
    room_url = room["url"]
    actual_room_name = room["name"]

    bot_token = await get_daily_token(actual_room_name, is_owner=True, user_name="AIQNEX Assistant")
    user_token = await get_daily_token(actual_room_name, is_owner=False, user_name="User")

    task = asyncio.create_task(spawn_bot(room_url, bot_token, actual_room_name))
    active_bots[actual_room_name] = task

    logger.info(f"Room created: {actual_room_name} → {room_url}")
    return RoomResponse(room_url=room_url, room_name=actual_room_name, token=user_token)


@app.get("/rooms")
async def list_rooms():
    return {"active_rooms": list(active_bots.keys()), "count": len(active_bots)}


@app.delete("/room/{room_name}")
async def delete_room(room_name: str):
    """Stop bot and delete a room."""
    if room_name in active_bots:
        active_bots[room_name].cancel()
        del active_bots[room_name]

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=SSL_CTX)) as session:
        async with session.delete(
            f"{DAILY_API_URL}/rooms/{room_name}",
            headers={"Authorization": f"Bearer {DAILY_API_KEY}"},
        ) as resp:
            return {"status": "deleted", "room_name": room_name}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("BOT_SERVER_PORT", "8080")))
    logger.info(f"Starting AIQNEX voice-bot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
