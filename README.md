# AIQNEX Voice Bot

Voice AI assistant for [AIQNEX](https://aiqnex.com) — AI & Quantum Computing training in Singapore.

## Architecture

```
Browser → Cloudflare Worker (frontend) → /api/* proxy → Python Backend (FastAPI)
                                                            ↓
                                                    Pipecat Pipeline:
                                              Daily.co ↔ Riva STT ↔ NVIDIA LLM ↔ Riva TTS
```

## Quick Start

### 1. Python Backend

```bash
cd bot
pip install -r requirements.txt

# Edit .env with your API keys
#   DAILY_API_KEY=...
#   NVIDIA_API_KEY=...

python server.py
```

### 2. Cloudflare Worker (Frontend)

```bash
cd worker

# Update BACKEND_URL in wrangler.toml to your cloudflared tunnel URL
wrangler deploy
```

### 3. Expose Backend via Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8080
```

Update `BACKEND_URL` in `worker/wrangler.toml` to the tunnel URL, then redeploy.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /room | Create Daily.co room + spawn bot |
| GET | /health | Health check |
| GET | /rooms | List active rooms |
| DELETE | /room/{name} | Stop bot + delete room |

## Stack

- **STT**: NVIDIA Riva Parakeet RNNT 1.1b
- **LLM**: NVIDIA Llama 3.1 8B Instruct (via Integrate API)
- **TTS**: NVIDIA Riva Magpie Multilingual
- **Transport**: Daily.co WebRTC
- **Orchestration**: Pipecat
- **Frontend**: Cloudflare Workers
