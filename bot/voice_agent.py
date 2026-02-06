"""
AIQNEX Voice Agent — Pipecat pipeline with NVIDIA Riva STT/TTS and NVIDIA LLM.

Pipeline: DailyTransport → Riva STT (Parakeet) → NVIDIA LLM → Riva TTS (Magpie) → DailyTransport
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.services.openai.llm import OpenAILLMContext
from pipecat.transports.daily.transport import DailyParams, DailyTransport

load_dotenv()

# ---------------------------------------------------------------------------
# AIQNEX system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the AIQNEX Voice Assistant — a friendly, professional AI representative for AIQNEX (aiqnex.com), an AI & Quantum Computing training institution based in Singapore.

ABOUT AIQNEX:
AIQNEX is a premier training institution specializing in Artificial Intelligence and Quantum Computing education. We empower professionals, businesses, and curious learners with cutting-edge skills for the future.

PROGRAMS OFFERED:
1. AI Engineering Bootcamp — Intensive weekend program covering machine learning, deep learning, and AI application development.
2. Business Leadership Workshops — Helping leaders understand and leverage AI strategy for competitive advantage.
3. Self-Paced Modules — Flexible online courses for independent learners covering AI and Quantum Computing fundamentals.
4. Community Webinars — Free monthly sessions on trending AI and Quantum topics.
5. Career Readiness Program — Interview prep, portfolio building, and job placement support for AI careers.
6. Corporate Training — Customized AI and Quantum Computing training for enterprise teams.

UPCOMING COURSES:
- "Mastering Agentic AI" — Feb 28 - Mar 1, 2025 (Weekend). Price: SGD $800. Covers autonomous AI agents, tool use, multi-agent systems, and practical deployment.
- "Quantum Computing Fundamentals" — Feb 7-8, 2025 (Weekend). Price: SGD $800. Covers qubits, quantum gates, circuits, algorithms (Grover's, Shor's), and hands-on with Qiskit.

LEADERSHIP TEAM:
- Vinod Martin — Co-Founder. 30+ years in IT industry, passionate about democratizing AI and Quantum education.
- Shreya Dasaur Chadha — Training Program Manager. Expert in curriculum design and learning experience.
- Shashank — Technology Advisor. Guides the technical direction of AIQNEX programs.

CONTACT INFORMATION:
- Email: contact@aiqnex.com
- Phone: +65 8974 9095
- Address: 10 Ubi Crescent, #04-33 Ubi Techpark, Singapore 408564
- Website: aiqnex.com

CONVERSATION GUIDELINES:
- Be warm, enthusiastic, and professional.
- Keep responses concise (2-3 sentences) since this is a voice conversation.
- NEVER say the company name out loud. Instead refer to it as "we", "us", "our institute", or "our training programs". For example say "Welcome to our AI and Quantum Computing training institute!" instead of mentioning the name.
- If asked about topics outside our scope, politely redirect to what we offer.
- Encourage users to visit our website or contact us for more details.
- Use natural, conversational language appropriate for spoken dialogue.
"""


# ---------------------------------------------------------------------------
# Main bot entry point
# ---------------------------------------------------------------------------

async def run_bot(room_url: str, token: str):
    """Run the AIQNEX voice agent bot in a Daily.co room."""
    logger.info(f"Starting AIQNEX bot in room: {room_url}")

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    # ---- Daily.co transport ------------------------------------------------
    transport = DailyTransport(
        room_url,
        token,
        "AIQNEX Assistant",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=False,
        ),
    )

    # ---- NVIDIA Riva STT (Parakeet RNNT) -----------------------------------
    stt = NvidiaSTTService(
        api_key=nvidia_api_key,
        server=os.getenv("RIVA_ASR_URL", "grpc.nvcf.nvidia.com:443"),
        model_function_map={
            "function_id": os.getenv("RIVA_ASR_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081"),
            "model_name": "parakeet-ctc-1.1b-asr",
        },
        sample_rate=int(os.getenv("RIVA_ASR_SAMPLE_RATE", "16000")),
    )
    logger.info("Riva STT ready (Parakeet)")

    # ---- NVIDIA LLM --------------------------------------------------------
    llm = NvidiaLLMService(
        api_key=nvidia_api_key,
        base_url=os.getenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct"),
    )
    logger.info(f"NVIDIA LLM ready ({os.getenv('NVIDIA_LLM_MODEL', 'meta/llama-3.1-8b-instruct')})")

    # ---- NVIDIA Riva TTS (Magpie Multilingual) -----------------------------
    tts = NvidiaTTSService(
        api_key=nvidia_api_key,
        server=os.getenv("RIVA_TTS_URL", "grpc.nvcf.nvidia.com:443"),
        voice_id=os.getenv("RIVA_TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Sofia"),
        model_function_map={
            "function_id": os.getenv("RIVA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969"),
            "model_name": "magpie-tts-multilingual",
        },
        sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "16000")),
    )
    logger.info("Riva TTS ready (Magpie)")

    # ---- LLM context -------------------------------------------------------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ---- Pipeline -----------------------------------------------------------
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ---- Event handlers -----------------------------------------------------
    greeted = {"value": False}

    WELCOME_MESSAGE = (
        "Welcome! I'm your AI assistant for our AI and Quantum Computing training institute in Singapore. "
        "I can tell you about our programs, upcoming courses, pricing, and more. "
        "How can I help you today?"
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant.get('id')}")
        # Speak a fixed welcome message immediately (no LLM delay, no pronunciation issues)
        await task.queue_frames([TTSSpeakFrame(text=WELCOME_MESSAGE)])
        greeted["value"] = True

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant.get('id')}, reason: {reason}")
        await task.cancel()

    # Auto-greet fallback
    async def _autogreet():
        try:
            await asyncio.sleep(3.0)
            if not greeted["value"]:
                await task.queue_frames([TTSSpeakFrame(text=WELCOME_MESSAGE)])
                logger.info("Auto-greeting enqueued")
        except Exception:
            logger.debug("autogreet failed")

    asyncio.create_task(_autogreet())

    # ---- Run ----------------------------------------------------------------
    runner = PipelineRunner()
    await runner.run(task)
    logger.info("Bot pipeline finished")
