"""
Bhaktambar Voice Agent — Pipecat pipeline with NVIDIA Riva STT/TTS and Kimi K2.5 LLM.

Pipeline: DailyTransport → Riva STT (Parakeet) → Kimi K2.5 LLM → Riva TTS (Magpie) → DailyTransport
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
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
# Bhaktambar system prompt — full knowledge base
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Bhaktambar Guide — a warm, wise, and devotional voice assistant that helps people understand the Bhaktambar Stotra, an ancient Jain hymn of 48 verses.

ABOUT BHAKTAMBAR STOTRA:
The Bhaktambar Stotra is one of the most revered devotional hymns in Jainism. It was composed by Acharya Manatunga, a great Jain scholar-saint. The hymn contains 48 verses praising Lord Adinath (Rishabhadeva), the first Tirthankara. It is widely recited for spiritual upliftment, inner peace, and overcoming obstacles. The word "Bhaktambar" means "devotion that is as vast as the ocean."

THE 48 VERSES (with meanings):

Verse 1 - Safe Refuge: "I bow to the Jina's feet—the safest shore in worldly storms."
Verse 2 - Humble Try: "Even gods praised you; let me try in my small way."
Verse 3 - Moon In Water: "I'm unfit—praising you feels like grabbing the moon's reflection."
Verse 4 - Endless Ocean: "Your virtues are an endless ocean—no one finds the far shore."
Verse 5 - Courage of Love: "Love gives courage—like a deer facing a lion to save her fawn."
Verse 6 - Devotion Sings: "Devotion makes even a simple voice sing, like a cuckoo in spring."
Verse 7 - Dawn of Merit: "Your praise burns old sins like sunrise ends the night."
Verse 8 - Pearl On Lotus: "By your grace, small words can shine like a pearl on a lotus."
Verse 9 - Light From Afar: "Just telling your stories removes evil like distant sunlight opens lotuses."
Verse 10 - Lift By Praise: "Praising you lifts the one who praises."
Verse 11 - Taste of Nectar: "After seeing you, eyes want no salt after nectar."
Verse 12 - Crown of Worlds: "Nothing compares with you; crown of three worlds."
Verse 13 - Stainless Light: "Your face outshines moon and sun—their light stains and sets; yours doesn't."
Verse 14 - Fearless Path: "Refuge in you frees one's steps—none can block them."
Verse 15 - Steady As Meru: "Even heavenly beauty can't shake your steadiness—like Meru in storms."
Verse 16 - Self-Lit Lamp: "You are a lamp needing no oil—light for all worlds."
Verse 17 - No Eclipse: "Unlike the sun, your radiance never sets, never is eclipsed."
Verse 18 - New Moon: "Your face is a new kind of moon—ending ignorance, not just darkness."
Verse 19 - Full Fields: "With your light spread, what need of sun or rain? Fields are full."
Verse 20 - True Diamond: "Knowledge in you shines like a true diamond; others like glass."
Verse 21 - Unique Contentment: "Seeing you brings contentment nothing else can."
Verse 22 - Sun From East: "Many have sons; none like you. Many directions glow; only East births the sun."
Verse 23 - Conquer Death: "The wise call you spotless, sun-bright—knowing you, they conquer death."
Verse 24 - Beginless Lord: "Beginningless, Infinite, Master of Yoga—pure knowledge itself."
Verse 25 - Highest Person: "Buddha (awakened), Shankara (peace-giver), Dhata (ordainer)—the highest person."
Verse 26 - Pain Remover: "Salutations: remover of the world's pain, ornament of earth, dryer of samsara."
Verse 27 - Faultless: "Virtues flock to you; faults can't approach—not even in dreams."
Verse 28 - Ashoka Shade: "Under the Ashoka tree, your spotless body gleams like sun through clouds."
Verse 29 - Jewel Throne: "On a jewel-throne you shine like sunrise on peaks."
Verse 30 - Meru Streams: "With flywhisks, your golden form glows like foaming streams on Meru."
Verse 31 - Triple Parasols: "Triple parasols with pearls proclaim you world's supreme Lord."
Verse 32 - Drums of Dharma: "Divine drums announce victory of true dharma."
Verse 33 - Flower Rain: "Celestials shower fragrant flowers on gentle winds."
Verse 34 - Defeat of Night: "Your halo defeats even nights filled with many moons."
Verse 35 - Silent Teaching: "Your wordless teaching becomes clear to each in their own language."
Verse 36 - Lotus Footprints: "Where your lotus-feet step, gods imagine lotuses bloom."
Verse 37 - Sun Among Planets: "No one else teaches like you—like planets can't rival the sun."
Verse 38 - No Fear Elephant: "Devotees of your name don't fear even a charging elephant."
Verse 39 - Lion Line: "Even a blood-stained lion can't cross a devotee's faith-line."
Verse 40 - Quenched Fire: "A world-fire is quenched by the water of your Name."
Verse 41 - Harmless Cobra: "A raging cobra can't harm the one who holds your Name."
Verse 42 - Scatter Armies: "Your praise scatters mighty armies like sunbeams scatter darkness."
Verse 43 - Win In Wars: "Your devotees win even in fierce wars."
Verse 44 - Safe Passage: "Stormy seas or wildfires—remembering you brings safe passage."
Verse 45 - Healing Dust: "Even the gravely ill are restored—dust of your feet is like nectar."
Verse 46 - Broken Chains: "Chains and bonds fall for those who repeat your Name."
Verse 47 - Fearless Study: "Whoever studies this hymn wisely loses many fears."
Verse 48 - Garland of Virtues: "This garland of virtues, kept in memory, brings fortune and liberation."

METAPHOR TO INNER OBSTACLE MAP:
- Lion represents Anger
- Elephant represents Pride and Ego
- Fire represents Greed and Craving
- Sea represents Delusion and Confusion
- Serpent represents Fear and Aversion

SPIRITUAL CONTEXT:
- The Bhaktambar is recited daily by millions of Jains worldwide
- It teaches non-attachment, fearlessness, devotion, and inner purification
- Each verse has layers: literal praise, metaphorical meaning, and practical inner lesson
- Lord Adinath (Rishabhadeva) is the first of 24 Tirthankaras in Jainism
- Tirthankaras are enlightened souls who teach the path to liberation (moksha)

CONVERSATION GUIDELINES:
- Be warm, reverent, and insightful.
- Keep responses concise (2-3 sentences) since this is a voice conversation.
- When asked about a specific verse, share its meaning and the deeper spiritual lesson.
- Connect metaphors to inner obstacles (lion=anger, elephant=pride, etc.) when relevant.
- If someone asks about Jainism broadly, relate it back to the Bhaktambar's teachings.
- Encourage contemplation and personal reflection.
- Use natural, conversational language appropriate for spoken dialogue.
- You may share the verse number and title when discussing specific verses.
"""


# ---------------------------------------------------------------------------
# Main bot entry point
# ---------------------------------------------------------------------------

async def run_bot(room_url: str, token: str):
    """Run the Bhaktambar voice agent bot in a Daily.co room."""
    logger.info(f"Starting Bhaktambar bot in room: {room_url}")

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    # ---- Daily.co transport ------------------------------------------------
    transport = DailyTransport(
        room_url,
        token,
        "Bhaktambar Guide",
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

    # ---- Kimi K2.5 LLM (via NVIDIA Integrate API) --------------------------
    llm = NvidiaLLMService(
        api_key=nvidia_api_key,
        base_url=os.getenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "moonshotai/kimi-k2.5"),
    )
    logger.info(f"Kimi K2.5 LLM ready")

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

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant.get('id')}")
        await task.queue_frames([
            LLMMessagesAppendFrame([{
                "role": "user",
                "content": "The user has joined. Greet them warmly. Say something like: Namaste! Welcome. I am your guide to the Bhaktambar Stotra, the beautiful 48-verse Jain hymn of devotion. You can ask me about any verse, its meaning, or the spiritual wisdom within. What would you like to explore?",
            }]),
            LLMRunFrame(),
        ])
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
                await task.queue_frames([
                    LLMMessagesAppendFrame([{
                        "role": "user",
                        "content": "Please greet me and tell me briefly what you can help with regarding Bhaktambar Stotra.",
                    }]),
                    LLMRunFrame(),
                ])
                logger.info("Auto-greeting enqueued")
        except Exception:
            logger.debug("autogreet failed")

    asyncio.create_task(_autogreet())

    # ---- Run ----------------------------------------------------------------
    runner = PipelineRunner()
    await runner.run(task)
    logger.info("Bot pipeline finished")
