"""Shared HTTP contract for local Piper and Modal-hosted speech engines."""

from collections.abc import Callable
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import wave

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .pipeline import validate_wav
from .tts import PiperEngine, SpeechEngine


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice: str | None = Field(default=None, min_length=1, max_length=64)


def create_app(
    engine_factory: Callable[[], SpeechEngine],
    *,
    cache_dir: Path | None = None,
    commit_cache: Callable[[], None] | None = None,
) -> FastAPI:
    """Load one engine per container; optionally persist successful inference results."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.engine = engine_factory()
        application.state.lock = Lock()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        yield

    application = FastAPI(title='Audiobook TTS', lifespan=lifespan)

    def resolve_engine(voice: str | None) -> SpeechEngine:
        engine = application.state.engine
        if voice is None:
            return engine
        selector = getattr(engine, 'for_voice', None)
        if selector is None:
            raise HTTPException(status_code=400, detail='This endpoint does not support voice selection.')
        try:
            return selector(voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.get('/health')
    def health():
        return {'status': 'ok'}

    @application.get('/info')
    def info(voice: str | None = None):
        engine = resolve_engine(voice)
        return {'identity': engine.identity,
                'voices': getattr(engine, 'available_voices', []),
                'default_voice': getattr(engine, 'default_voice', None)}

    @application.post('/synthesize')
    def synthesize(request: SpeechRequest):
        if not request.text.strip():
            raise HTTPException(status_code=422, detail='Text must contain readable characters')
        engine = resolve_engine(request.voice)
        if not application.state.lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail='Model busy; retry later')
        try:
            key = hashlib.sha256(json.dumps([engine.identity, request.text],
                                            ensure_ascii=False).encode()).hexdigest()
            cached = cache_dir / f'{key}.wav' if cache_dir is not None else None
            cache_hit = False
            if cached is not None and cached.exists():
                try:
                    validate_wav(cached)
                    cache_hit = True
                except (OSError, ValueError, EOFError, wave.Error):
                    pass
            if cache_hit:
                data = cached.read_bytes()
            else:
                # With a persistent cache the temp file must be on that same filesystem.
                with tempfile.TemporaryDirectory(prefix='tts-', dir=cache_dir) as directory:
                    target = Path(directory) / 'speech.wav'
                    engine.synthesize(request.text, target)
                    validate_wav(target)
                    data = target.read_bytes()
                    if cached is not None:
                        target.replace(cached)
            # Also commit a cache hit: a prior failed commit may have left a local file.
            # Return only after the Volume confirms persistence.
            if commit_cache is not None:
                commit_cache()
            return Response(data, media_type='audio/wav', headers={
                'X-Voice-Identity': engine.identity,
                'X-Audio-Cache': 'hit' if cache_hit else 'miss',
            })
        finally:
            application.state.lock.release()

    return application


def local_engine() -> SpeechEngine:
    return PiperEngine(Path(os.environ.get('TTS_MODEL', '/models/en_US-lessac-medium.onnx')),
                       float(os.environ.get('TTS_SPEED', '1.0')))


app = create_app(local_engine)
