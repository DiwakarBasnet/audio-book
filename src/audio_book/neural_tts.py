"""Optional neural adapters. Heavy libraries are imported only inside Modal images."""

import hashlib
from copy import copy
from importlib.metadata import distributions
import json
from pathlib import Path
import wave

from .text import chunk_text
from .tts import digest


def reference_fingerprint(reference: Path) -> str:
    """Validate a short PCM16 reference and bind caching to its exact contents."""
    if not reference.is_file():
        raise ValueError(f'Chatterbox reference audio not found: {reference}')
    if reference.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Reference audio must be a WAV file smaller than 20 MiB.')
    try:
        with wave.open(str(reference), 'rb') as audio:
            if (audio.getcomptype() != 'NONE' or audio.getsampwidth() != 2
                    or audio.getnchannels() not in {1, 2}):
                raise ValueError('Reference audio must be mono or stereo PCM16 WAV.')
            duration = audio.getnframes() / audio.getframerate()
            if not 0 < duration <= 30 or audio.getframerate() < 8000:
                raise ValueError('Reference WAV must be nonempty, at most 30 seconds, and at least 8 kHz.')
            expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(audio.getnframes())) != expected:
                raise ValueError('Reference WAV is truncated.')
    except (wave.Error, EOFError) as exc:
        raise ValueError('Reference audio must be a readable PCM16 WAV file.') from exc
    return digest(reference)


def engine_identity(kind: str, files: list[Path], settings: dict) -> str:
    # Stable across cold starts; sensitive to weights, adapter code, and runtime packages.
    runtime = sorted((dist.metadata['Name'], dist.version) for dist in distributions())
    spec = {'adapter': digest(Path(__file__)), 'runtime': runtime,
            'weights': {path.name: digest(path) for path in files}, 'settings': settings}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return f'{kind}:{fingerprint}'


def write_pcm(target: Path, samples, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('Model returned empty or non-finite audio.')
    # All adapters publish uncompressed mono PCM16 compatible with the WAV merger.
    sf.write(str(target), samples, sample_rate, format='WAV', subtype='PCM_16')


class KokoroEngine:
    def __init__(self, directory: Path, voice: str = 'af_heart', speed: float = 1.0,
                 device: str = 'cuda', voices: tuple[str, ...] | None = None):
        from kokoro import KModel, KPipeline
        import torch
        if not 0.5 <= speed <= 2.0:
            raise ValueError('Kokoro speed must be between 0.5 and 2.0.')
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Kokoro device must be cpu or cuda.')
        # Initial comparison targets English editions: a = US, b = UK.
        self.available_voices = tuple(dict.fromkeys((voice, *(voices or ()))))
        if any(name[:1] not in {'a', 'b'} or not name.replace('_', '').isalnum()
               for name in self.available_voices):
            raise ValueError('Choose a US/UK English Kokoro voice, e.g. af_heart or bf_emma.')
        files = [directory / 'config.json', directory / 'kokoro-v1_0.pth']
        self.default_voice = voice
        self._identities = {
            name: engine_identity('kokoro', [*files, directory / 'voices' / f'{name}.pt'],
                                  {'voice': name, 'speed': speed, 'device': device})
            for name in self.available_voices
        }
        model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(files[0]), model=str(files[1]))
        model = model.to(device).eval()
        self._pipelines = {
            lang: KPipeline(lang_code=lang, repo_id='hexgrad/Kokoro-82M', model=model)
            for lang in {name[0] for name in self.available_voices}
        }
        self._voice_packs = {
            name: torch.load(directory / 'voices' / f'{name}.pt', map_location='cpu', weights_only=True)
            for name in self.available_voices
        }
        self.identity = self._identities[voice]
        self.pipeline = self._pipelines[voice[0]]
        self.voice = self._voice_packs[voice]
        self.speed = speed

    def for_voice(self, voice: str):
        if voice not in self.available_voices:
            raise ValueError(f'Unknown Kokoro voice {voice!r}. Choose: {", ".join(self.available_voices)}')
        # Request-specific view: share weights, without changing another request's narrator.
        selected = copy(self)
        selected.identity = self._identities[voice]
        selected.pipeline = self._pipelines[voice[0]]
        selected.voice = self._voice_packs[voice]
        return selected

    def synthesize(self, text: str, target: Path) -> None:
        import numpy as np
        import torch
        audio = []
        with torch.inference_mode():
            for _, _, samples in self.pipeline(text, voice=self.voice, speed=self.speed):
                if samples is not None:
                    audio.append(samples.detach().cpu().numpy())
        if not audio:
            raise ValueError('Kokoro produced no audio for this text.')
        write_pcm(target, np.concatenate(audio), 24000)


class ChatterboxEngine:
    def __init__(self, directory: Path, exaggeration: float = 0.5,
                 cfg_weight: float = 0.5, seed: int = 42,
                 reference_audio: Path | None = None):
        reference_hash = reference_fingerprint(reference_audio) if reference_audio is not None else None
        from chatterbox.tts import ChatterboxTTS
        files = [directory / name for name in
                 ('ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt')]
        self.identity = engine_identity('chatterbox', files, {
            'voice': 'reference' if reference_hash else 'builtin',
            'reference_sha256': reference_hash,
            'exaggeration': exaggeration, 'cfg_weight': cfg_weight,
            'seed': seed, 'device': 'cuda', 'internal_chunk_chars': 250,
        })
        self.model = ChatterboxTTS.from_local(str(directory), device='cuda')
        if reference_audio is not None:
            # Prepare once at startup; generate() then reuses these speaker conditionals.
            self.model.prepare_conditionals(str(reference_audio), exaggeration=exaggeration)
        self.exaggeration, self.cfg_weight, self.seed = exaggeration, cfg_weight, seed

    def synthesize(self, text: str, target: Path) -> None:
        import numpy as np
        import random
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        audio = []
        # Original Chatterbox limits generation length. Short subsegments reduce cutoffs.
        # All subsegments are concatenated before the outer chunk is checkpointed.
        with torch.inference_mode():
            for part in chunk_text(text, max_chars=250):
                samples = self.model.generate(part, exaggeration=self.exaggeration,
                                              cfg_weight=self.cfg_weight)
                audio.append(samples.detach().cpu().numpy().reshape(-1))
        if not audio:
            raise ValueError('Chatterbox produced no audio for this text.')
        write_pcm(target, np.concatenate(audio), self.model.sr)
