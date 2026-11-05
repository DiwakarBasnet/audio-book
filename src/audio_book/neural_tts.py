"""Optional neural adapters. Heavy libraries are imported only inside Modal images."""

import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path

from .text import chunk_text
from .tts import digest


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
                 device: str = 'cuda'):
        from kokoro import KModel, KPipeline
        import torch
        if not 0.5 <= speed <= 2.0:
            raise ValueError('Kokoro speed must be between 0.5 and 2.0.')
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Kokoro device must be cpu or cuda.')
        # Initial comparison targets English editions: a = US, b = UK.
        if voice[:1] not in {'a', 'b'} or not voice.replace('_', '').isalnum():
            raise ValueError('Choose a US/UK English Kokoro voice, e.g. af_heart or bf_emma.')
        files = [directory / 'config.json', directory / 'kokoro-v1_0.pth',
                 directory / 'voices' / f'{voice}.pt']
        self.identity = engine_identity('kokoro', files, {'voice': voice, 'speed': speed, 'device': device})
        model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(files[0]), model=str(files[1]))
        model = model.to(device).eval()
        self.pipeline = KPipeline(lang_code=voice[0], repo_id='hexgrad/Kokoro-82M', model=model)
        self.voice = torch.load(files[2], map_location='cpu', weights_only=True)
        self.speed = speed

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
                 cfg_weight: float = 0.5, seed: int = 42):
        from chatterbox.tts import ChatterboxTTS
        files = [directory / name for name in
                 ('ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt')]
        self.identity = engine_identity('chatterbox', files, {
            'voice': 'builtin', 'exaggeration': exaggeration, 'cfg_weight': cfg_weight,
            'seed': seed, 'device': 'cuda', 'internal_chunk_chars': 250,
        })
        self.model = ChatterboxTTS.from_local(str(directory), device='cuda')
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
