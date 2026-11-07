"""Speech adapters share one small interface: identity and synthesize(text, path)."""

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Protocol
from urllib.request import Request, urlopen
from urllib.parse import urlencode, urlsplit
import wave


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


class SpeechEngine(Protocol):
    identity: str

    def synthesize(self, text: str, target: Path) -> None: ...


class PiperEngine:
    def __init__(self, model: Path, speed: float = 1.0):
        if not 0.5 <= speed <= 2.0:
            raise ValueError('Speed must be between 0.5 and 2.0.')
        config = Path(str(model) + '.json')
        if not model.is_file() or not config.is_file():
            raise ValueError(f'Expected model {model} and configuration {config}')
        try:
            from piper import PiperVoice, SynthesisConfig
            from importlib.metadata import version
        except ImportError as exc:
            raise ValueError('Local TTS requires: uv sync --extra tts') from exc
        self.identity = f'piper:{version("piper-tts")}:{digest(model)}:{digest(config)}:{speed}'
        self.voice = PiperVoice.load(str(model))
        self.config = SynthesisConfig(length_scale=1.0 / speed)

    def synthesize(self, text: str, target: Path) -> None:
        with wave.open(str(target), 'wb') as stream:
            self.voice.synthesize_wav(text, stream, syn_config=self.config)


class RemoteEngine:
    def __init__(self, endpoint: str, timeout: float = 300, voice: str | None = None):
        self.endpoint = endpoint.strip().rstrip('/')
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Endpoint must be an HTTP(S) base URL without credentials, query, or fragment.')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Request timeout must be a positive finite number of seconds.')
        self.timeout = timeout
        self.voice = voice
        self.headers = {'Content-Type': 'application/json'}
        # A scaled-to-zero Modal container may need time to initialize its model.
        info_url = self.endpoint + '/info'
        if voice is not None:
            info_url += '?' + urlencode({'voice': voice})
        print(f'Connecting to {info_url} (timeout {timeout:g}s)...', file=sys.stderr, flush=True)
        with urlopen(Request(info_url, headers=self.headers), timeout=self.timeout) as response:
            self.identity = json.load(response)['identity']
        print('Remote model ready.', file=sys.stderr, flush=True)

    def synthesize(self, text: str, target: Path) -> None:
        payload = {'text': text}
        if self.voice is not None:
            payload['voice'] = self.voice
        request = Request(self.endpoint + '/synthesize', headers=self.headers,
                          data=json.dumps(payload).encode())
        print(f'Requesting speech for {len(text)} characters...', file=sys.stderr, flush=True)
        with urlopen(request, timeout=self.timeout) as response:
            if response.headers.get('X-Voice-Identity') != self.identity:
                raise ValueError('Remote voice changed during this job. Restart with a new output directory.')
            target.write_bytes(response.read())
            print(f'Audio received (server cache: {response.headers.get("X-Audio-Cache", "unknown")}).',
                  file=sys.stderr, flush=True)
