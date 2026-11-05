"""Durable, sequential jobs. Interrupted chunks are regenerated on the next run."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

from .text import chunk_text, extract
from .tts import SpeechEngine, digest


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def job_lock(directory: Path):
    # flock releases on exit/crash; no stale-lock deletion required. Linux/macOS deployment.
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process is using this output directory.') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare(source: Path, directory: Path, max_chars: int) -> dict:
    with job_lock(directory):
        if (directory / 'manifest.json').exists():
            raise ValueError('Output already contains a job. Use a new directory or render the existing job.')
        text = extract(source)
        chunks = chunk_text(text, max_chars)
        manifest = {'schema': 1, 'source': source.name, 'source_sha256': digest(source),
                    'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'chunks': chunks, 'engine': None, 'completed': {}}
        (directory / 'text.txt').write_text(text, encoding='utf-8')
        write_json(directory / 'manifest.json', manifest)
        return manifest


def validate_wav(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), 'rb') as stream:
        if stream.getcomptype() != 'NONE' or stream.getnframes() == 0:
            raise ValueError(f'Invalid or empty PCM WAV: {path}')
        expected = stream.getnframes() * stream.getnchannels() * stream.getsampwidth()
        size = 0
        while block := stream.readframes(65536):
            size += len(block)
        if size != expected:
            raise ValueError(f'Truncated WAV: {path}')
        return stream.getnchannels(), stream.getsampwidth(), stream.getframerate()


def join_wavs(paths: list[Path], target: Path) -> None:
    temporary = target.with_suffix('.wav.tmp')
    audio_format = validate_wav(paths[0])
    with wave.open(str(temporary), 'wb') as output:
        output.setparams((*audio_format, 0, 'NONE', 'not compressed'))
        for path in paths:
            if validate_wav(path) != audio_format:
                raise ValueError('TTS returned inconsistent audio formats.')
            with wave.open(str(path), 'rb') as source:
                while block := source.readframes(65536):
                    output.writeframesraw(block)
    temporary.replace(target)


def render(directory: Path, engine: SpeechEngine, limit: int | None = None, mp3: bool = False) -> Path:
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive.')
    with job_lock(directory):
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['engine'] not in (None, engine.identity):
            raise ValueError('Voice or speed differs from this job. Prepare a new output directory.')
        manifest['engine'] = engine.identity
        write_json(manifest_path, manifest)
        chunk_dir = directory / 'chunks'
        chunk_dir.mkdir(exist_ok=True)
        chunks = manifest['chunks'][:limit]
        if not chunks:
            raise ValueError('Job contains no text chunks.')
        paths = []
        for index, text in enumerate(chunks):
            key = str(index)
            path = chunk_dir / f'{index:06d}.wav'
            record = manifest['completed'].get(key)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            # Read old audio-only checkpoints, while writing text-bound records from now on.
            if isinstance(record, str):
                record = {'audio_sha256': record, 'text_sha256': text_hash}
            reusable = (path.exists() and isinstance(record, dict)
                        and record.get('text_sha256') == text_hash
                        and record.get('audio_sha256') == digest(path))
            if not reusable:
                temporary = path.with_suffix('.wav.tmp')
                engine.synthesize(text, temporary)
                validate_wav(temporary)
                temporary.replace(path)
                manifest['completed'][key] = {'audio_sha256': digest(path), 'text_sha256': text_hash}
                write_json(manifest_path, manifest)
            else:
                validate_wav(path)
            paths.append(path)
            action = 'reused' if reusable else 'saved'
            print(f'Chunk {index + 1}/{len(chunks)} {action}', flush=True)
        stem = 'sample' if limit is not None else 'audiobook'
        target = directory / f'{stem}.wav'
        join_wavs(paths, target)
        if mp3:
            ffmpeg = shutil.which('ffmpeg')
            if ffmpeg is None:
                raise ValueError(f'WAV saved at {target}. Install FFmpeg to export MP3.')
            target = directory / f'{stem}.mp3'
            temporary = target.with_suffix('.tmp.mp3')
            result = subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(directory / f'{stem}.wav'),
                                     '-codec:a', 'libmp3lame', '-b:a', '128k', str(temporary)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f'FFmpeg failed: {result.stderr[-2000:]}')
            temporary.replace(target)
        return target
