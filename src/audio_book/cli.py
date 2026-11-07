import argparse
from pathlib import Path
import sys

from .pipeline import prepare, render
from .tts import PiperEngine, RemoteEngine


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert ebooks into resumable audiobooks.')
    commands = parser.add_subparsers(dest='command', required=True)
    ingest = commands.add_parser('prepare', help='Extract and chunk text for review, without loading a model')
    ingest.add_argument('source', type=Path)
    ingest.add_argument('--output', type=Path, default=Path('output'))
    ingest.add_argument('--chunk-size', type=int, default=800)
    speak = commands.add_parser('render', help='Generate or resume audio from a prepared job')
    speak.add_argument('job', type=Path)
    backend = speak.add_mutually_exclusive_group(required=True)
    backend.add_argument('--model', type=Path, help='Local Piper .onnx model')
    backend.add_argument('--endpoint', help='Hosted TTS API base URL')
    speak.add_argument('--voice', help='Kokoro voice name, e.g. am_onyx or af_heart (requires --endpoint)')
    speak.add_argument('--speed', type=float, default=1.0, help='Local voice speed multiplier (0.5–2)')
    speak.add_argument('--limit', type=int, help='Render only the first N chunks to sample.wav')
    speak.add_argument('--mp3', action='store_true', help='Also encode MP3 using FFmpeg')
    speak.add_argument('--request-timeout', type=float, default=300,
                       help='Timeout in seconds for each remote request (default: 300)')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            manifest = prepare(args.source, args.output, args.chunk_size)
            print(f'Prepared {len(manifest["chunks"])} chunks. Review {args.output / "text.txt"}.')
        else:
            if not (args.job / 'manifest.json').is_file():
                raise ValueError(f'No prepared job at {args.job}. Expected manifest.json; '
                                 'use the directory created by audiobook prepare.')
            if args.endpoint and args.speed != 1.0:
                raise ValueError('Remote speed is configured on the server with TTS_SPEED.')
            if args.voice is not None and args.endpoint is None:
                raise ValueError('--voice requires a remote --endpoint. Local Piper uses its model voice.')
            engine = (RemoteEngine(args.endpoint, timeout=args.request_timeout, voice=args.voice)
                      if args.endpoint else PiperEngine(args.model, args.speed))
            print(render(args.job, engine, args.limit, args.mp3))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
