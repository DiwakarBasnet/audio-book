"""Deploy with: uv run modal deploy deployment/modal_app.py

Two independently built model services expose the same HTTP API. No local GPU needed.
"""

import modal

app = modal.App('audiobook-tts')
kokoro_cache = modal.Volume.from_name('audiobook-kokoro-audio', create_if_missing=True)
chatterbox_cache = modal.Volume.from_name('audiobook-chatterbox-audio', create_if_missing=True)

# Change these constants, then redeploy, to change the narrator/settings.
KOKORO_VOICE = 'am_onyx'
KOKORO_SPEED = 1.0
# Set to 'cpu' to deploy Kokoro without requesting a GPU. Redeploy after changing.
KOKORO_DEVICE = 'cuda'
CHATTERBOX_EXAGGERATION = 0.5
CHATTERBOX_CFG_WEIGHT = 0.5
CHATTERBOX_SEED = 42

kokoro_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('espeak-ng', 'libsndfile1')
    .pip_install('torch==2.6.0', 'kokoro==0.9.4', 'soundfile==0.13.1',
                 'transformers==4.51.3', 'huggingface-hub[cli]==0.36.0',
                 'fastapi==0.115.12', 'numpy==1.26.4', 'spacy==3.8.7',
                 'click==8.1.8')
    .pip_install('https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl')
    # spaCy imports click through its CLI even during normal library imports.
    # Verify this dependency chain on CPU during build, before requesting a GPU.
    .run_commands(
        'python -c "import click; from kokoro import KModel, KPipeline; '
        "import spacy; spacy.load('en_core_web_sm'); "
        "print('Kokoro imports and English language model OK')\""
    )
    .run_commands(
        f'hf download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/{KOKORO_VOICE}.pt --local-dir /models/kokoro'
    )
    .add_local_python_source('audio_book')
)

chatterbox_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libsndfile1', 'ffmpeg', 'git')
    .pip_install('chatterbox-tts==0.1.6', 'soundfile==0.13.1',
                 'huggingface-hub[cli]==0.36.0', 'fastapi==0.115.12')
    .run_commands(
        'hf download ResembleAI/chatterbox ve.safetensors t3_cfg.safetensors s3gen.safetensors tokenizer.json conds.pt --local-dir /models/chatterbox'
    )
    .add_local_python_source('audio_book')
)


@app.function(image=kokoro_image, gpu='T4' if KOKORO_DEVICE == 'cuda' else None,
              cpu=2, memory=8192,
              volumes={'/audio-cache': kokoro_cache},
              timeout=600, scaledown_window=120, max_containers=1)
@modal.asgi_app()
def kokoro():
    from pathlib import Path
    from audio_book.neural_tts import KokoroEngine
    from audio_book.server import create_app
    return create_app(
        lambda: KokoroEngine(Path('/models/kokoro'), KOKORO_VOICE, KOKORO_SPEED,
                             device=KOKORO_DEVICE),
        cache_dir=Path('/audio-cache'), commit_cache=kokoro_cache.commit,
    )


@app.function(image=chatterbox_image, gpu='T4', cpu=2, memory=16384,
              volumes={'/audio-cache': chatterbox_cache},
              timeout=600, scaledown_window=120, max_containers=1)
@modal.asgi_app()
def chatterbox():
    from pathlib import Path
    from audio_book.neural_tts import ChatterboxEngine
    from audio_book.server import create_app
    return create_app(
        lambda: ChatterboxEngine(Path('/models/chatterbox'), CHATTERBOX_EXAGGERATION,
                                 CHATTERBOX_CFG_WEIGHT, CHATTERBOX_SEED),
        cache_dir=Path('/audio-cache'), commit_cache=chatterbox_cache.commit,
    )
