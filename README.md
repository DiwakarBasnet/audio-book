# Ebook to Audiobook

## Objective

Convert novels and biographies from MOBI, EPUB, PDF, or TXT into audiobooks, with *The Motorcycle Diaries* as the initial target. Use Kokoro or Chatterbox on Modal, or Piper locally. Chatterbox supports both its default voice and cloning from a sample recording.

## How it works

1. `prepare` extracts the book's text and splits it into chunks. The job directory contains `text.txt` for review and `manifest.json` for narration and progress.
2. `render` sends each chunk to the selected model and saves its audio. It produces a sample or combines the chunks into a full WAV, with optional MP3 output.
3. Rerunning the same render command resumes the job. Completed chunks are checked and reused; Modal also caches generated audio by text and voice.

Use a separate job directory for each model or cloned voice. Scanned PDFs need OCR first; DRM-protected books are unsupported.

## Commands

Run from the project root using Python 3.11 and `uv`.

### Install

```bash
uv sync --extra modal --extra pdf
mkdir -p books output voices

# Debian/Ubuntu: Calibre handles MOBI/EPUB; FFmpeg exports MP3.
sudo apt-get update
sudo apt-get install calibre ffmpeg
```

### Deploy Kokoro and Chatterbox to Modal

```bash
uv run --extra modal modal token new
uv run --extra modal modal deploy deployment/modal_app.py
```

Copy the endpoint URLs printed by deployment. No application token is required. Model settings are in `deployment/modal_app.py`; redeploy after changing them. Set `KOKORO_DEVICE = 'cpu'` to run Kokoro without a GPU.

### Prepare a book

```bash
uv run audiobook prepare "books/The Motorcycle Diaries.mobi" \
  --output output/motorcycle-kokoro --chunk-size 300
```

Review `output/motorcycle-kokoro/text.txt` before narration. To correct it, prepare an edited copy into a new job directory.

### Generate a sample, then the full audiobook

Replace the dummy endpoint with your deployed Kokoro URL:

```bash
uv run audiobook render output/motorcycle-kokoro \
  --endpoint https://your-workspace--audiobook-tts-kokoro.us-east.modal.run \
  --voice af_heart --limit 3

uv run audiobook render output/motorcycle-kokoro \
  --endpoint https://your-workspace--audiobook-tts-kokoro.us-east.modal.run \
  --voice af_heart --mp3
```

Kokoro accepts `--voice am_onyx` or `--voice af_heart`. Omit `--voice` to use the configured default (`am_onyx`). Keep the same voice when resuming; use a new job directory to switch voices.

Outputs are `sample.wav`, `audiobook.wav`, and, with `--mp3`, `audiobook.mp3`. Rerun the same command after an interruption to resume.

### Use Chatterbox

```bash
uv run audiobook prepare "books/The Motorcycle Diaries.mobi" \
  --output output/motorcycle-chatterbox --chunk-size 300

uv run audiobook render output/motorcycle-chatterbox \
  --endpoint https://your-workspace--audiobook-tts-chatterbox.us-east.modal.run \
  --limit 3
```

Remove `--limit 3` for the full audiobook; add `--mp3` if needed.

For the default voice, keep this setting in `deployment/modal_app.py`:

```python
CHATTERBOX_REFERENCE_AUDIO = None
```

For cloning, create a short PCM16 WAV recording:

```bash
ffmpeg -i sample.mp3 -t 10 -ac 1 -ar 24000 -c:a pcm_s16le voices/narrator.wav
```

Then set:

```python
CHATTERBOX_REFERENCE_AUDIO = 'voices/narrator.wav'
```

Redeploy and prepare a new job directory for the cloned voice. Use the same Chatterbox render command with that directory.

### Run Piper locally

```bash
uv sync --extra tts --extra pdf
mkdir -p models
uv run python -m piper.download_voices en_US-lessac-medium --data-dir models
uv run audiobook prepare examples/sample.txt --output output/piper-demo
uv run audiobook render output/piper-demo \
  --model models/en_US-lessac-medium.onnx --limit 3
```
