# Architecture and implementation phases

## Objective

Convert a locally supplied ebook, especially a DRM-free MOBI edition of *The Motorcycle Diaries*, into a personal audiobook. No book or model is bundled. CPU inference is the initial baseline; Python 3.11 is the documented runtime to avoid depending on Python 3.14 support across native ML packages.

## Data flow

```text
TXT / PDF / MOBI / EPUB / other supported formats
    -> text.extract (UTF-8 / pypdf / Calibre)
    -> normalize + bounded chunks
    -> text.txt + manifest.json (reviewable job)
    -> SpeechEngine (PiperEngine or RemoteEngine)
    -> checksummed WAV chunks + incremental manifest
    -> streaming WAV assembly -> optional FFmpeg MP3
```

The CLI orchestrates jobs; extraction knows nothing about voices. The pipeline depends on the `SpeechEngine` protocol. Piper runs locally; `KokoroEngine` and `ChatterboxEngine` run in separate Modal GPU images behind the same HTTP contract. The model server never needs Calibre or access to the original ebook.

## Phase 1 — ingest and review (implemented)

`text.py`, `pipeline.prepare`, and CLI `prepare` extract text, normalize paragraph whitespace, and split paragraphs into bounded pieces, preferring sentence boundaries. MOBI and EPUB go through Calibre, which handles their internal reading order. PDF uses pypdf and requires an existing text layer. TXT and Markdown use UTF-8; Markdown syntax is not stripped.

Artifacts: source hash, normalized text hash, immutable ordered chunks, readable text preview. Review before paying the inference cost. Chapter detection, page-header removal, and OCR are deliberately left to a later ingestion enhancement; automatic cleanup can accidentally delete narration. To correct text, edit a copy of `text.txt` and prepare that TXT into a new job. Render reads the manifest, so editing an existing preview does not change its chunks.

## Phase 2 — local narration and recovery (implemented)

`tts.PiperEngine`, `pipeline.render`, and CLI `render` load one voice per invocation, synthesize sequentially, and save a manifest after each completed chunk. New checkpoints record SHA-256 hashes of both the WAV and its text; old audio-only checkpoints remain readable. A missing or modified chunk is requested again; an interrupted temporary file is ignored. Model weights, voice configuration, Piper package version, and speed participate in the engine identity. Switching identity requires a new job directory. Logs distinguish `saved` and `reused` chunks.

An advisory OS file lock prevents two processes writing one job simultaneously. This uses `fcntl` and targets Linux/macOS, including WSL. Final assembly copies audio frames in bounded blocks rather than loading the entire audiobook into memory. Output publication uses temporary files plus rename. An interrupted merge is repeated on resume. Chunk files are retained for recovery.

`--limit N` creates `sample.wav` without marking the entire book complete. A subsequent full render reuses those chunks. Output is one continuous WAV or MP3, without chapter metadata. Standard WAV has a roughly 4 GiB size limit; for exceptionally long books, prepare multiple text sections as independent jobs. Allow disk space for chunks plus the assembled WAV and optional MP3.

## Phase 3 — deploy the model (implemented)

`server.py` exposes an unauthenticated `GET /info` and `POST /synthesize` (up to 4,000 text characters), plus an unauthenticated readiness endpoint `GET /health`. The FastAPI lifespan loads the model once. Use one Uvicorn worker per model instance. One request runs at a time; concurrent synthesis returns 429. A failed remote request stops the client; rerunning resumes from its checkpoint.

The remote adapter obtains the model identity before rendering and checks it again on every audio response, preventing silently mixed voices after a server rollout. Deploy the supplied Docker image to a CPU VM or container host, and mount a voice directory. In this split deployment the book, extraction tools, and job state stay on the client; only text chunks reach the server. HTTP clients wait up to 300 seconds per chunk; select smaller chunks on slower hosts.

## Follow-on phases (not implemented)

- Human-reviewed chapter segmentation and M4B output with chapter metadata.
- OCR and configurable pronunciation dictionaries for Spanish names and place names.
- A persistent queue and job API if multiple users or browser uploads become necessary.

A distributed queue, database, and upload UI are unnecessary for the first single-user audiobook. Whole-book work belongs in the resumable CLI rather than a long-running HTTP request.

## External interfaces

- Piper Python API: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
- Voice download CLI: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
- Calibre conversion: https://manual.calibre-ebook.com/generated/en/ebook-convert.html

Piper's current engine is GPL-licensed; voice models have their own licenses. Inspect the model card before selecting a voice for redistribution or commercial deployment.

## Phase 4 — Modal and model comparison (implemented)

`deployment/modal_app.py` defines independent Kokoro and original English Chatterbox endpoints. Separate images isolate ML dependencies. Primary model files are downloaded at image build time using `hf download`. The shared `server.create_app` loads each engine once in its lifespan, validates generated audio, and publishes PCM16 WAVs. Both endpoints use L4 GPUs, scale to zero, and cap scaling at one container per endpoint.

Neural engine identities bind weight/configuration hashes, synthesis settings, adapter code, and runtime package versions. The remote adapter reads `/info` and verifies the same identity on every WAV response. Prepared job directories remain model-specific, enabling side-by-side comparison of the same extracted text without mixed-model output.

Each endpoint also has a persistent Modal audio Volume. A SHA-256 of engine identity plus exact request text addresses its cached WAV. Successful inference is committed before the response is sent. This covers the lost-response gap after remote completion; local resume avoids the request entirely when a valid client checkpoint exists. Failed work before a durable checkpoint can still be repeated. The cache is designed for one deployment per Volume and retains speech until manually removed.

See [MODAL.md](MODAL.md) for deployment, comparison, and the precise resume guarantees.

## Reference narration with Chatterbox

`CHATTERBOX_REFERENCE_AUDIO` optionally points to a local PCM16 WAV. Deployment validates and copies that sample into the image; the remote engine validates it again and prepares speaker conditionals once at startup. Every generated subsegment then reuses that narrator. The sample's content hash participates in engine identity, preventing resume/cache reuse across different reference voices. No reference path or audio upload is accepted by the public HTTP API: narrator selection is a deployment configuration. See [MODAL.md](MODAL.md) for sample preparation and commands.

## Kokoro voice selection per request

The deployment bundles `am_onyx` and `af_heart`. One `KokoroEngine` loads the model and available voice tensors once. `for_voice` returns a request-specific view sharing the model and language pipeline, with the selected tensor and identity; it does not mutate the default narrator. The HTTP contract accepts optional `voice` in synthesis JSON and `/info` query parameters. The remote client uses the same voice in both requests. Local checkpoints and server cache keys therefore distinguish identical text spoken by different voices. Single-voice endpoints reject unsupported voice selection instead of ignoring it.
