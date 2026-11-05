# Modal deployment and model comparison

## Deploy the endpoints

Run these commands from the project root. The local machine only needs Python/uv and Calibre for MOBI input; Torch and model weights are installed in Modal images.

```bash
uv sync --extra modal --extra pdf
uv run --extra modal modal token new
```

Complete Modal's browser login, then deploy. No application secret is required:

```bash
uv run --extra modal modal deploy deployment/modal_app.py
```

The `/info` and `/synthesize` routes are public: anyone with the endpoint URL can call them.

Deployment builds **two separate images** and prints a base URL for each function, `kokoro` and `chatterbox`. Copy the URLs from your own deployment output; do not append `/synthesize` to the CLI's `--endpoint` value.

```bash
export KOKORO_URL='https://YOUR-KOKORO-ENDPOINT.modal.run'
export CHATTERBOX_URL='https://YOUR-CHATTERBOX-ENDPOINT.modal.run'
curl --fail "$KOKORO_URL/info"
curl --fail "$CHATTERBOX_URL/info"
```

For an ephemeral development deployment, use `uv run --extra modal modal serve deployment/modal_app.py` instead. Its URLs last while that command runs.

The implementation follows Modal's [ASGI app deployment](https://modal.com/docs/guide/webhooks) and [model weight storage](https://modal.com/docs/guide/model-weights) interfaces. This repository provides deployment code; no cloud deployment or GPU inference was performed during implementation.

## Models and configuration

| Endpoint | Runtime | Narrator/settings | Output |
| --- | --- | --- | --- |
| `kokoro` | Kokoro 0.9.4 / Kokoro-82M | `am_onyx`, US English, speed 1.0 | Mono PCM16, 24 kHz |
| `chatterbox` | chatterbox-tts 0.1.6, original English Chatterbox | Built-in voice, exaggeration 0.5, CFG 0.5, seed 42 | Mono PCM16, model sample rate |

This compares the original English Chatterbox model, rather than Turbo or Multilingual. The initial adapters target an **English edition** of your book. They do not provide a fair comparison for a Spanish-language edition. Kokoro supports more languages upstream, but this deployment validates US/UK English voices only.

Change constants at the top of `deployment/modal_app.py` and redeploy to try another narrator or configuration. Kokoro voice changes also change the voice file downloaded into its image. Remote speed is configured in the deployment, not with client `--speed`. Chatterbox uses its built-in reference voice; custom voice cloning is not exposed. Its seed reduces random variation, without guaranteeing bit-identical GPU output.

Both endpoints request one L4 GPU, scale to zero after 120 idle seconds, and allow at most one container each. Each container loads one model and serializes inference; a simultaneous synthesis request receives HTTP 429. After a failed request, rerun the render command. The client does not automatically retry.

Images download the primary weights with `hf download` during build. The package versions are pinned separately for each model; they are not installed in your local environment. Primary weights use the repositories' default revisions at build time. Their actual file hashes are included in the engine identity, so a rebuild with changed weights cannot silently reuse another voice's job. Modal may cache an unchanged build layer. To select a specific weight revision, add `--revision COMMIT_SHA` to its `hf download` command and redeploy.

The reference APIs are [Kokoro](https://github.com/hexgrad/kokoro) and [Chatterbox](https://github.com/resemble-ai/chatterbox). Main weights are baked into the images; library auxiliary resources may still be fetched at startup. The first build and cold request can take longer. The client allows up to 300 seconds for model info and each synthesis request; Modal's web request limits also apply. Use small chunks if a request times out.

## Compare the same text

Start with the bundled example. Prepare two independent jobs with identical chunk sizes so each model narrates the same passages:

```bash
uv run audiobook prepare examples/sample.txt --output output/kokoro-demo --chunk-size 300
uv run audiobook prepare examples/sample.txt --output output/chatterbox-demo --chunk-size 300
uv run audiobook render output/kokoro-demo --endpoint "$KOKORO_URL" --limit 3
uv run audiobook render output/chatterbox-demo --endpoint "$CHATTERBOX_URL" --limit 3
```

Listen to `output/kokoro-demo/sample.wav` and `output/chatterbox-demo/sample.wav`. Compare pacing, pronunciation, sentence endings, repeated/skipped words, and voice consistency. Listen to names and Spanish place names from your actual book before choosing. A small generic sample cannot establish whole-book quality.

For the book, extract once, inspect/clean the text, then use that same text for both jobs:

```bash
uv run audiobook prepare "books/The Motorcycle Diaries.mobi" --output output/motorcycle-source
# Review output/motorcycle-source/text.txt before proceeding.
uv run audiobook prepare output/motorcycle-source/text.txt --output output/motorcycle-kokoro --chunk-size 300
uv run audiobook prepare output/motorcycle-source/text.txt --output output/motorcycle-chatterbox --chunk-size 300
uv run audiobook render output/motorcycle-kokoro --endpoint "$KOKORO_URL" --limit 5
uv run audiobook render output/motorcycle-chatterbox --endpoint "$CHATTERBOX_URL" --limit 5
```

Do not switch models inside one job directory. Its stored engine identity deliberately prevents that. When satisfied, drop `--limit` for your chosen model:

```bash
uv run audiobook render output/motorcycle-kokoro --endpoint "$KOKORO_URL" --mp3
# Or render output/motorcycle-chatterbox using "$CHATTERBOX_URL".
```

The first sample chunks are reused during the full render. Chatterbox further splits long chunks into at most 250-character subsegments to reduce generation cutoffs; these are concatenated into one checkpointed outer WAV. Listen for cutoffs anyway: a structurally valid WAV does not prove that a generative model spoke every word.

For a runtime comparison, measure the first uncached render of each job separately and record audio duration and elapsed time. Repeating a render measures local reuse or server cache speed. Both GPUs and persistent audio Volumes incur Modal usage; model comparison requires two deployments but idle endpoints scale to zero.

## How resume avoids repeated TTS

The pipeline has two checkpoints:

1. **Local job:** `manifest.json` holds the ordered text chunks, engine identity, and per-chunk WAV and text SHA-256 hashes. After a successful response, the client validates the WAV, renames its temporary file to `chunks/000000.wav`, and atomically saves the updated manifest. On restart, a matching WAV/text checkpoint prints `reused` and skips `/synthesize` entirely. The CLI still fetches `/info` once to verify the model identity. Missing, changed, or unfinished chunks are requested again. Old audio-only manifests remain readable under the original assumption that their prepared text is immutable.
2. **Modal audio cache:** each endpoint hashes `[engine_identity, exact_text]` and checks its persistent Volume for that WAV. A cache hit returns saved audio without inference. A miss synthesizes, validates, atomically publishes the WAV, and commits the Volume **before** returning the HTTP response. A response lost after that commit can therefore be recovered on the next request without running the model again. `X-Audio-Cache: hit` or `miss` reports this server behavior.

Engine identities include model/voice weight hashes, settings, neural adapter code, and installed package versions. Switching voice, speed, weights, or runtime can change the identity; prepare a new output directory rather than mixing old and new audio. Kokoro and Chatterbox caches remain separate.

For example, if chunks 1–40 are saved and the process stops during chunk 41, rerunning the same command reuses 1–40 and requests 41 onward. If Modal already saved chunk 41 but its response was lost, it returns that cached WAV. Final WAV assembly and optional MP3 encoding are repeated; they do not trigger TTS for completed chunks.

This is checkpointed recovery, **not an exactly-once inference guarantee**. A server crash before its audio is durably committed can require that in-flight chunk to be synthesized again. A local crash between publishing a WAV and updating the manifest can cause another HTTP request; a committed Modal cache avoids another inference in that case. Keep the local manifest/chunks and the Modal Volumes to preserve recovery. The guarantee concerns reuse of saved results, not deduplication of work that never reached a durable checkpoint.

The Volumes `audiobook-kokoro-audio` and `audiobook-chatterbox-audio` store generated speech until you remove it; there is no automatic retention policy. No ebook upload is needed, but text chunks reach Modal and their generated audio persists there. Use one app deployment per cache Volume: `max_containers=1` plus the in-process lock coordinates this single-user service, not independent deployments sharing a Volume.

## Verify locally

```bash
uv sync --extra modal --group dev
uv run python -m unittest discover -s tests -v
```

Tests use synthetic PCM and cover interrupted jobs, checksum repair, sample-to-full reuse, concurrent-writer rejection, API access without credentials and input validation, persistent cache reuse across app restarts, model changes, and a failed cache commit. They do not download models or evaluate GPU inference/speech quality.

## If render appears silent

The CLI now validates that your job directory contains `manifest.json` before loading a model or contacting Modal. Pass the actual directory used by `prepare`: if your manifest is `output/manifest.json`, render `output`, not `output/motorcycle-kokoro`.

Remote rendering prints `Connecting to .../info` before the network request, `Remote model ready` after startup, and a message before every synthesis request. If it stays at `Connecting`, use `--request-timeout 30` for a quicker network diagnostic. Cold model startup can exceed 30 seconds; restore the default 300-second timeout for normal rendering.

If even `Connecting` or a validation error never appears, try the installed CLI directly with `.venv/bin/audiobook render ...`, or use `uv run --no-sync audiobook render ...` to bypass uv environment synchronization. No book text is printed by the client progress messages.

### Requests stay pending with no container logs

A pending web request does not prove the model is loading. Modal may still be waiting to schedule a container. Check the function's queued inputs and active containers, alongside workspace GPU limits and billing/credit status. If a live request shows backlog greater than zero and zero containers throughout, model inference has not started. These counts do not identify the scheduler's underlying reason.

Kokoro can also run on CPU. To bypass its GPU requirement, change the constant in `deployment/modal_app.py`:

```python
KOKORO_DEVICE = 'cpu'
```

Then redeploy:

```bash
uv run --extra modal modal deploy deployment/modal_app.py
```

Use the Kokoro URL printed by that deployment and retry the same render command. This deploys Kokoro with no GPU request; Chatterbox retains its L4 requirement. The CPU setting participates in the model identity, so use a new job if the old one already has an engine identity. CPU narration may be slower. The image still contains Torch's GPU-capable libraries, but the Kokoro function requests CPU resources only.

If CPU starts and GPU does not, investigate GPU allocation/limits. If CPU also stays queued without containers, investigate workspace-wide limits, billing, and deployment state. Do not increase client timeouts indefinitely to mask the absence of workers. Restore `KOKORO_DEVICE = 'cuda'` and redeploy when GPU scheduling is resolved.
