# PDF to Audiobook

Convert a PDF into an audiobook with either Microsoft Edge TTS (local CLI) or OpenAI TTS. A small wizard guides you through choosing the backend, chunk size, and (optionally) estimating OpenAI cost before synthesis.

## Requirements

- Rust (1.88+ recommended)
- `ffmpeg` (for concatenating chunks)
- `poppler` / `pdftotext` (fallback extractor)
- One of the TTS backends:
  - **Edge TTS:** `edge-tts` CLI (Python). Install via `pip install edge-tts` inside a virtualenv.
  - **OpenAI TTS:** `OPENAI_API_KEY` set (starts with `sk-`), model `tts-1` or `tts-1-hd`.

## Running

```bash
cargo run -- /path/to/file.pdf
```

If no path is provided, the wizard will prompt for one. The wizard also asks:
- Backend: Edge TTS or OpenAI TTS.
- Chunk size (characters): larger reduces pauses; too large may time out. ~4000 is safe for OpenAI.
- For OpenAI: model, voice, and optional dry-run to estimate cost without generating audio.

Outputs:
- `audio_chunks/` (per-chunk text/MP3)
- `audiobook.mp3` in the project root

## Notes

- You can rerun after a failure; existing chunk MP3s are skipped.
- If you change chunk size or backend, delete `audio_chunks/` before re-running to regenerate with the new settings.
- Edge TTS is free but may be less natural; OpenAI TTS is more natural but billed per character (see current pricing).