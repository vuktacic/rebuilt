# Rebuilt

Rebuilt turns a short, fixed-camera phone recording of a small LEGO build into
an editable, frame-backed assembly guide. Programmer A owns the local FastAPI
backend and the shared API contract in `contracts/`.

## Run the backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
uvicorn backend.app.main:app --reload
```

The server stores uploads and generated frames under `.data/` by default. Set
`REBUILT_DATA_DIR` to use a separate runtime directory. `OPENAI_MODEL` defaults
to `gpt-5.4`; credentials remain server-side. The API is documented in
[`contracts/README.md`](contracts/README.md).

For a quick request, upload a supported video as the `video` multipart field:

```bash
curl -F video=@build.mp4 http://localhost:8000/jobs
curl http://localhost:8000/jobs/<jobId>
```

FFmpeg must be installed and available as `ffmpeg` (or configured with
`FFMPEG_BINARY`). The current backend accepts one active processing job and
limits uploads to 250 MiB.
