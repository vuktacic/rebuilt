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

## Set up local SAM3 analysis

The backend loads Meta's SAM3 lazily for local segmentation and tracking. The
one-time bootstrap reads `private/.hf_token`, creates `.venv-sam3`, downloads
the official checkpoint into `.models/sam3`, and performs a CUDA smoke check:

```bash
./scripts/bootstrap_sam3.sh
source .venv-sam3/bin/activate
uvicorn backend.app.main:app --reload
```

The backend automatically uses the generated `.models/sam3/sam3-bf16.pt`
checkpoint to reduce model and checkpoint-load memory by roughly half. The
bootstrap reuses that file on later runs. Set `SAM3_CHECKPOINT` only to
override it.

Local tracking defaults to 2 FPS after the 10 FPS hardware benchmark took
about 12.5 minutes for a 15.9-second clip. This keeps the 0.9B model usable on
the target laptop while requiring two stable observations around a change.
Set `REBUILT_ANALYSIS_FPS=10` for a slower high-detail pass.

To verify local segmentation and event extraction without calling OpenAI, run:

```bash
python scripts/benchmark_sam3.py testdata/IMG_3320.MOV
```

Set `REBUILT_VISION_BACKEND=noop` only for local API/contract work without a
CUDA model. Keep the token file out of source control; it is already ignored by
the repository.
