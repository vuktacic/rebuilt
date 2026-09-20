# Rebuilt

Rebuilt turns a short, fixed-camera phone recording of a small LEGO build into
an editable, frame-backed assembly guide. Programmer A owns the local FastAPI
backend and the shared API contract in `contracts/`.

## Run the backend

Create the local environment file once, then put your OpenAI key in it. The
file is ignored and the application reads simple `KEY=value` entries without
executing shell content. Existing process environment variables take priority.

```bash
cp .env.example .env
chmod 600 .env
${EDITOR:-vi} .env
```

Set `OPENAI_API_KEY` in `.env`. Keep the Hugging Face token in
`private/.hf_token`; it is loaded separately and never written to job output.
Gemini accepts `GEMINI_API_KEY` from the environment or `.env`; when that value
is empty, it reads the raw key from `GEMINI_API_KEY_FILE`, which defaults to
`private/.gemini_token`. Keep either credential file at mode `600`.

For the CUDA-backed SAM2 server, provision the environment, including the
optional Gemini SDK, and start the API and browser UI with one command:

```bash
./scripts/setup_server.sh
```

Use `./scripts/setup_server.sh --setup-only` to provision and validate without
starting the server. Set `INSTALL_GEMINI=0` if only the baseline/GPT runtime is
needed; the separate `scripts/bootstrap_gemini.sh` remains available for an
isolated Gemini environment.

After the SAM3 bootstrap below, start the local server with:

```bash
./scripts/run_backend.sh
```

Use `REBUILT_HOST=0.0.0.0` if the browser is outside WSL, or change
`REBUILT_PORT` when port 8000 is occupied.

For a UI-only walkthrough without a model or API key, open the server with
`?mock=1` appended to the URL. Mock mode never substitutes for live failures.

The manual equivalent remains available:

```bash
source .venv-sam3/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The server stores uploads and generated frames under `.data/` by default. Set
`REBUILT_DATA_DIR` to use a separate runtime directory. `OPENAI_MODEL` defaults
to `gpt-5.4`; the browser defaults to manual snapshots with
`MANUAL_COMPARE_MODEL=gpt-6-astra` and `MANUAL_WRITER_MODEL=gpt-5.6-luna`.
Credentials remain server-side. The API is documented in
[`contracts/README.md`](contracts/README.md).

The manual snapshot workflow extracts orientation-corrected frames at 5 FPS.
Choose the initial state and one settled state after each visible action, save
the storyboard, review each adjacent-pair finding, and explicitly include or
skip every pair before writing the editable guide. It does not use SAM2,
Gemini, automatic event detection, or automated guide verification.

Manual SAM2 jobs show replaceable piece-name and point suggestions after frame
extraction. Review and accept them before tracking. After the guide is drafted,
the browser can send the selected guide and bounded evidence frames to OpenAI
for an image-backed verification pass; corrections remain reviewable and user
edits invalidate an earlier verification.

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

To verify CUDA local segmentation and event extraction without calling OpenAI, run:

```bash
python scripts/benchmark_sam3.py testdata/IMG_3320.MOV --output reports/IMG_3320.sam3-cuda.json
```

### Apple Silicon macOS

On a native arm64 Mac (not Rosetta), bootstrap the isolated MLX environment and
pinned MXFP4 SAM3 snapshot, then launch the MLX runtime:

```bash
./scripts/bootstrap_mlx_sam3.sh
./scripts/run_backend_mlx.sh
```

`REBUILT_VISION_BACKEND=auto` resolves to `sam3-mlx` on native Apple Silicon
and to the existing CUDA runtime on supported CUDA hosts. Use explicit
`sam3-mlx` or `sam3-cuda` to override auto-detection; an unsupported selection
keeps `/health` live but rejects `/jobs` with a diagnostic. Run the
credential-free fixture report with:

```bash
unset PYTHONPATH
REBUILT_VISION_BACKEND=sam3-mlx .venv-mlx-sam3/bin/python \
  scripts/generate_fixture_report.py testdata/IMG_3320.MOV \
  --output reports/IMG_3320.sam3-mlx.json
```

The report has sanitized model/runtime provenance plus analysis, tracks, and
events; it never includes guide credentials or environment values. Set
`REBUILT_VISION_BACKEND=noop` only for local API/contract work without a model.
Keep the token file out of source control; it is already ignored by the
repository.

The MLX path defaults to a fast profile: SAM3 receives 336px inputs and checks
every other extracted frame. This keeps a fixed-camera demo responsive while
preserving the source frame IDs used for evidence. Set `MLX_SAM3_IMAGE_SIZE`
or `MLX_SAM3_FRAME_STRIDE=1` for a higher-detail pass; expect that to increase
local analysis time materially.

### Fast manual SAM2 reverse tracking on CUDA or Apple Silicon

The default `sam2` backend now uses a fast temporal profile: it propagates
every other extracted frame, always includes saved annotation frames and the
final frame, then restores source-frame IDs in the review track. This reduces
the default model work by about half while keeping the UI and guide references
in original video coordinates. Set `SAM2_FRAME_STRIDE=1` for maximum temporal
detail.

On a CUDA host, the standard SAM2 runtime uses the installed PyTorch CUDA build.
Bootstrap it, then start the same-origin API and browser UI together:

```bash
./scripts/bootstrap_sam2.sh
REBUILT_VISION_BACKEND=sam2 ./scripts/run_backend_sam2.sh
```

The bootstrap installs a CUDA PyTorch build on Linux when the default CUDA
index is used and keeps large installer temporary files under
`.third_party/sam2-tmp`; set `PYTORCH_INDEX_URL` or `SAM2_TMP_DIR` to override
either choice.

Open `http://127.0.0.1:8000/`, upload the recording, label its clearly separated
parts on the final frame, and select **Track backward**. The default stride of
two analyzes a 2 FPS extraction at one SAM2 inference frame per second; increase
`SAM2_FRAME_STRIDE` only after checking the warm-run timing on the demo GPU.

For a native MLX implementation of the same manual backward-tracking workflow,
use the separate pinned runtime:

```bash
./scripts/bootstrap_mlx_sam2.sh
./scripts/run_backend_mlx_sam2.sh
```

It defaults to SAM2.1 Hiera-Small at 768px with float16 memory and batched image
feature precomputation. Override the `SAM2_MLX_*` settings only when validating
quality against the source-frame overlay. `SAM2_VOS_OPTIMIZED=1` enables Meta's
experimental compiled PyTorch predictor; benchmark a warm server before using
it because compilation can make a first short video slower.
