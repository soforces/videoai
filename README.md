# FrameScript

An AI-powered video-to-frames pipeline: upload a video, extract frames, and run a
per-frame decision engine that selectively applies face restoration, deblurring,
upscaling, background removal/segmentation, and OCR — only on the frames that
actually need it — before packaging the results into a downloadable zip.

## Architecture

The app is split into two processes on purpose:

- **`app.py`** — a thin Flask web API. It accepts the video upload, builds a job
  config, enqueues the job, and serves status/progress/preset/download endpoints.
  It never touches the GPU directly.
- **`worker.py`** — a long-running consumer process (built on **Huey** as the task
  queue, backed by SQLite — `huey.sqlite3` / `jobs.sqlite3`) that actually runs the
  pipeline. Keeping this in its own process means AI models stay resident in GPU
  memory across jobs instead of reloading per request, which is critical for
  staying under the project's sub-60s-per-job performance target.

A small singleton lock (`core/singleton_lock.py`) prevents two copies of either
process from running at once, and `core/job_store.py` tracks job status/progress
so the Flask side can poll it.

### Pipeline stages (`core/pipeline.py`)

1. **Extraction** (`extraction/ffmpeg_extract.py`) — pulls frames out of the video
   via **FFmpeg**, supporting multiple extraction modes (all frames, fixed FPS,
   N evenly-sampled frames, a timestamp range, or scene-change detection).
2. **Analysis** (`analysis/`) — per-frame blur scoring, perceptual-hash based
   duplicate/low-value detection, scene-change detection, face detection, OCR-text
   likelihood, and a general quality score.
3. **Decision engine** (`core/decision_engine.py`) — for every frame, decides
   *which* AI models (if any) should run, based on face-damage score, blur level,
   requested output resolution, and the selected processing mode
   (`fast` / `balanced` / `max_quality`). This is what keeps the pipeline fast:
   most frames skip most models entirely.
4. **AI plugin stages** (`plugins/`) — face restoration, deblurring/restoration,
   upscaling, background removal, segmentation, and OCR, applied only where the
   decision engine calls for them.
5. **Export** (`plugins/export/`) — writes processed frames (JPG/PNG), an optional
   contact sheet, and zips everything for download.

### Plugin system

Every AI model is wrapped in a small `BasePlugin` subclass (`plugins/base.py`)
with `load()` / `process()` / `unload()`, registered in a central `registry`
(`plugins/registry.py`). This keeps the pipeline pluggable: stages and models can
be added, removed, or reordered (`PipelineConfig.stage_order`) without touching
the orchestrator.

Because the target GPU has only 4GB of VRAM — far less than the combined footprint
of every model the project supports — plugins implement an **LRU eviction +
retry** scheme: on a CUDA out-of-memory error, the least-recently-used *other*
GPU-resident plugin is unloaded and the call retried, repeating until it succeeds
or there's nothing left to evict. Cached CUDA memory is also explicitly released
between frames to fight fragmentation.

## AI models / technologies used

| Stage | Model(s) |
|---|---|
| Face restoration | GFPGAN, CodeFormer |
| Deblur / general restoration | NAFNet, Restormer |
| Super-resolution / upscaling | Real-ESRGAN, SwinIR (and a Real-ESRGAN+SwinIR ensemble in max-quality mode) |
| Background removal | rembg |
| Segmentation | SAM2 (Segment Anything 2) |
| OCR | PaddleOCR |
| Image-quality / anomaly scoring | MUSIQ (via `pyiqa`), CLIP-based anomaly scoring |
| Frame interpolation | RIFE |
| Face detection | facexlib-based detector |

## Core technology stack

- **Python 3**, **PyTorch** (CUDA 11.8 build) as the deep-learning runtime
- **Flask** for the web API and UI (Jinja templates in `templates/`)
- **Huey** + **SQLite** for the background job queue
- **FFmpeg** for video decoding/frame extraction
- **OpenCV**, **NumPy**, **scikit-image**, **Pillow** for image I/O and processing
- **ImageHash** for perceptual-hash-based deduplication
- Model-specific libraries: `basicsr`, `facexlib`, `realesrgan`, `gfpgan`,
  `codeformer-pip`, `sam2`, `paddleocr`/`paddlepaddle`, `rembg`, `open_clip_torch`,
  `timm`, `transformers`

## Running it

Two processes, started separately:

```bash
python worker.py     # GPU worker / job consumer
python app.py         # Web API + UI
```

Then open `http://127.0.0.1:5000`.

## Processing modes

- **fast** — cheapest models only (GFPGAN for faces, NAFNet for blur,
  Real-ESRGAN for upscale), and only when a frame actually needs them.
- **balanced** — picks between GFPGAN/CodeFormer and NAFNet/Restormer based on
  damage/blur severity; adds an optional SwinIR fallback for upscaling.
- **max_quality** — prefers CodeFormer, runs the Real-ESRGAN+SwinIR ensemble,
  full Restormer, and allows SAM2 for segmentation/background removal.
