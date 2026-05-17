# game-gen-ai

A small offline pipeline that turns a short, natural-language prompt into a
stylised cartoon image. A local LLM first expands the prompt into a detailed,
SD-idiomatic English description; an SD-family checkpoint then renders it.
Both text-to-image and image-to-image modes are supported. Runs on Apple
Silicon (MPS), CUDA, or CPU.

The aesthetic target is the 3D-rendered, stylised cartoon look common in
mobile casual games — friendly characters, warm lighting, polished
materials, saturated palette.

## Architecture

```
user prompt (any language, possibly very short)
        │
        ▼
   src/llm.py          Ollama → llama3.2:3b
   (system prompt instructs the model to act as an SD prompt engineer:
    translate to English, bake in style/lighting/composition tags
    appropriate for stylised 3D cartoon mobile-game art)
        │
        ▼
   detailed English SD prompt (40-70 words, comma-tag style)
        │
        ▼
   src/image_gen.py    diffusers → AutoPipelineForText2Image / Image2Image
        ├─ default:   Lykon/dreamshaper-8 (SD 1.5 finetune, 3D cartoon)
        ├─ swap-in:   any HF SD 1.5 / SDXL checkpoint via --model flag
        └─ on MPS:    SD 1.5 in fp32; SDXL in fp16 with
                      madebyollin/sdxl-vae-fp16-fix VAE (see "Precision" below)
        │
        ▼
   outputs/{timestamp}.png  +  outputs/{timestamp}.json (metadata)
```

Two design choices worth calling out:

1. **LLM in front of SD as a prompt engineer.** Short user prompts and
   non-English prompts both produce mediocre results from SD's CLIP text
   encoder. The local LLM translates to English and bakes in
   style/lighting/composition tags. The diffusion model only sees clean,
   stylised English prompts. This also lets the *same* CLI handle
   `"a brave knight"` and `"中世纪城堡，雷雨夜"` without any branching
   in the SD code.

2. **Auto-routing between SD 1.5 and SDXL pipelines.** A single
   `--model <hf-repo-id>` flag works for both families because
   `AutoPipelineForText2Image` / `AutoPipelineForImage2Image` inspect each
   checkpoint's `model_index.json` to pick the right pipeline class. The
   per-device dtype/VAE logic in `_pipeline_kwargs()` then routes around
   each backend's known precision quirks.

## Layout

| Path | Role |
|---|---|
| `src/llm.py` | `enhance_prompt(user_text)` → Ollama call, returns SD-ready English |
| `src/image_gen.py` | `load_pipeline()`, `load_img2img_pipeline()`, `generate()`, `generate_img2img()` — diffusers wrapper, auto-handles SDXL vs SD 1.5 + per-device dtype |
| `src/main.py` | CLI entry; orchestrates LLM → SD, writes png + json sidecar |
| `outputs/` | Generated images and metadata (gitignored) |
| `pyproject.toml` | uv-managed dependency list |

## Setup

```bash
# install local tooling
brew install uv ollama
brew services start ollama
ollama pull llama3.2:3b

# install Python deps into a project-local venv
uv sync
```

SD checkpoints are downloaded on first use into `~/.cache/huggingface/`.

## Run

```bash
# Text-to-image, default model (DreamShaper 8, 512x512, 25 steps)
uv run python -m src.main "a friendly fox character holding a glowing gem"

# Reproducible
uv run python -m src.main "a treasure chest overflowing with gold coins" --seed 42

# Skip the LLM expansion step (compare quality with/without)
uv run python -m src.main "a brave knight" --no-enhance

# Different checkpoint (any HF text-to-image repo id)
uv run python -m src.main "a wizard" --model stable-diffusion-v1-5/stable-diffusion-v1-5

# Override generation parameters
uv run python -m src.main "a wizard" --steps 30 --guidance 7.5 --width 768 --height 768

# Image-to-image — pass --init-image and (optionally) --strength
uv run python -m src.main "a castle on a hill at sunset" \
  --init-image some_init.png --strength 0.7 --width 512 --height 768
```

Output goes to `outputs/{timestamp}.png` with a sibling `.json` recording
the user prompt, the expanded prompt, the model id, mode, seed, and device.

## Performance on Mac Mini M4 (16GB unified memory)

| Stage | Time (DreamShaper 8 / SD 1.5, default) |
|---|---|
| Ollama prompt expansion, cached | 1-5 s |
| Pipeline load (cold) | 5-8 s |
| Generation @ 25 steps, 512×512 | ~25-30 s |
| **End-to-end cold** | **~35-45 s** |

The first call to Ollama after boot is slower because the model has to be
loaded from disk into RAM; subsequent calls within ~5 min are sub-second.
The SD pipeline is loaded once per process and freed on exit.

## Precision strategy on MPS (the black-image trap)

Stable Diffusion in `torch.float16` on Apple's MPS backend has a well-known
failure mode: somewhere in the UNet or VAE the activations overflow to NaN,
which silently casts to 0 in the uint8 conversion — the user sees a black
PNG with only a `RuntimeWarning: invalid value encountered in cast` in the
logs. This came up firsthand with base SD 1.5 fp16; the workaround:

| Backend | UNet dtype | VAE | Why |
|---|---|---|---|
| MPS + SD 1.5 | fp32 | fp32 | Mixed fp16-UNet/fp32-VAE doesn't help — UNet itself produces NaN latents. fp32 is the only reliable option and is barely slower (~1.3 it/s either way on M4). |
| MPS + SDXL | fp16 | `madebyollin/sdxl-vae-fp16-fix` | SDXL fp32 (~14GB) doesn't fit in 16GB unified memory. The community fp16-fix VAE has rescaled weights that stay in-range during decode, so fp16 *should* work end-to-end — though I've still seen NaN issues on MPS that don't reproduce on CUDA. Open issue. |
| CUDA | fp16 | default | fp16 is well-supported on CUDA across both SD versions. |
| CPU | fp32 | fp32 | fp16 isn't usefully supported. |

`src/image_gen.py:_pipeline_kwargs` branches on this automatically — the
caller just passes the HF model id.

## Memory & isolation

- All Python deps live in `.venv/` (uv-managed) — system Python untouched.
- Ollama runs as a background service (`brew services start ollama`) and
  unloads idle models from memory after 5 min. Set `OLLAMA_KEEP_ALIVE=0`
  to unload after each call; `brew services stop ollama` to stop entirely.
- The SD pipeline is loaded only while the CLI runs and is freed when the
  Python process exits — this is a one-shot CLI, not a server.

## Notes / Caveats

- `runwayml/stable-diffusion-v1-5` was removed from HF in 2024; the
  community mirror `stable-diffusion-v1-5/stable-diffusion-v1-5` serves
  the same weights and is what the code defaults will reach for.
- The LLM is just an Ollama HTTP call — swap `llama3.2:3b` → `phi3:mini`
  → `qwen2.5` with `--llm-model`.
- The SDXL-on-MPS row in the precision table is the most fragile part of
  the pipeline; on a 16GB M-series machine the SD 1.5 path is the
  reliable default.
