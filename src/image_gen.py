"""Image generation via diffusers, supporting both SD 1.5 and SDXL pipelines.

`AutoPipelineForText2Image` auto-detects the right pipeline class from each
checkpoint's `model_index.json`, so SD 1.5 finetunes (e.g. DreamShaper 8) and
SDXL finetunes (e.g. DreamShaper XL Turbo) work through the same entry point.

On Apple Silicon (MPS):
- SD 1.5 runs in fp32 (fp16 produces black images, see README).
- SDXL runs in fp16 + the `madebyollin/sdxl-vae-fp16-fix` VAE
  (community fix that prevents the same NaN issue at the SDXL VAE step).
"""
from __future__ import annotations

import torch
from diffusers import AutoencoderKL, AutoPipelineForImage2Image, AutoPipelineForText2Image
from PIL import Image

# SD 1.5 finetune leaning toward 3D-cartoon output. SDXL finetunes such as
# Lykon/dreamshaper-xl-v2-turbo work on CUDA but have been observed to produce
# black-image (NaN) outputs on MPS in fp16 / bf16 even with sdxl-vae-fp16-fix,
# and SDXL fp32 won't fit on a 16GB Apple-Silicon box. The SD 1.5 default below
# is the reliable choice on M-series hardware; --model lets you switch.
DEFAULT_MODEL = "Lykon/dreamshaper-8"
SDXL_VAE_FP16_FIX = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_NEGATIVE = (
    "low quality, blurry, distorted, watermark, text, signature, "
    "deformed, ugly, bad anatomy, extra limbs"
)

# Markers that identify a "Turbo / distilled" model needing few steps + low CFG.
_TURBO_MARKERS = ("turbo", "lightning", "lcm", "hyper")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_sdxl(model_id: str) -> bool:
    return "xl" in model_id.lower()


def is_turbo(model_id: str) -> bool:
    lid = model_id.lower()
    return any(m in lid for m in _TURBO_MARKERS)


def recommended_defaults(model_id: str) -> tuple[int, int, float]:
    """Return (size, steps, guidance_scale) tuned for the given checkpoint."""
    if is_turbo(model_id):
        return (1024, 6, 2.0)
    if is_sdxl(model_id):
        return (1024, 25, 7.0)
    return (512, 25, 7.5)


def _pipeline_kwargs(model_id: str) -> tuple[dict, str, torch.dtype]:
    """Shared (kwargs, device, dtype) for both text2img and img2img loaders."""
    device = pick_device()
    sdxl = is_sdxl(model_id)
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        # SD 1.5 fp16 on MPS → black images. SDXL fp16 + vae-fp16-fix works.
        dtype = torch.float16 if sdxl else torch.float32
    else:
        dtype = torch.float32

    kwargs: dict = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if sdxl and dtype == torch.float16:
        kwargs["vae"] = AutoencoderKL.from_pretrained(SDXL_VAE_FP16_FIX, torch_dtype=dtype)
    return kwargs, device, dtype


def load_pipeline(model_id: str = DEFAULT_MODEL):
    kwargs, device, _ = _pipeline_kwargs(model_id)
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe, device


def load_img2img_pipeline(model_id: str = DEFAULT_MODEL):
    kwargs, device, _ = _pipeline_kwargs(model_id)
    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe, device


def prepare_init_image(path: str, width: int, height: int) -> Image.Image:
    """Load an init image, force RGB, and resize to (width, height) with LANCZOS."""
    img = Image.open(path).convert("RGB")
    return img.resize((width, height), Image.LANCZOS)


def generate(
    pipe,
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int | None = None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


def generate_img2img(
    pipe,
    prompt: str,
    init_image: Image.Image,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE,
    strength: float = 0.7,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int | None = None,
):
    """Img2img: start from `init_image` (already at target size), inject noise
    proportional to `strength` (0=identity, 1≈text2img), denoise with `prompt`.

    The actual diffusion steps run is roughly `steps * strength` because
    diffusers reuses the same scheduler — so to keep effective work similar
    to a text2img run, pass `steps` ≈ `target_text2img_steps / strength`."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]
