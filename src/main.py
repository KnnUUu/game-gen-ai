"""CLI entry: short prompt -> LLM expansion -> Stable Diffusion image."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .image_gen import (
    DEFAULT_MODEL,
    generate,
    generate_img2img,
    load_img2img_pipeline,
    load_pipeline,
    prepare_init_image,
    recommended_defaults,
)
from .llm import DEFAULT_MODEL as DEFAULT_LLM
from .llm import enhance_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stylised cartoon images from short prompts via a local LLM + Stable Diffusion.")
    p.add_argument("prompt", help="Short description, e.g. 'a friendly fox holding a magic gem'.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Stable Diffusion checkpoint, HF repo id (default {DEFAULT_MODEL}).")
    p.add_argument("--init-image", default=None,
                   help="Path to a reference image to use as img2img init. If given, runs img2img instead of text2img.")
    p.add_argument("--strength", type=float, default=0.7,
                   help="img2img strength: 0=identity, 1≈text2img. Default 0.7. Only used with --init-image.")
    p.add_argument("--steps", type=int, default=None, help="Diffusion steps. Default depends on model.")
    p.add_argument("--guidance", type=float, default=None, help="Classifier-free guidance scale. Default depends on model.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--no-enhance", action="store_true", help="Skip LLM prompt enhancement.")
    p.add_argument("--llm-model", default=DEFAULT_LLM, help=f"Ollama model name (default {DEFAULT_LLM}).")
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    default_size, default_steps, default_guidance = recommended_defaults(args.model)
    steps = args.steps if args.steps is not None else default_steps
    guidance = args.guidance if args.guidance is not None else default_guidance
    width = args.width if args.width is not None else default_size
    height = args.height if args.height is not None else default_size

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    user_prompt = args.prompt
    print(f"[input]    {user_prompt}")

    if args.no_enhance:
        sd_prompt = user_prompt
    else:
        print(f"[llm]      enhancing with {args.llm_model}...")
        t0 = time.time()
        sd_prompt = enhance_prompt(user_prompt, model=args.llm_model)
        print(f"[enhanced] ({time.time() - t0:.1f}s) {sd_prompt}")

    mode = "img2img" if args.init_image else "text2img"
    print(f"[sd]       loading {args.model} ({mode})...")
    t0 = time.time()
    if mode == "img2img":
        pipe, device = load_img2img_pipeline(args.model)
    else:
        pipe, device = load_pipeline(args.model)
    print(f"[sd]       loaded on {device} (dtype={pipe.unet.dtype}) in {time.time() - t0:.1f}s")

    t0 = time.time()
    if mode == "img2img":
        init = prepare_init_image(args.init_image, width, height)
        print(f"[sd]       init={args.init_image} resized to {width}x{height}, strength={args.strength}, {steps} steps, cfg={guidance}...")
        image = generate_img2img(
            pipe,
            sd_prompt,
            init,
            strength=args.strength,
            steps=steps,
            guidance_scale=guidance,
            seed=args.seed,
        )
    else:
        print(f"[sd]       generating {steps} steps at {width}x{height}, cfg={guidance}...")
        image = generate(
            pipe,
            sd_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            seed=args.seed,
        )
    print(f"[sd]       done in {time.time() - t0:.1f}s")

    img_path = out_dir / f"{stamp}.png"
    meta_path = out_dir / f"{stamp}.json"
    image.save(img_path)
    meta_path.write_text(
        json.dumps(
            {
                "user_prompt": user_prompt,
                "sd_prompt": sd_prompt,
                "sd_model": args.model,
                "llm_model": None if args.no_enhance else args.llm_model,
                "mode": mode,
                "init_image": args.init_image,
                "strength": args.strength if mode == "img2img" else None,
                "steps": steps,
                "guidance": guidance,
                "seed": args.seed,
                "width": width,
                "height": height,
                "device": device,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[done]     {img_path}")


if __name__ == "__main__":
    main()
