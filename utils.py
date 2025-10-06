import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def save_obs_image(obs, out_dir: str, step_idx: int, action_token: str, scale_factor: int = 8,
                   model_name: str = None, replay_buffer_size: int = None) -> None:
    """Save post-step observation image, optionally upscaled for visibility, with text overlay."""
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(obs, dict) and "image" in obs:
        arr = obs["image"]
    else:
        arr = obs

    img = Image.fromarray(np.asarray(arr).astype(np.uint8))

    if scale_factor > 1:
        new_size = (img.width * scale_factor, img.height * scale_factor)
        img = img.resize(new_size, resample=Image.NEAREST)  # keep pixelated aesthetic

    # Add text overlay with model name, step number, and replay buffer size
    if model_name is not None or replay_buffer_size is not None:
        draw = ImageDraw.Draw(img)

        # Try to use a default font, fall back to PIL's default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)  # macOS
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)  # Linux
                except (OSError, IOError):
                    font = ImageFont.load_default()

        # Prepare text lines
        text_lines = []
        if model_name:
            text_lines.append(f"Model: {model_name}")
        text_lines.append(f"Step: {step_idx}")
        if replay_buffer_size is not None:
            text_lines.append(f"Buffer: {replay_buffer_size}")

        # Calculate text positioning
        y_offset = 5
        for line in text_lines:
            # Get text size
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text at top-left with some padding
            x = 5
            y = y_offset

            # Draw text with black outline for better visibility
            outline_width = 1
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))

            # Draw main text in white
            draw.text((x, y), line, font=font, fill=(255, 255, 255))

            y_offset += text_height + 2

    fname = f"step_{step_idx:03d}_action_taken_{action_token}.png"
    img.save(os.path.join(out_dir, fname))

# ---- Utilities ----
def obs_image_to_png_bytes(obs) -> bytes:
    """
    Works for:
      - RGBImgObsWrapper: obs is dict with key 'image' shape(H,W,3)
      - render_mode='rgb_array': env.render() returns ndarray
    """
    if isinstance(obs, dict) and "image" in obs:
        arr = obs["image"]
    else:
        arr = obs  # assume ndarray(H,W,3)
    img = Image.fromarray(np.asarray(arr).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def slugify(s: str) -> str:
    # Keep alnum, dash, underscore; replace everything else with '-'
    return ''.join(ch if ch.isalnum() or ch in ('-','_') else '-' for ch in s)