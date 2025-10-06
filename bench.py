# mvp_minigrid_vla.py
import os
import io
import json
import time
import argparse
import utils
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from datetime import datetime
import google.generativeai as genai

import numpy as np
from PIL import Image

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ---- Action mapping ----
# Canonical tokens we’ll ask the LLM to output:
CANONICAL_ACTIONS = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

# Map canonical → MiniGrid Actions enum index (filled per-env after reset)
# We'll detect available actions from env.actions if present; else fall back to standard MiniGrid.
def build_action_maps(env) -> Tuple[Dict[str, int], Dict[str, str]]:
    # MiniGrid usually exposes env.actions.{left,right,forward,pickup,drop,toggle,done}
    name_to_id = {}
    synonyms = {
        "turn left": "left", "rotate left": "left",
        "turn right": "right", "rotate right": "right",
        "move forward": "forward", "step forward": "forward",
        "pick up": "pickup", "pick": "pickup", "grab": "pickup",
        "put down": "drop", "place": "drop",
        "use": "toggle", "interact": "toggle",
        "finish": "done", "stop": "done", "terminate": "done"
    }

    # Build canonical mapping if possible
    if hasattr(env, "actions"):
        # env.actions is an Enum; getattr(env.actions, name).value is int
        for token in CANONICAL_ACTIONS:
            if hasattr(env.actions, token):
                name_to_id[token] = getattr(env.actions, token).value
    else:
        # Fallback to default indexing used widely in MiniGrid:
        # 0=left,1=right,2=forward,3=pickup,4=drop,5=toggle,6=done
        name_to_id = {k: i for i, k in enumerate(CANONICAL_ACTIONS)}

    return name_to_id, synonyms

def canonicalize_action(s: str, synonyms: Dict[str, str]) -> Optional[str]:
    t = s.strip().lower()
    if t in CANONICAL_ACTIONS:
        return t
    if t in synonyms:
        return synonyms[t]
    # try light normalization
    t = t.replace("-", " ").replace("_", " ").strip()
    if t in CANONICAL_ACTIONS:
        return t
    if t in synonyms:
        return synonyms[t]
    return None

# ---- Replay buffer types ----
@dataclass
class Transition:
    # Full SARSA transition: S_{i-1}, A_{i-1}, R_{i-1}
    prev_state_image: Optional[bytes]  # PNG bytes of S_{i-1}
    prev_action: Optional[str]  # canonical string of A_{i-1}
    prev_reward: Optional[float]  # R_{i-1}

# ---- Prompt builder ----
PROMPT_SYSTEM = (
    "You are a helpful robot policy. Your job is to pick the NEXT action for a MiniGrid agent.\n"
    "You will be shown: environment info, SARSA history with previous state images, and the CURRENT state image.\n"
    "The CURRENT state image is always the LAST image shown. You must choose an action for this CURRENT state.\n"
    "Previous state images are shown for context to understand the agent's trajectory.\n"
    "You MUST output strictly valid minified JSON with this schema:\n"
    '{"action":"<one-of: left|right|forward|pickup|drop|toggle|done>","reasoning":"<brief>"}\n'
    "Rules:\n"
    "- Choose exactly one action from the allowed list for the CURRENT (final) state image.\n"
    "- Keep reasoning under 20 words. Do NOT include extra keys or text.\n"
    "- If uncertain, prefer safe exploration (often 'left'/'right' to orient or 'forward' if clear).\n"
    "- The goal is to get to the green goal square. You can enter this square."
)

def build_user_prompt(env_id: str,
                      mission: Optional[str],
                      step_idx: int,
                      reward_so_far: float,
                      allowed_actions: List[str],
                      replay: List[Transition],
                      max_replay: int) -> str:
    lines = []
    lines.append(f"ENVIRONMENT: {env_id}")
    if mission:
        lines.append(f"MISSION: {mission}")
    lines.append(f"STEP: {step_idx}")
    lines.append(f"RETURN_SO_FAR: {reward_so_far:.3f}")
    lines.append("ALLOWED_ACTIONS: " + ", ".join(allowed_actions))
    lines.append(f"REPLAY_BUFFER_SIZE: {min(len(replay), max_replay)}")

    # Show SARSA history with image references
    recent_replay = list(reversed(replay[-max_replay:]))
    for i, tr in enumerate(recent_replay, 1):
        if tr.prev_state_image is not None:
            lines.append(f"REPLAY[-{i}]: prev_state=<image_{i}>, prev_action={tr.prev_action}, prev_reward={tr.prev_reward}")
        else:
            lines.append(f"REPLAY[-{i}]: prev_action={tr.prev_action}, prev_reward={tr.prev_reward}")

    lines.append("CURRENT_STATE: <current_image> (CHOOSE ACTION FOR THIS STATE)")
    lines.append("RESPOND WITH JSON ONLY.")
    return "\n".join(lines)

# ---- LLM client interfaces ----
class LLMClient:
    def propose_action(self, images: List[bytes], prompt_system: str, prompt_user: str) -> str:
        """
        Propose an action given multiple images and prompts.

        Args:
            images: List of PNG image bytes. Last image is the current state,
                   previous images are historical states from replay buffer.
            prompt_system: System prompt
            prompt_user: User prompt with image references
        """
        raise NotImplementedError

class OpenAIClient(LLMClient):
    """
    Minimal adapter. Requires:
      - env OPENAI_API_KEY
      - model like 'gpt-4o' or 'o4-mini'
    """
    def __init__(self, model: str = "gpt-4o", top_p: float = 1.0):
        from openai import OpenAI  # type: ignore
        self.client = OpenAI()
        self.model = model
        self.top_p = top_p

    def propose_action(self, images: List[bytes], prompt_system: str, prompt_user: str) -> str:
        # Build content with text and multiple images
        import base64
        content = [{"type": "text", "text": prompt_user}]

        # Add previous state images (replay history)
        for i, img_bytes in enumerate(images[:-1], 1):
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        # Add current state image (last one)
        if images:
            current_b64 = base64.b64encode(images[-1]).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{current_b64}"}
            })

        resp = self.client.chat.completions.create(
            model=self.model,
            top_p=self.top_p,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": content},
            ]
        )
        return resp.choices[0].message.content.strip()

class GeminiClient(LLMClient):
    """
    Minimal adapter. Requires:
      - env GEMINI_API_KEY
      - model like 'gemini-1.5-pro' / 'gemini-2.5-pro'
    """
    def __init__(self, model: str = "gemini-2.5-pro", top_p: float = 1.0):
        self.model = genai.GenerativeModel(model)
        self.generation_config = {"top_p": top_p}

    def propose_action(self, images: List[bytes], prompt_system: str, prompt_user: str) -> str:
        # Gemini accepts a list: [text, image1, image2, ...]
        # We fold system+user into a single text prompt.
        text = prompt_system + "\n\n" + prompt_user
        content = [text]

        # Add all images to content (Gemini handles multiple images well)
        for img_bytes in images:
            content.append({
                "mime_type": "image/png",
                "data": img_bytes,
            })

        resp = self.model.generate_content(
            content,
            generation_config=self.generation_config
        )
        return resp.text.strip()

def parse_llm_json(s: str) -> Optional[dict]:
    # Be forgiving: find the first JSON object in the string.
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # crude fallback: try to extract {...}
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
    return None

# ---- Runner ----
def run_episode(env_id: str,
                client: LLMClient,
                replay_len: int = 1,
                seed: int = 0,
                max_steps: Optional[int] = None,
                verbose: bool = True,
                out_dir: Optional[str] = None,
                model_name: Optional[str] = None) -> Dict:
    env = gym.make(env_id, render_mode="rgb_array", tile_size=8)

    obs, info = env.reset()
    mission = obs.get("mission")

    print(f"mission: {mission}")

    env = RGBImgObsWrapper(env)     # add 'image' to obs
    env = ImgObsWrapper(env)        # make obs = image only (np array)

    if seed is not None:
        env.reset(seed=seed)

    name_to_id, synonyms = build_action_maps(env)
    allowed = [a for a in CANONICAL_ACTIONS if a in name_to_id]

    replay: deque[Transition] = deque([], maxlen=replay_len)
    ep_return = 0.0
    step_idx = 0
    done = False
    truncated = False

    while not (done or truncated):
        if max_steps is not None and step_idx >= max_steps:
            break

        current_img_bytes = utils.obs_image_to_png_bytes(obs)
        user_prompt = build_user_prompt(
            env_id=env_id,
            mission=mission,
            step_idx=step_idx,
            reward_so_far=ep_return,
            allowed_actions=allowed,
            replay=list(replay),
            max_replay=replay_len
        )

        # Collect images for LLM: previous states from replay + current state
        images = []
        # Add previous state images from replay buffer (in chronological order)
        recent_replay = list(replay)[-replay_len:]
        for tr in recent_replay:
            if tr.prev_state_image is not None:
                images.append(tr.prev_state_image)
        # Add current state image (this is what we need to choose action for)
        images.append(current_img_bytes)

        # Call LLM with retry on formatting errors
        retries = 3
        action_token = None
        last_error = None
        for _ in range(retries):
            raw = client.propose_action(images, PROMPT_SYSTEM, user_prompt)
            parsed = parse_llm_json(raw)
            if not parsed or "action" not in parsed:
                last_error = f"Bad JSON from LLM: {raw[:200]}"
                # tighten reminder
                user_prompt = user_prompt + "\nREMINDER: JSON ONLY per schema."
                continue
            cand = canonicalize_action(str(parsed["action"]), synonyms)
            if cand is None or cand not in name_to_id:
                last_error = f"Invalid action from LLM: {parsed}"
                user_prompt = user_prompt + "\nREMINDER: pick one allowed action token."
                continue
            action_token = cand
            break

        if action_token is None:
            raise RuntimeError(last_error or "LLM failed to produce an action.")

        action_id = name_to_id[action_token]
        next_obs, reward, done, truncated, info = env.step(action_id)

        if verbose:
            print(f"[{step_idx:03d}] action={action_token:8s} (id={action_id})  reward={reward}  done={done}  truncated={truncated}")

        # Save the post-step state image (i.e., state after executing this action)
        if out_dir is not None:
            utils.save_obs_image(next_obs, out_dir, step_idx, action_token, scale_factor=8,
                               model_name=model_name, replay_buffer_size=len(replay))

        # update replay and state
        # Store the current state image, action taken, and reward received
        replay.append(Transition(
            prev_state_image=current_img_bytes,
            prev_action=action_token,
            prev_reward=float(reward)
        ))
        ep_return += float(reward)
        obs = next_obs
        step_idx += 1

    env.close()
    return {"return": ep_return, "steps": step_idx, "done": done, "truncated": truncated}

# ---- CLI ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="MiniGrid-Empty-5x5-v0")
    p.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai")
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--replay_len", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=128)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--experiments_dir", type=str, default="experiments",
                   help="Base directory where run images will be saved.")
    args = p.parse_args()

    # Create a unique run directory: experiments/<timestamp>_<env>_<model>_<uuid8>
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env_slug = utils.slugify(args.env_id)
    model_slug = utils.slugify(args.model)
    run_id = f"{timestamp}_{env_slug}_{model_slug}_{uuid4().hex[:8]}"
    run_dir = os.path.join(args.experiments_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    if not args.quiet:
        print(f"Saving step images to: {run_dir}")

    if args.provider == "openai":
        client = OpenAIClient(model=args.model, top_p=args.top_p)
    else:
        client = GeminiClient(model=args.model, top_p=args.top_p)

    totals = []
    for ep in range(args.episodes):
        if not args.quiet:
            print(f"\n=== Episode {ep+1}/{args.episodes} | env={args.env_id} | model={args.model} ({args.provider}) ===")
        # For multiple episodes, keep putting images in the same run_dir; filenames are step-indexed.
        stats = run_episode(
            env_id=args.env_id,
            client=client,
            replay_len=args.replay_len,
            seed=args.seed + ep,
            max_steps=args.max_steps,
            verbose=not args.quiet,
            out_dir=run_dir,
            model_name=args.model
        )
        totals.append(stats)

    ret = np.mean([t["return"] for t in totals])
    steps = np.mean([t["steps"] for t in totals])
    print(f"\nSummary over {args.episodes} ep: avg_return={ret:.3f} avg_steps={steps:.1f}")
    # Optionally write a small manifest for convenience
    manifest = {
        "env_id": args.env_id,
        "provider": args.provider,
        "model": args.model,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "replay_len": args.replay_len,
        "seed_start": args.seed,
        "run_dir": run_dir,
        "timestamp": timestamp,
        "run_uid": run_id,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
        # ---- Create GIF of trajectory ----
    try:
        import glob
        from PIL import Image

        pattern = os.path.join(run_dir, "step_*_action_taken_*.png")
        frames = []
        for filename in sorted(glob.glob(pattern)):
            img = Image.open(filename)
            frames.append(img.copy())

        if frames:
            gif_path = os.path.join(run_dir, "trajectory.gif")
            # duration = 400ms per frame, loop forever
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=400,
                loop=0,
                optimize=True,
            )
            print(f"Saved trajectory GIF: {gif_path}")
        else:
            print("No step images found; skipping GIF creation.")

    except Exception as e:
        print(f"GIF creation failed: {e}")

if __name__ == "__main__":
    main()