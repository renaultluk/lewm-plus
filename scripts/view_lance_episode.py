"""Inspect and export episodes from a Lance dataset.

Examples
--------
# Print dataset metadata and episode info
python scripts/view_lance_episode.py --dataset ~/.stable-wm/datasets/reacher_multitask_v5_smoke.lance

# Export episode 0 to MP4
python scripts/view_lance_episode.py --dataset ~/.stable-wm/datasets/reacher_multitask_v5_smoke.lance --episode 0 --output /tmp/reacher_ep0.mp4

# Export episode 2 to GIF at 12 FPS
python scripts/view_lance_episode.py --dataset ~/.stable-wm/datasets/reacher_multitask_v5_smoke.lance --episode 2 --fps 12 --output /tmp/reacher_ep2.gif
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import stable_worldmodel as swm


REACHER_TASKS = [
    "reach_red_spot",
    "push_blue_object_to_blue_spot",
    "push_purple_ball_to_edge",
    "fold_in_on_itself",
    "trace_circle",
]


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _convert_frames_to_hwc_uint8(frames):
    arr = _to_numpy(frames)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got shape {arr.shape}")

    if arr.shape[-1] == 3:
        hwc = arr
    elif arr.shape[1] == 3:
        hwc = arr.transpose(0, 2, 3, 1)
    else:
        raise ValueError(f"Cannot infer image channel dimension from shape {arr.shape}")

    if hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    return hwc


def _task_name(task_id):
    if 0 <= task_id < len(REACHER_TASKS):
        return REACHER_TASKS[task_id]
    return f"unknown({task_id})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to a .lance dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to inspect/export")
    parser.add_argument("--image_key", default="pixels", help="Image column to export")
    parser.add_argument("--fps", type=int, default=20, help="FPS for video/gif export")
    parser.add_argument("--output", default=None, help="Output video path (.mp4 or .gif)")
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print metadata, do not export video",
    )
    args = parser.parse_args()

    ds = swm.data.get_format("lance").open_reader(args.dataset)

    num_episodes = len(ds.lengths)
    if num_episodes == 0:
        raise ValueError(f"Dataset is empty: {args.dataset}")
    if args.episode < 0 or args.episode >= num_episodes:
        raise ValueError(f"Invalid --episode {args.episode}; dataset has {num_episodes} episodes")

    print(f"dataset: {args.dataset}")
    print(f"rows: {len(ds)}")
    print(f"episodes: {num_episodes}")
    print(f"columns: {list(ds.column_names)}")
    print(f"episode_lengths: {ds.lengths.tolist()}")

    ep = ds.load_episode(args.episode)
    print(f"episode: {args.episode}")
    print(f"episode_keys: {list(ep.keys())}")

    if "task_id" in ep:
        task_vals = _to_numpy(ep["task_id"]).reshape(-1)
        unique_ids = sorted(set(int(v) for v in task_vals.tolist()))
        task_names = [_task_name(v) for v in unique_ids]
        print(f"task_ids: {unique_ids}")
        print(f"task_names: {task_names}")

    if args.print_only:
        return

    if args.image_key not in ep:
        raise ValueError(f"Image key '{args.image_key}' not found. Available keys: {list(ep.keys())}")

    frames = _convert_frames_to_hwc_uint8(ep[args.image_key])

    output = args.output
    if output is None:
        output = f"episode_{args.episode}.mp4"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext not in {".mp4", ".gif"}:
        raise ValueError("--output must end with .mp4 or .gif")

    imageio.mimsave(str(output_path), frames, fps=args.fps)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
