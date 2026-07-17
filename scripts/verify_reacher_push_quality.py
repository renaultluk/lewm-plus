"""Verify collision and push quality for Reacher task-agnostic Lance datasets.

This script is intended as a quality gate before long data generation runs.
It evaluates only push tasks and reports whether enough episodes show:
- positive object displacement,
- reduced object-to-goal distance,
- fingertip-object contact events.
"""

import argparse

import numpy as np
import stable_worldmodel as swm


PUSH_TASKS = {"push_blue_object_to_blue_spot", "push_purple_ball_to_edge"}


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _task_name_from_episode(ep):
    if "task_name" in ep:
        raw = _to_numpy(ep["task_name"]).reshape(-1)
        if raw.size > 0:
            first = raw[0]
            if isinstance(first, (bytes, bytearray)):
                return first.decode()
            return str(first)
    return "unknown"


def _episode_metrics(ep):
    dist = _to_numpy(ep["object_to_goal_dist"]).reshape(-1).astype(np.float32)
    tip = _to_numpy(ep["tip_to_object_dist"]).reshape(-1).astype(np.float32)
    contact = _to_numpy(ep["push_contact"]).reshape(-1).astype(bool)

    valid = np.isfinite(dist)
    if not valid.any():
        return None

    d = dist[valid]
    t = tip[valid] if tip.size == dist.size else np.full_like(d, np.nan)
    c = contact[valid] if contact.size == dist.size else np.zeros(len(d), dtype=bool)

    return {
        "start_dist": float(d[0]),
        "end_dist": float(d[-1]),
        "min_dist": float(np.min(d)),
        "improvement": float(d[0] - d[-1]),
        "contact_steps": int(np.sum(c)),
        "min_tip_to_object": float(np.nanmin(t)) if np.isfinite(t).any() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to .lance dataset")
    parser.add_argument("--min_push_episodes", type=int, default=4)
    parser.add_argument("--min_improved_frac", type=float, default=0.7)
    parser.add_argument("--min_contact_frac", type=float, default=0.7)
    parser.add_argument("--min_close_frac", type=float, default=0.0)
    parser.add_argument("--close_threshold", type=float, default=0.05)
    args = parser.parse_args()

    ds = swm.data.get_format("lance").open_reader(args.dataset)
    num_eps = len(ds.lengths)

    rows = []
    for ep_idx in range(num_eps):
        ep = ds.load_episode(ep_idx)
        task_name = _task_name_from_episode(ep)
        if task_name not in PUSH_TASKS:
            continue
        m = _episode_metrics(ep)
        if m is None:
            continue
        rows.append((ep_idx, task_name, m))

    push_n = len(rows)
    if push_n == 0:
        print("No push episodes found (requires task_name/object_to_goal_dist/push_contact columns).")
        raise SystemExit(2)

    improved = np.array([r[2]["improvement"] > 0.0 for r in rows], dtype=bool)
    had_contact = np.array([r[2]["contact_steps"] > 0 for r in rows], dtype=bool)
    reached_close = np.array(
        [r[2]["min_dist"] <= args.close_threshold for r in rows], dtype=bool
    )

    improved_frac = float(improved.mean())
    contact_frac = float(had_contact.mean())
    close_frac = float(reached_close.mean())

    print(f"dataset: {args.dataset}")
    print(f"episodes_total: {num_eps}")
    print(f"push_episodes: {push_n}")
    print(f"improved_frac: {improved_frac:.3f}")
    print(f"contact_frac: {contact_frac:.3f}")
    print(f"close_frac@{args.close_threshold:.3f}: {close_frac:.3f}")

    print("sample_push_episodes:")
    for ep_idx, task_name, m in rows[:8]:
        print(
            f"  ep={ep_idx} task={task_name} start={m['start_dist']:.3f} end={m['end_dist']:.3f} "
            f"min={m['min_dist']:.3f} improve={m['improvement']:.3f} contacts={m['contact_steps']}"
        )

    checks = [
        (push_n >= args.min_push_episodes, f"push_episodes < {args.min_push_episodes}"),
        (improved_frac >= args.min_improved_frac, f"improved_frac < {args.min_improved_frac}"),
        (contact_frac >= args.min_contact_frac, f"contact_frac < {args.min_contact_frac}"),
    ]
    if args.min_close_frac > 0.0:
        checks.append((close_frac >= args.min_close_frac, f"close_frac < {args.min_close_frac}"))
    failures = [msg for ok, msg in checks if not ok]

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
