#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show that 3D Gaussian vectors induce uniform directions on the sphere.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples", type=int, default=20_000, help="Number of Gaussian vectors to sample.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gaussian_directions_3d.png"),
        help="Path to save the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    vectors = rng.normal(size=(args.num_samples, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = vectors / norms

    x, y, z = directions.T
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    fig = plt.figure(figsize=(15, 4.5))
    ax_scatter = fig.add_subplot(1, 3, 1, projection="3d")
    ax_z = fig.add_subplot(1, 3, 2)
    ax_phi = fig.add_subplot(1, 3, 3)

    sample_idx = np.arange(min(5000, args.num_samples))
    ax_scatter.scatter(x[sample_idx], y[sample_idx], z[sample_idx], s=2, alpha=0.35)
    ax_scatter.set_title("Normalized Gaussian samples")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.set_zlabel("z")
    ax_scatter.set_box_aspect((1, 1, 1))

    bins_z = np.linspace(-1.0, 1.0, 31)
    ax_z.hist(z, bins=bins_z, density=True, alpha=0.75, edgecolor="black")
    ax_z.axhline(0.5, color="crimson", linestyle="--", linewidth=2, label="Uniform[-1, 1] density")
    ax_z.set_title("z-coordinate of directions")
    ax_z.set_xlabel("z")
    ax_z.set_ylabel("density")
    ax_z.legend()

    bins_phi = np.linspace(0.0, 2 * np.pi, 31)
    ax_phi.hist(phi, bins=bins_phi, density=True, alpha=0.75, edgecolor="black")
    ax_phi.axhline(1 / (2 * np.pi), color="crimson", linestyle="--", linewidth=2, label="Uniform[0, 2π) density")
    ax_phi.set_title("Azimuth of directions")
    ax_phi.set_xlabel("phi")
    ax_phi.set_ylabel("density")
    ax_phi.legend()

    fig.suptitle("A 3D isotropic Gaussian has uniform direction after normalization", fontsize=14)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved figure to {args.output}")


if __name__ == "__main__":
    main()
