"""
Generate a CNN pipeline diagram (Input → Extract features → CNN → Classify)
from three MRNet-style .npy volumes: axial, sagittal, coronal.

Usage:
    python scripts/generate_cnn_pipeline_diagram.py \
        --axial path/to/axial.npy \
        --sagittal path/to/sagittal.npy \
        --coronal path/to/coronal.npy \
        --output output/knee_mri_pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


MEAN = 58.09
STDDEV = 49.73
CROP_SIZE = 224


def load_volume(npy_path: Path) -> np.ndarray:
    volume = np.load(str(npy_path))
    if volume.ndim != 3:
        raise ValueError(f"{npy_path} must be 3D [slices, height, width], got shape {volume.shape}")
    return volume.astype(np.float32)


def normalize_slice_for_display(slice_2d: np.ndarray) -> np.ndarray:
    """Min-max normalize a slice to [0, 1] for visualization."""
    slice_min = float(slice_2d.min())
    slice_max = float(slice_2d.max())
    if slice_max - slice_min < 1e-6:
        return np.zeros_like(slice_2d, dtype=np.float32)
    return (slice_2d - slice_min) / (slice_max - slice_min)


def center_crop_slice(slice_2d: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    height, width = slice_2d.shape
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    cropped = slice_2d[top : top + crop_size, left : left + crop_size]
    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        from PIL import Image

        resized = Image.fromarray(cropped).resize((crop_size, crop_size), Image.BILINEAR)
        cropped = np.asarray(resized, dtype=np.float32)
    return cropped


def pick_representative_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def draw_stage_arrow(ax, x_start: float, x_end: float, y: float = 0.5) -> None:
    arrow = FancyArrowPatch(
        (x_start, y),
        (x_end, y),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.5,
        color="#2563eb",
        clip_on=False,
    )
    ax.add_patch(arrow)


def draw_cnn_schematic(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box = FancyBboxPatch(
        (0.3, 0.8),
        9.4,
        8.4,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        linewidth=1.5,
        edgecolor="#94a3b8",
        facecolor="#f8fafc",
        linestyle="--",
    )
    ax.add_patch(box)

    branch_labels = ["Axial", "Sagittal", "Coronal"]
    branch_y_positions = [7.2, 5.0, 2.8]
    branch_colors = ["#0ea5e9", "#8b5cf6", "#f97316"]

    for label, y_center, color in zip(branch_labels, branch_y_positions, branch_colors):
        x = 0.8
        conv_sizes = [(1.6, 1.6), (1.2, 1.2), (0.9, 0.9)]
        for index, (width, height) in enumerate(conv_sizes):
            rect = Rectangle(
                (x + index * 1.5, y_center - height / 2),
                width,
                height,
                linewidth=1.2,
                edgecolor=color,
                facecolor=color,
                alpha=0.25 + index * 0.15,
            )
            ax.add_patch(rect)
            if index < len(conv_sizes) - 1:
                ax.annotate(
                    "",
                    xy=(x + (index + 1) * 1.5 - 0.05, y_center),
                    xytext=(x + index * 1.5 + width + 0.05, y_center),
                    arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2),
                )

        pool_x = 5.6
        pool_rect = Rectangle(
            (pool_x, y_center - 0.55),
            0.9,
            1.1,
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.35,
        )
        ax.add_patch(pool_rect)
        ax.text(pool_x + 0.45, y_center - 1.1, "Max\nPool", ha="center", va="top", fontsize=7, color="#475569")

        ax.annotate(
            "",
            xy=(pool_x - 0.05, y_center),
            xytext=(4.7, y_center),
            arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2),
        )

        embed_x = 7.0
        ax.add_patch(
            FancyBboxPatch(
                (embed_x, y_center - 0.45),
                1.0,
                0.9,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=1.2,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(embed_x + 0.5, y_center, "256-d", ha="center", va="center", fontsize=7)

        ax.annotate(
            "",
            xy=(embed_x - 0.05, y_center),
            xytext=(pool_x + 0.95, y_center),
            arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2),
        )

        ax.text(0.5, y_center + 1.0, label, fontsize=8, color=color, fontweight="bold")

    concat_x = 8.5
    ax.add_patch(
        FancyBboxPatch(
            (concat_x, 3.6),
            1.1,
            2.8,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.5,
            edgecolor="#334155",
            facecolor="#e2e8f0",
        )
    )
    ax.text(concat_x + 0.55, 5.0, "Concat\n768-d", ha="center", va="center", fontsize=8, fontweight="bold")

    for y_center in branch_y_positions:
        ax.annotate(
            "",
            xy=(concat_x - 0.05, 5.0),
            xytext=(8.0, y_center),
            arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.0, connectionstyle="arc3,rad=0.08"),
        )

    ax.add_patch(
        FancyBboxPatch(
            (concat_x + 1.4, 4.55),
            0.8,
            0.9,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor="#cbd5e1",
        )
    )
    ax.text(concat_x + 1.8, 5.0, "FC", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.annotate(
        "",
        xy=(concat_x + 1.35, 5.0),
        xytext=(concat_x + 1.1, 5.0),
        arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2),
    )

    ax.text(5.0, 9.3, "TripleMRNet (AlexNet × 3 branches)", ha="center", va="center", fontsize=10, fontweight="bold")


def generate_diagram(
    axial_path: Path,
    sagittal_path: Path,
    coronal_path: Path,
    output_path: Path,
    probabilities: dict[str, float] | None = None,
) -> Path:
    volumes = {
        "axial": load_volume(axial_path),
        "sagittal": load_volume(sagittal_path),
        "coronal": load_volume(coronal_path),
    }

    if probabilities is None:
        probabilities = {"abnormal": 0.78, "acl": 0.05, "meniscus": 0.34}

    fig = plt.figure(figsize=(16, 5), facecolor="white")
    stage_titles = ["Input image", "Extract features", "CNN Computing", "Classify"]
    axes = []

    for index in range(4):
        ax = fig.add_axes([0.02 + index * 0.245, 0.18, 0.21, 0.68])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes.append(ax)

    # Stage 1: Input — sagittal middle slice as primary display
    sagittal_slice = pick_representative_slice(volumes["sagittal"])
    axes[0].imshow(normalize_slice_for_display(sagittal_slice), cmap="gray", aspect="equal")
    axes[0].set_title("Sagittal MRI", fontsize=9, pad=6)
    axes[0].text(
        0.5,
        -0.08,
        f"Shape: {volumes['sagittal'].shape[0]} slices",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#64748b",
    )

    # Stage 2: Extract features — three views with crop boxes
    view_configs = [
        ("axial", volumes["axial"], "#0ea5e9"),
        ("sagittal", volumes["sagittal"], "#8b5cf6"),
        ("coronal", volumes["coronal"], "#f97316"),
    ]
    thumbnail_axes_positions = [(0.08, 0.62), (0.08, 0.36), (0.08, 0.10)]
    crop_axes_positions = [(0.58, 0.62), (0.58, 0.36), (0.58, 0.10)]

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")

    for (view_name, volume, color), thumb_pos, crop_pos in zip(
        view_configs, thumbnail_axes_positions, crop_axes_positions
    ):
        slice_2d = pick_representative_slice(volume)
        display_slice = normalize_slice_for_display(slice_2d)
        cropped = center_crop_slice(slice_2d)
        cropped_display = normalize_slice_for_display(cropped)

        inset_thumb = axes[1].inset_axes([thumb_pos[0], thumb_pos[1], 0.34, 0.22])
        inset_thumb.imshow(display_slice, cmap="gray")
        inset_thumb.set_xticks([])
        inset_thumb.set_yticks([])
        inset_thumb.set_title(view_name.capitalize(), fontsize=7, color=color, pad=2)

        height, width = display_slice.shape
        crop_top = max(0, (height - CROP_SIZE) // 2)
        crop_left = max(0, (width - CROP_SIZE) // 2)
        rect = Rectangle(
            (crop_left, crop_top),
            min(CROP_SIZE, width),
            min(CROP_SIZE, height),
            linewidth=1.5,
            edgecolor="#ec4899",
            facecolor="none",
        )
        inset_thumb.add_patch(rect)

        inset_crop = axes[1].inset_axes([crop_pos[0], crop_pos[1], 0.28, 0.18])
        inset_crop.imshow(cropped_display, cmap="gray")
        inset_crop.set_xticks([])
        inset_crop.set_yticks([])
        for spine in inset_crop.spines.values():
            spine.set_edgecolor("#ec4899")
            spine.set_linewidth(1.5)

        axes[1].annotate(
            "",
            xy=(crop_pos[0] - 0.02, crop_pos[1] + 0.09),
            xytext=(thumb_pos[0] + 0.34 + 0.02, thumb_pos[1] + 0.11),
            xycoords=axes[1].transAxes,
            textcoords=axes[1].transAxes,
            arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.0, linestyle="dashed"),
        )

    axes[1].text(0.5, 0.98, "Center crop 224×224 per slice", ha="center", va="top", fontsize=8, color="#64748b")

    # Stage 3: CNN schematic
    draw_cnn_schematic(axes[2])

    # Stage 4: Classify
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].axis("off")

    class_items = [
        ("Abnormal?", probabilities["abnormal"], "#dc2626"),
        ("ACL tear?", probabilities["acl"], "#2563eb"),
        ("Meniscus tear?", probabilities["meniscus"], "#16a34a"),
    ]
    y_positions = [0.72, 0.45, 0.18]

    for (label, prob, color), y_pos in zip(class_items, y_positions):
        box = FancyBboxPatch(
            (0.12, y_pos - 0.08),
            0.76,
            0.16,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#f8fafc",
        )
        axes[3].add_patch(box)
        axes[3].text(0.18, y_pos + 0.02, label, fontsize=11, va="center", color="#1e293b")
        axes[3].text(0.82, y_pos + 0.02, f"{prob:.1%}", fontsize=11, va="center", ha="right", color=color, fontweight="bold")

        axes[3].annotate(
            "",
            xy=(0.08, y_pos + 0.02),
            xytext=(-0.12, y_pos + 0.02),
            xycoords=axes[3].transAxes,
            textcoords=axes[3].transAxes,
            arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2, linestyle="dashed"),
        )

    axes[3].text(0.5, 0.95, "Sigmoid → probability", ha="center", va="top", fontsize=8, color="#64748b")

    # Stage labels at bottom
    for index, (ax, title) in enumerate(zip(axes, stage_titles)):
        ax.text(0.5, -0.14, title, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")

    # Inter-stage arrows
    arrow_ax = fig.add_axes([0, 0, 1, 1])
    arrow_ax.axis("off")
    arrow_x_pairs = [(0.23, 0.475), (0.475, 0.72), (0.72, 0.95)]
    for x_start, x_end in arrow_x_pairs:
        draw_stage_arrow(arrow_ax, x_start, x_end, y=0.52)

    fig.suptitle(
        "Knee MRI Diagnosis Pipeline: Three .npy Volumes → TripleMRNet → Three-Task Classification",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_demo_npy_files(output_dir: Path) -> tuple[Path, Path, Path]:
    """Create three synthetic MRNet-style demo volumes when real data is unavailable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def make_knee_like_volume(num_slices: int, height: int, width: int, orientation: str) -> np.ndarray:
        volume = np.zeros((num_slices, height, width), dtype=np.float32)
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        for slice_index in range(num_slices):
            center_shift = int(8 * np.sin(slice_index / max(num_slices - 1, 1) * np.pi))
            bone_mask = ((x_grid - width // 2) ** 2 + (y_grid - height // 2 + center_shift) ** 2) < (55 + slice_index % 5) ** 2
            tissue_mask = (
                ((x_grid - width // 2) ** 2 + (y_grid - height // 2 + center_shift) ** 2) < (85 + slice_index % 7) ** 2
            ) & ~bone_mask
            volume[slice_index, bone_mask] = 180 + rng.normal(0, 8, size=bone_mask.sum())
            volume[slice_index, tissue_mask] = 90 + rng.normal(0, 12, size=tissue_mask.sum())
            if orientation == "sagittal" and 0.35 * num_slices < slice_index < 0.65 * num_slices:
                lesion_y = height // 2 + center_shift - 15
                lesion_x = width // 2 + 20
                lesion = ((x_grid - lesion_x) ** 2 + (y_grid - lesion_y) ** 2) < 12 ** 2
                volume[slice_index, lesion] = 140 + rng.normal(0, 10, size=lesion.sum())
            volume[slice_index] += rng.normal(0, 5, size=(height, width))
        volume = np.clip(volume, 0, 255)
        return volume

    paths = {}
    configs = [
        ("axial", 28, 256, 256),
        ("sagittal", 32, 256, 256),
        ("coronal", 26, 256, 256),
    ]
    for name, slices, height, width in configs:
        volume = make_knee_like_volume(slices, height, width, name)
        file_path = output_dir / f"{name}.npy"
        np.save(str(file_path), volume)
        paths[name] = file_path

    return paths["axial"], paths["sagittal"], paths["coronal"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate knee MRI CNN pipeline diagram from three .npy files.")
    parser.add_argument("--axial", type=Path, help="Path to axial .npy volume")
    parser.add_argument("--sagittal", type=Path, help="Path to sagittal .npy volume")
    parser.add_argument("--coronal", type=Path, help="Path to coronal .npy volume")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/knee_mri_pipeline_diagram.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use auto-generated demo .npy files when paths are not provided",
    )
    parser.add_argument("--prob-abnormal", type=float, default=0.78)
    parser.add_argument("--prob-acl", type=float, default=0.051)
    parser.add_argument("--prob-meniscus", type=float, default=0.338)
    args = parser.parse_args()

    if args.axial and args.sagittal and args.coronal:
        axial_path, sagittal_path, coronal_path = args.axial, args.sagittal, args.coronal
    elif args.demo:
        demo_dir = Path("demo_npy")
        axial_path, sagittal_path, coronal_path = create_demo_npy_files(demo_dir)
        print(f"Demo .npy files created in: {demo_dir.resolve()}")
    else:
        parser.error("Provide --axial/--sagittal/--coronal paths, or use --demo to generate sample data.")

    output_path = generate_diagram(
        axial_path=axial_path,
        sagittal_path=sagittal_path,
        coronal_path=coronal_path,
        output_path=args.output,
        probabilities={
            "abnormal": args.prob_abnormal,
            "acl": args.prob_acl,
            "meniscus": args.prob_meniscus,
        },
    )
    print(f"Diagram saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
