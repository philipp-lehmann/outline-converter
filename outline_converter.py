#!/usr/bin/env python3
"""
outline_converter.py
────────────────────
Convert JPG/PNG images to clean SVG outlines with globally-normalised,
standardised stroke widths.

The script runs in two passes:
  Pass 1 – measures stroke widths (in px) across ALL input images using
            a distance-transform + skeletonisation approach, then computes
            global percentile breakpoints so every image in the batch uses
            the same mapping from "thin" → "thick".
  Pass 2 – extracts contours, assigns each contour a normalised stroke
            width, and writes one SVG per input image.

Usage
─────
  # Convert every JPG/PNG in a folder (default: 2 stroke levels, 1pt / 2.5pt)
  python outline_converter.py icons/

  # Explicit stroke widths (thin → thick)
  python outline_converter.py icons/ -s 1.0 2.5          # 2 levels (default)
  python outline_converter.py icons/ -s 0.75 1.5 3.0     # 3 levels
  python outline_converter.py icons/ -s 1.5              # single level

  # Other options
  python outline_converter.py icons/ -o out/ -c "#1a1a1a" -e 1.0 -v
  python outline_converter.py dark_icons/ --invert        # light-on-dark source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# scikit-image gives the best 1-px skeleton; fall back to a pure-OpenCV
# morphological thinning if it is not installed.
try:
    from skimage.morphology import skeletonize as _ski_skel  # type: ignore

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────────────────────────────────────
# Utility: collect input images
# ─────────────────────────────────────────────────────────────────────────────


def collect_images(inputs: List[str]) -> List[Path]:
    """Return sorted, deduplicated list of image paths from files/dirs."""
    paths: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                paths.extend(p.glob(f"*{ext}"))
                paths.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(p)
        else:
            print(f"  [warn] skipping '{s}' – not a supported image or directory")
    return sorted(set(paths))


# ─────────────────────────────────────────────────────────────────────────────
# Image loading & binarisation
# ─────────────────────────────────────────────────────────────────────────────


def load_binary(
    path: Path,
    threshold: Optional[int] = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Load *path* and return a binary mask where foreground pixels = 255.

    Handles:
      • RGB / grayscale images
      • RGBA images (alpha composited onto white before binarisation)
      • Light-on-dark sources via *invert*
      • Auto-threshold (Otsu) when *threshold* is None
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Cannot read '{path}'")

    # ── alpha compositing onto white ──────────────────────────────────────────
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3:].astype(np.float32) / 255.0
        rgb = raw[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        blended = (rgb * alpha + white * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    elif raw.ndim == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw.copy()

    # ── denoise ───────────────────────────────────────────────────────────────
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ── optional invert (light-on-dark source) ────────────────────────────────
    if invert:
        gray = 255 - gray

    # ── binarise (foreground = dark ink → invert so foreground = 255) ────────
    if threshold is None:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # ── remove salt-and-pepper noise ──────────────────────────────────────────
    k = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Skeletonisation
# ─────────────────────────────────────────────────────────────────────────────


def skeletonise(binary: np.ndarray) -> np.ndarray:
    """Return a 1-px-wide skeleton of *binary* (foreground = 255)."""
    if _HAS_SKIMAGE:
        return (_ski_skel(binary > 0) * 255).astype(np.uint8)

    # ── fallback: Zhang-Suen via iterative morphological erosion ─────────────
    skel = np.zeros_like(binary)
    tmp = binary.copy()
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(tmp, cross)
        skel |= cv2.subtract(tmp, cv2.dilate(eroded, cross))
        tmp = eroded
        if not cv2.countNonZero(tmp):
            break
    return skel


# ─────────────────────────────────────────────────────────────────────────────
# Stroke-width measurement
# ─────────────────────────────────────────────────────────────────────────────


def measure_stroke_widths(binary: np.ndarray) -> np.ndarray:
    """
    Return an array of estimated stroke widths (px) for the image.

    Method: distance-transform + skeletonisation.
    At each skeleton pixel, dist_transform_value × 2 ≈ local stroke width.
    """
    if not binary.any():
        return np.array([], dtype=np.float32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skel = skeletonise(binary)
    widths = dist[skel > 0] * 2.0
    return widths[widths > 0].astype(np.float32)


def compute_breakpoints(all_widths: np.ndarray, n_buckets: int) -> np.ndarray:
    """
    Compute (n_buckets - 1) percentile breakpoints from all measured widths.
    Breakpoints are used to assign each contour to a stroke-width bucket.
    """
    if n_buckets <= 1 or all_widths.size == 0:
        return np.array([], dtype=np.float32)
    pcts = np.linspace(0, 100, n_buckets + 1)[1:-1]
    return np.percentile(all_widths, pcts).astype(np.float32)


def bucket_index(width_px: float, breakpoints: np.ndarray, n_buckets: int) -> int:
    """Map a measured width (px) to a bucket index 0…n_buckets-1."""
    return int(min(np.searchsorted(breakpoints, width_px), n_buckets - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton tracing  →  ordered polylines
# ─────────────────────────────────────────────────────────────────────────────

_DX8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def _nb8(px: int, py: int, pixel_set: set) -> List[Tuple[int,int]]:
    return [(px+dx, py+dy) for dx,dy in _DX8 if (px+dx, py+dy) in pixel_set]


def prune_skeleton_spurs(skel: np.ndarray, iterations: int) -> np.ndarray:
    """
    Remove short spur branches from the skeleton by iteratively deleting
    endpoint pixels (pixels with ≤1 eight-connected skeleton neighbour).

    After *iterations* passes any branch shorter than *iterations* pixels
    is erased.  This eliminates noise artefacts from skeletonisation without
    touching the main stroke paths.

    Side-effect: legitimate stroke ends are also shortened by ~iterations px,
    which is visually negligible for typical icon strokes (50–500 px long).
    Use --prune 0 to disable entirely.
    """
    if iterations <= 0 or not skel.any():
        return skel
    skel = skel.copy()
    for _ in range(iterations):
        ys, xs = np.where(skel > 0)
        pset: set = set(zip(xs.tolist(), ys.tolist()))
        to_remove = [p for p in pset if len(_nb8(*p, pset)) <= 1]
        if not to_remove:
            break
        for px, py in to_remove:
            skel[py, px] = 0
    return skel


def skeleton_to_polylines(skeleton: np.ndarray) -> List[np.ndarray]:
    """
    Convert a skeleton image into a list of ordered polyline arrays.

    Each returned item is an (N, 2) float32 array of (x, y) pixel coordinates
    forming one connected stroke path.

    Strategy
    ────────
    1. Classify every skeleton pixel:
         endpoint   – 0 or 1 eight-connected neighbour  (path termini)
         regular    – exactly 2 neighbours               (mid-stroke)
         branch     – 3 or more neighbours               (junctions)
    2. Remove branch pixels → leaves only simple chains as connected
       components.
    3. Trace each chain from its endpoint outward, then re-attach the
       nearest branch pixel at each end so paths meet cleanly at junctions.
    4. Handle closed loops (no endpoints) by starting anywhere.
    """
    if not skeleton.any():
        return []

    ys, xs = np.where(skeleton > 0)
    if xs.size == 0:
        return []

    pixel_set: set = set(zip(xs.tolist(), ys.tolist()))

    # ── classify pixels ────────────────────────────────────────────────────
    branch_pixels: set = set()
    for p in pixel_set:
        if len(_nb8(*p, pixel_set)) >= 3:
            branch_pixels.add(p)

    segment_pixels = pixel_set - branch_pixels

    polylines: List[np.ndarray] = []

    # ── trace segment components ───────────────────────────────────────────
    if segment_pixels:
        seg_mask = np.zeros(skeleton.shape, np.uint8)
        for x, y in segment_pixels:
            seg_mask[y, x] = 255

        num_labels, labels = cv2.connectedComponents(seg_mask, connectivity=8)

        for label in range(1, num_labels):
            comp_ys, comp_xs = np.where(labels == label)
            comp_set: set = set(zip(comp_xs.tolist(), comp_ys.tolist()))

            # Find endpoint (≤1 neighbour within component) to start from
            start = next(
                (p for p in comp_set if len(_nb8(*p, comp_set)) <= 1),
                next(iter(comp_set)),
            )

            # Greedy chain-follow within the component
            chain: List[Tuple[int,int]] = [start]
            visited: set = {start}
            while True:
                nbrs = [p for p in _nb8(*chain[-1], comp_set) if p not in visited]
                if not nbrs:
                    break
                chain.append(nbrs[0])
                visited.add(nbrs[0])

            # Attach branch-pixel endpoints so paths connect at junctions
            head_b = [b for b in _nb8(*chain[0], pixel_set) if b in branch_pixels]
            tail_b = [b for b in _nb8(*chain[-1], pixel_set) if b in branch_pixels]
            if head_b:
                chain = [head_b[0]] + chain
            if tail_b and (not head_b or tail_b[0] != head_b[0]):
                chain = chain + [tail_b[0]]

            if len(chain) >= 2:
                polylines.append(np.array(chain, dtype=np.float32))

    # Leftover isolated branch-pixel clusters (junction artefacts) are dropped
    # intentionally — stitching them would create spurious connections.

    return polylines


# ─────────────────────────────────────────────────────────────────────────────
# SVG element generation
# ─────────────────────────────────────────────────────────────────────────────

def _is_line_art(binary: np.ndarray, dist: np.ndarray) -> bool:
    """
    Heuristic: is the image line art (thin strokes) or filled solid shapes?

    Line art: most foreground pixels are close to the background boundary,
    so the median distance-transform value is low.
    Threshold ≤ 8 px → line art (covers strokes up to ~16 px wide).
    """
    fg = binary > 0
    if not fg.any():
        return True
    return float(np.median(dist[fg])) <= 8.0


def _pts_to_path_d(pts: np.ndarray, closed: bool, straighten: float = 1.0) -> str:
    """
    Build an SVG path 'd' string from an (N, 2) float array.

    straighten = 0.0  →  smooth Catmull-Rom cubic bezier curves
    straighten = 1.0  →  straight line segments (L commands only)
    0 < straighten < 1 →  Catmull-Rom with tension = straighten
                          (higher = tighter curves, less overshoot)

    The Catmull-Rom → cubic bezier conversion uses:
        CP1 = P1 + α·(P2 − P0)
        CP2 = P2 − α·(P3 − P1)
    where α = (1 − straighten) / 6  (α=1/6 at full smooth, 0 at full straight).
    """
    if len(pts) < 2:
        return ""

    p = pts.astype(float)

    if straighten >= 0.999:
        # ── straight lines only ───────────────────────────────────────────
        cmds = [f"M {p[0,0]:.1f} {p[0,1]:.1f}"]
        end = len(p) - (1 if closed else 0)
        for x, y in p[1:end]:
            cmds.append(f"L {x:.1f} {y:.1f}")
        if closed:
            cmds.append("Z")
        return " ".join(cmds)

    # ── Catmull-Rom cubic bezier ───────────────────────────────────────────
    alpha = (1.0 - straighten) / 6.0

    cmds = [f"M {p[0,0]:.1f} {p[0,1]:.1f}"]

    if closed:
        # Wrap endpoints: [last, p0, p1, ..., p(n-1), p0, p1]
        ext = np.vstack([p[-1:], p, p[:2]])
        n_segs = len(p)
    else:
        # Reflect endpoints: [p0, p0, p1, ..., p(n-1), p(n-1)]
        ext = np.vstack([p[:1], p, p[-1:]])
        n_segs = len(p) - 1

    for i in range(n_segs):
        p0, p1, p2, p3 = ext[i], ext[i+1], ext[i+2], ext[i+3]
        cp1 = p1 + alpha * (p2 - p0)
        cp2 = p2 - alpha * (p3 - p1)
        cmds.append(
            f"C {cp1[0]:.1f},{cp1[1]:.1f} {cp2[0]:.1f},{cp2[1]:.1f} "
            f"{p2[0]:.1f},{p2[1]:.1f}"
        )

    if closed:
        cmds.append("Z")
    return " ".join(cmds)


def extract_svg_elements(
    binary: np.ndarray,
    breakpoints: np.ndarray,
    stroke_widths_pt: List[float],
    simplify_epsilon: float,
    color: str,
    straighten: float = 1.0,
    prune: int = 3,
) -> List[Dict]:
    """
    Extract SVG stroke elements from a binary mask.

    Line art  → skeleton tracing produces single centerline strokes.
                Spur branches shorter than *prune* pixels are removed first.
    Filled    → contour tracing outlines the shape boundary.

    straighten  0.0 = smooth Catmull-Rom bezier curves
                1.0 = straight line segments
    prune       remove skeleton spurs shorter than this many pixels (0 = off)
    """
    if not binary.any():
        return []

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    n = len(stroke_widths_pt)
    line_art = _is_line_art(binary, dist)
    elements: List[Dict] = []

    # Adjust D-P epsilon: preserve more points when using bezier (straighten≈0)
    # so curves have richer data; relax epsilon as straighten→1.
    dp_epsilon = simplify_epsilon * (0.4 + 0.6 * straighten + straighten * 0.6)

    if line_art:
        # ── skeleton centerline tracing ────────────────────────────────────
        skel = skeletonise(binary)
        skel = prune_skeleton_spurs(skel, prune)

        for chain in skeleton_to_polylines(skel):
            if len(chain) < 2:
                continue

            # Width from dist-transform at skeleton pixels
            xi = np.clip(chain[:, 0].astype(int), 0, dist.shape[1] - 1)
            yi = np.clip(chain[:, 1].astype(int), 0, dist.shape[0] - 1)
            w_vals = dist[yi, xi] * 2.0
            avg_w_px = float(np.median(w_vals)) if w_vals.size else 2.0

            b = bucket_index(avg_w_px, breakpoints, n)
            sw_pt = stroke_widths_pt[b]

            # Douglas-Peucker on open path
            pts_raw = chain.reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(pts_raw, dp_epsilon, closed=False)
            spts = approx.reshape(-1, 2)
            if len(spts) < 2:
                continue

            # Detect if this path loops back to its start (closed stroke)
            is_closed = bool(np.linalg.norm(spts[0] - spts[-1]) <= 1.5)
            d = _pts_to_path_d(spts, closed=is_closed, straighten=straighten)
            if d:
                elements.append({"d": d, "stroke_width_pt": sw_pt, "color": color})

    else:
        # ── contour outline tracing (for solid/filled icons) ───────────────
        contours, _ = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < 9:
                continue
            approx = cv2.approxPolyDP(cnt, dp_epsilon, True)
            if len(approx) < 2:
                continue

            # Width: 75th-pct of dist at foreground pixels inside this contour
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
            fg_inside = (mask > 0) & (binary > 0)
            avg_w_px = (
                float(np.percentile(dist[fg_inside] * 2.0, 75))
                if fg_inside.any()
                else 2.0
            )
            b = bucket_index(avg_w_px, breakpoints, n)
            sw_pt = stroke_widths_pt[b]

            spts = approx.reshape(-1, 2).astype(float)
            d = _pts_to_path_d(spts, closed=True, straighten=straighten)
            if d:
                elements.append({"d": d, "stroke_width_pt": sw_pt, "color": color})

    return elements


# ─────────────────────────────────────────────────────────────────────────────
# SVG writer
# ─────────────────────────────────────────────────────────────────────────────


def write_svg(
    out_path: Path,
    elements: List[Dict],
    width: int,
    height: int,
    background: Optional[str] = None,
) -> None:
    """Write a self-contained SVG file."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
    ]
    if background and background.lower() != "transparent":
        lines.append(f'  <rect width="100%" height="100%" fill="{background}"/>')

    for el in elements:
        if not el.get("d"):
            continue
        sw = el["stroke_width_pt"]
        c = el["color"]
        lines.append(
            f'  <path d="{el["d"]}" '
            f'stroke="{c}" stroke-width="{sw:.2f}pt" '
            f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="outline_converter",
        description=(
            "Convert JPG/PNG images to SVG outlines with globally-normalised "
            "stroke widths.\n\n"
            "Stroke widths are measured across ALL input images in a first pass "
            "so that the thin/medium/thick mapping is consistent across the whole "
            "batch, regardless of how each individual icon was drawn."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # Whole folder, default 2-level strokes (1 pt thin / 2.5 pt thick)
  python outline_converter.py icons/

  # Explicit 2-level widths
  python outline_converter.py icons/ -s 1.0 2.5

  # 3-level widths
  python outline_converter.py icons/ -s 0.75 1.5 3.0

  # Single stroke weight (no width measurement pass needed)
  python outline_converter.py icons/ -s 1.5

  # Mixed inputs, custom output folder and colour
  python outline_converter.py icons/ logo.png -o out/ -c "#222222"

  # Light-on-dark source images
  python outline_converter.py dark_icons/ --invert
        """,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input images (JPG/PNG) and/or directories to scan (non-recursive).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default="svg_output",
        metavar="DIR",
        help="Output directory for SVG files (default: svg_output/).",
    )
    p.add_argument(
        "-s",
        "--stroke-widths",
        nargs="+",
        type=float,
        default=[1.0, 2.5],
        metavar="PT",
        help=(
            "Standardised stroke widths in pt, listed thin → thick "
            "(default: 1.0 2.5).  "
            "1 value = single weight; 2 = thin/thick; 3 = thin/medium/thick."
        ),
    )
    p.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=None,
        metavar="0-255",
        help="Manual binarisation threshold (default: auto via Otsu's method).",
    )
    p.add_argument(
        "-e",
        "--simplify",
        type=float,
        default=1.5,
        metavar="PX",
        help="Douglas-Peucker simplification epsilon in pixels (default: 1.5).",
    )
    p.add_argument(
        "-c",
        "--color",
        default="#000000",
        metavar="HEX",
        help="Stroke colour as a CSS hex value (default: #000000).",
    )
    p.add_argument(
        "-b",
        "--background",
        default="transparent",
        metavar="COLOR",
        help="SVG background colour or 'transparent' (default: transparent).",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert image before processing (for light-on-dark source icons).",
    )
    p.add_argument(
        "--straighten",
        type=float,
        default=0.0,
        metavar="0-1",
        help=(
            "Path style: 0.0 = smooth Catmull-Rom bezier curves (default), "
            "1.0 = straight line segments only.  "
            "Values in between blend curve tension."
        ),
    )
    p.add_argument(
        "--prune",
        type=int,
        default=3,
        metavar="PX",
        help=(
            "Remove skeleton spur branches shorter than this many pixels "
            "(default: 3).  Eliminates noise artefacts from skeletonisation.  "
            "Use 0 to disable."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print extra detail (measured widths, path counts, etc.).",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = build_parser().parse_args()

    stroke_widths: List[float] = sorted(args.stroke_widths)
    n_levels = len(stroke_widths)

    images = collect_images(args.inputs)
    if not images:
        print("No images found – nothing to do.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    straighten = max(0.0, min(1.0, args.straighten))
    prune_px   = max(0, args.prune)

    levels_str = "  /  ".join(f"{w} pt" for w in stroke_widths)
    curve_desc = (
        "straight lines" if straighten >= 0.999
        else f"bezier (tension {straighten:.2f})" if straighten > 0.001
        else "smooth bezier curves"
    )
    print(f"\noutline_converter")
    print(f"  images  : {len(images)}")
    print(f"  output  : {out_dir}/")
    print(f"  strokes : {n_levels} level{'s' if n_levels > 1 else ''}  [{levels_str}]")
    print(f"  paths   : {curve_desc}")
    print(f"  prune   : {prune_px} px spur removal")
    print(f"  simplify: {args.simplify} px epsilon")
    if not _HAS_SKIMAGE:
        print("  [info] scikit-image not found – using fallback skeletonisation")

    # ── Pass 1: global stroke-width measurement (only if n_levels > 1) ────────
    breakpoints: np.ndarray = np.array([], dtype=np.float32)
    binary_cache: Dict[Path, np.ndarray] = {}

    if n_levels > 1:
        print(f"\nPass 1 / 2  –  measuring stroke widths …")
        all_widths: List[float] = []

        for img_path in images:
            try:
                binary = load_binary(img_path, args.threshold, args.invert)
                binary_cache[img_path] = binary
                widths = measure_stroke_widths(binary)
                all_widths.extend(widths.tolist())

                if args.verbose:
                    if widths.size:
                        print(
                            f"  {img_path.name:40s}  "
                            f"median {np.median(widths):.1f} px  "
                            f"({widths.size} skeleton pts)"
                        )
                    else:
                        print(f"  {img_path.name:40s}  (no strokes detected)")
            except Exception as exc:
                print(f"  [warn] {img_path.name}: {exc}")

        all_w_arr = np.array(all_widths, dtype=np.float32)
        breakpoints = compute_breakpoints(all_w_arr, n_levels)

        if all_w_arr.size:
            print(
                f"\n  Global width range : {all_w_arr.min():.1f} – {all_w_arr.max():.1f} px"
            )
            if breakpoints.size:
                print(
                    f"  Bucket breakpoints : {' | '.join(f'{b:.1f}' for b in breakpoints)} px"
                )
            bucket_labels = [
                f"≤{breakpoints[0]:.1f} px → {stroke_widths[0]} pt"
            ]
            for i, bp in enumerate(breakpoints[:-1]):
                bucket_labels.append(
                    f"{bp:.1f}–{breakpoints[i+1]:.1f} px → {stroke_widths[i+1]} pt"
                )
            bucket_labels.append(
                f">{breakpoints[-1]:.1f} px → {stroke_widths[-1]} pt"
            )
            for lbl in bucket_labels:
                print(f"    {lbl}")
        else:
            print("  [warn] No stroke widths measured – all images will use thinnest level.")
    else:
        print(f"\nSingle stroke level – skipping measurement pass.")

    # ── Pass 2: generate SVGs ─────────────────────────────────────────────────
    pass_label = "Pass 2 / 2" if n_levels > 1 else "Processing"
    print(f"\n{pass_label}  –  generating SVGs …")

    ok = 0
    for img_path in images:
        try:
            binary = (
                binary_cache[img_path]
                if img_path in binary_cache
                else load_binary(img_path, args.threshold, args.invert)
            )
            h, w = binary.shape

            elements = extract_svg_elements(
                binary,
                breakpoints,
                stroke_widths,
                args.simplify,
                args.color,
                straighten=straighten,
                prune=prune_px,
            )

            out_path = out_dir / (img_path.stem + ".svg")
            write_svg(out_path, elements, w, h, args.background)

            if args.verbose:
                sw_counts: Dict[float, int] = {}
                for el in elements:
                    sw = el["stroke_width_pt"]
                    sw_counts[sw] = sw_counts.get(sw, 0) + 1
                breakdown = "  ".join(
                    f"{sw}pt×{cnt}" for sw, cnt in sorted(sw_counts.items())
                )
                print(f"  ✓  {img_path.name:40s}  {len(elements):3d} paths  [{breakdown}]")
            else:
                print(f"  ✓  {img_path.name}  →  {out_path.name}")
            ok += 1

        except Exception as exc:
            print(f"  ✗  {img_path.name}: {exc}")

    print(f"\nDone – {ok} / {len(images)} converted  →  {out_dir}/\n")


if __name__ == "__main__":
    main()
