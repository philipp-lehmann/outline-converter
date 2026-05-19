# CLAUDE.md — outline-converter

Developer reference for working with this codebase using Claude.

---

## Project purpose

`outline_converter.py` converts raster images (JPG, PNG) into stroke-based SVG
outlines. It is designed for icon batches: stroke widths are measured globally
across all input images in a first pass so that thin/thick mappings are
consistent across the whole set.

---

## Architecture

Single-file script. No package structure. All logic lives in
`outline_converter.py`. The pipeline is:

```
load_binary()  →  [Pass 1: measure widths]  →  [Pass 2: extract paths]  →  write_svg()
```

### Key functions

| Function | Purpose |
|---|---|
| `load_binary(path, threshold, invert)` | Load image → binary mask (foreground=255). Handles RGBA, grayscale, alpha compositing onto white |
| `skeletonise(binary)` | 1-px skeleton via scikit-image (preferred) or OpenCV morphological thinning fallback |
| `measure_stroke_widths(binary)` | Distance transform + skeleton → array of stroke width measurements in px. Used in Pass 1 only |
| `compute_breakpoints(all_widths, n_buckets)` | Global percentile breakpoints for N stroke levels |
| `prune_skeleton_spurs(skel, iterations)` | Remove short spur artefacts by iteratively deleting 1-neighbour endpoint pixels |
| `skeleton_to_polylines(skeleton)` | Trace skeleton pixel graph into ordered (N,2) float32 arrays. Handles endpoints, branch points, closed loops |
| `_is_line_art(binary, dist)` | Heuristic: median dist-transform ≤ 8 px → line art (use skeleton tracing), else filled shape (use contour tracing) |
| `_pts_to_path_d(pts, closed, straighten)` | (N,2) array → SVG `d` string. `straighten=0` = Catmull-Rom bezier, `1` = straight L commands |
| `extract_svg_elements(...)` | Orchestrates skeletonise/contour → D-P simplify → width assignment → path_d per image |
| `write_svg(path, elements, w, h, bg)` | Write final SVG file with `fill="none"` stroke paths |

---

## Design decisions

**Why not vtracer or potrace?**
Both output fill-based paths. We need stroke-based paths with per-path width
assignment derived from the measured source stroke thickness. This requires a
custom pipeline.

**Why two passes?**
Global normalisation. If each image measured its own widths independently, a
batch of mixed-weight icons would end up with inconsistent stroke weights. Pass 1
pools all measurements so the thin/thick breakpoints are calibrated to the
whole set.

**Skeleton vs contour for line art**
`findContours` on a binary stroke region traces the *boundary* of the stroke —
producing two parallel edge paths (double outline). Skeleton tracing gives the
single centerline, which is what "convert to outline" actually means for line art.

**Branch-pixel handling in `skeleton_to_polylines`**
Branch pixels (≥3 skeleton neighbours) are removed before finding connected
components. Each component is then a simple chain that can be greedily
traversed. The nearest branch pixel is reattached at each chain end so strokes
connect cleanly at junctions. Isolated leftover branch-pixel clusters are
dropped (not stitched into paths) to avoid spurious connections.

**Catmull-Rom bezier**
`_pts_to_path_d` uses the Catmull-Rom → cubic bezier conversion:
```
α = (1 − straighten) / 6
CP1 = P1 + α · (P2 − P0)
CP2 = P2 − α · (P3 − P1)
```
At `straighten=0`, α=1/6 gives standard smooth Catmull-Rom. As
`straighten→1`, α→0 and the control points collapse onto the anchor points,
degrading naturally to straight L segments without any branching logic.

**D-P epsilon scaling with `straighten`**
`dp_epsilon = base * (0.4 + 1.2 * straighten)` — bezier paths get a smaller
epsilon (more waypoints to work with for smooth curves); straight-line paths get
a larger epsilon (more aggressive simplification to clean polygons).

---

## Parameters and defaults

| Param | Default | Notes |
|---|---|---|
| `--stroke-widths` | `1.0 2.5` | 1–3 values in pt, thin→thick |
| `--straighten` | `0.0` | 0 = bezier, 1 = straight |
| `--prune` | `3` | Spur removal iterations. Good range: 3–10 |
| `--simplify` | `1.5` | D-P epsilon in px. Scales with `straighten` internally |
| `--threshold` | auto | Manual override for Otsu binarisation |
| `--invert` | off | Must use for white-on-dark source images |

---

## Common tasks

**Add a new output format**
Extend `write_svg()` or add a parallel `write_pdf()` / `write_dxf()` function.
`extract_svg_elements()` returns plain dicts (`{d, stroke_width_pt, color}`) —
the geometry is format-agnostic.

**Change the line-art detection threshold**
Edit the `<= 8.0` value in `_is_line_art()`. Higher = more images treated as
line art (use skeleton). Lower = more images treated as filled (use contours).

**Add recursive directory scanning**
`collect_images()` uses `p.glob(f"*{ext}")` (one level). Change to
`p.rglob(f"*{ext}")` for recursive.

**Adjust bezier smoothness independently of D-P**
`extract_svg_elements()` calls `_pts_to_path_d(..., straighten=straighten)`.
You can pass a separate `curve_tension` value if you want to decouple
simplification from smoothness.

---

## Dependencies

```
opencv-python >= 4.5   # cv2: binarisation, distance transform, contours, D-P
numpy >= 1.20          # array ops throughout
scikit-image >= 0.19   # skeletonize() — optional, falls back to OpenCV thinning
```

Install: `pip install -r requirements.txt`

---

## Testing

No formal test suite. Quick smoke test:

```bash
# Generate synthetic test icons
python3 - <<'EOF'
import cv2, numpy as np
img = 255 * np.ones((64,64), dtype=np.uint8)
cv2.line(img, (10,32), (54,32), 0, 3)   # thin horizontal
cv2.line(img, (32,10), (32,54), 0, 8)   # thick vertical
cv2.imwrite('/tmp/test_cross.png', img)
EOF

python outline_converter.py /tmp/test_cross.png -o /tmp/svg_out/ -s 1.0 2.5 -v --straighten 0.0
# Expect: 3 paths (horizontal + 2 vertical arms split at junction)
#         thin paths at 1pt, thick paths at 2.5pt
#         SVG contains C commands (bezier)
```

When `--straighten 1.0` the same command should produce `L` commands only.
