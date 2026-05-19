# outline-converter


![Conversion example](docs/conversion-example.png.jpg "Outline conversion")
Convert JPG/PNG raster images to clean, stroke-based SVG outlines.
Designed for icon sets — stroke widths are measured and normalised globally
across an entire batch so every icon in a run uses the same thin/thick mapping.

## How it works

The script runs in two passes:

**Pass 1 — global stroke-width measurement**
Every input image is binarised and skeletonised. The distance transform at each
skeleton pixel gives the local half-width of the stroke. These measurements are
pooled across the whole batch and split into N percentile buckets, so the
breakpoints between "thin" and "thick" are consistent regardless of how each
individual icon was drawn.

**Pass 2 — path extraction and SVG output**
For **line-art icons** (thin strokes on a plain background) the skeleton is
traced directly into ordered polylines — one centerline path per stroke, no
double outlines. For **filled/solid icons** the outer contour boundary is
traced instead. Each path is assigned a standardised stroke width based on the
global buckets from Pass 1, then written as a `<path>` element with
`fill="none"`.

Path geometry can be output as smooth **Catmull-Rom bezier curves** or
**straight line segments**, controlled by the `--straighten` parameter.


## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**

| Package | Version | Notes |
|---|---|---|
| opencv-python | ≥ 4.5 | core image processing |
| numpy | ≥ 1.20 | array ops |
| scikit-image | ≥ 0.19 | best-quality skeletonisation (optional but recommended) |

If `scikit-image` is not installed the script falls back to a pure-OpenCV
morphological thinning implementation.


## Usage

```
python outline_converter.py [inputs ...] [options]
```

`inputs` can be any mix of image files (`.jpg`, `.jpeg`, `.png`) and
directories. Directories are scanned one level deep (non-recursive).

### Common recipes

```bash
# Whole folder, default 2 stroke levels (1 pt thin / 2.5 pt thick)
python outline_converter.py icons/

# White-on-dark source images (invert before binarising)
python outline_converter.py icons/ --invert

# Smooth bezier output (default)
python outline_converter.py icons/ --invert --straighten 0.0

# Straight lines only
python outline_converter.py icons/ --invert --straighten 1.0

# Custom stroke levels
python outline_converter.py icons/ -s 1.0 2.5          # 2 levels (default)
python outline_converter.py icons/ -s 0.75 1.5 3.0     # 3 levels
python outline_converter.py icons/ -s 1.5              # single weight

# Custom output, colour, and simplification
python outline_converter.py icons/ -o svg/ -c "#1a1a1a" -e 1.0

# More aggressive noise cleanup for rough/hand-drawn sources
python outline_converter.py icons/ --prune 8

# Verbose (shows per-image measurements and path breakdown)
python outline_converter.py icons/ -v
```


## Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output-dir DIR` | `svg_output/` | Output directory for SVG files |
| `-s`, `--stroke-widths PT…` | `1.0 2.5` | Standardised stroke widths in pt, listed thin → thick. 1 value = single weight, 2 = thin/thick, 3 = thin/medium/thick |
| `-t`, `--threshold 0-255` | auto | Manual binarisation threshold. Default uses Otsu's method |
| `-e`, `--simplify PX` | `1.5` | Douglas-Peucker epsilon in pixels. Higher = fewer path nodes |
| `-c`, `--color HEX` | `#000000` | Stroke colour |
| `-b`, `--background COLOR` | `transparent` | SVG background fill, or `transparent` |
| `--invert` | off | Invert image before processing. Use for white-on-dark source icons |
| `--straighten 0-1` | `0.0` | `0` = smooth Catmull-Rom bezier curves, `1` = straight line segments, values in between blend the curve tension |
| `--prune PX` | `3` | Remove skeleton spur branches shorter than this many pixels. Reduces noise artefacts from skeletonisation. Use `0` to disable |
| `-v`, `--verbose` | off | Print per-image width measurements and path-count breakdown |


## Stroke-width normalisation

The key feature is **global normalisation across the batch**. Rather than
measuring each icon independently, all stroke-width measurements are collected
in Pass 1 and split into N equal-percentile buckets. The breakpoints are then
applied uniformly in Pass 2.

Example output with `-s 1.0 2.5 -v`:

```
Pass 1 / 2  –  measuring stroke widths …
  icon-menu.png      median  4.0 px  (123 skeleton pts)
  icon-close.png     median 10.0 px   (89 skeleton pts)
  icon-circle.png    median  7.2 px  (123 skeleton pts)

  Global width range : 4.0 – 16.0 px
  Bucket breakpoints : 7.2 px
    ≤7.2 px → 1.0 pt
    >7.2 px → 2.5 pt
```

Every icon uses that same 7.2 px breakpoint, so thin strokes across the whole
set are consistently 1 pt and thick strokes are consistently 2.5 pt.


## Output format

Each SVG preserves the source pixel dimensions as its `viewBox` and `width`/
`height`. Paths use `pt` units for stroke widths and have `fill="none"`,
`stroke-linecap="round"`, and `stroke-linejoin="round"`.

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <path d="M 12.0 17.0 C …" stroke="#000000" stroke-width="1.00pt"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
```


## Tips

- **White-on-dark icons** — always use `--invert`. The binariser expects dark
  ink on a light background.
- **Noisy or hand-drawn sources** — increase `--prune` (try 5–10) to remove
  skeleton whiskers and stray pixel chains before tracing.
- **Too angular / too smooth** — `--straighten` is the primary control.
  `0.3`–`0.5` is usually a good middle ground for hand-drawn work.
- **Too many nodes** — raise `-e` (try `2.0`–`4.0`). This widens the
  Douglas-Peucker tolerance, reducing node count in both L and C paths.
- **Single consistent weight** — pass one value to `-s` (e.g. `-s 1.5`).
  Pass 1 is skipped entirely and every path gets that weight.


## Pipeline overview

```
JPG / PNG
    │
    ▼
load_binary()
  ├─ alpha composite onto white
  ├─ grayscale + Gaussian blur
  ├─ Otsu binarisation (or manual threshold)
  └─ morphological open (noise removal)
    │
    ├──[Pass 1]──────────────────────────────────────────────────────────┐
    │  distanceTransform + skeletonise → skeleton pixel widths           │
    │  collect across all images → compute global percentile breakpoints │
    └────────────────────────────────────────────────────────────────────┘
    │
    ├──[Pass 2]──────────────────────────────────────────────────────────┐
    │                                                                    │
    │  is_line_art?                                                      │
    │  ├─ YES → skeletonise → prune_spurs → skeleton_to_polylines        │
    │  │          → D-P simplify → assign width bucket                  │
    │  │          → _pts_to_path_d (bezier or straight)                 │
    │  └─ NO  → findContours → D-P simplify → assign width bucket       │
    │             → _pts_to_path_d (bezier or straight)                 │
    │                                                                    │
    └─ write_svg() → .svg file per image ───────────────────────────────┘
```


## Thanks
https://github.com/syaltamimi/image-to-vector
