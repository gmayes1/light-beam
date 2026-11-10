# Light Beam

Headless bird-photo processing on Apple silicon. Light Beam develops RAW files with Core Image's RAW decoder, finds the bird with Vision, scores focus and stacks bursts on the GPU with Metal and MPSGraph, and writes a finished JPEG with a look that separates the bird from its habitat. The heavy per-pixel work runs on the GPU, and your photos stay on your Mac.

## Tools

| Tool | What it does |
|---|---|
| `look_pipeline recipe.json` | Develops one RAW into a finished JPEG: subject mask, white balance, bird and environment looks, auto crop, quality checks. Writes `<out>.jpg`, `<out>_mask.png` and `<out>_report.json`. |
| `stack_burst out_dir reference.RAW other.RAW ...` | Aligns a burst to the reference frame (Vision whole-frame registration, then a tile search on MPSGraph) and merges it with outlier rejection, cutting noise without ghosting a moving bird. |
| `bird_focus file.RAW ...` | Scores focus on the bird and on the autofocus point for each frame and prints a TSV: `file focus eye_focus detail noise bird_px flags`. Use it to pick the sharpest frame of a burst. |

## Requirements

- macOS 26 or later on Apple silicon (Core Image RAW, Vision, Metal, MPSGraph)
- Xcode command-line tools (`swiftc`)
- [ExifTool](https://exiftool.org) on `PATH`, for capture times, ISO and autofocus-point metadata

Autofocus-point metadata is read from Nikon MakerNotes (tested on the Z8); other cameras fall back to the Vision subject mask.

## Build

```sh
./build.sh
```

Binaries go to `bin/`. `look_pipeline` finds `stack_burst` next to itself, or in `$LIGHT_BEAM_BIN`.

## Recipe

```json
{
 "raw": "/path/to/photos/DSC_0001.NEF",
 "out": "/path/to/output/DSC_0001_edit",
 "auto_crop": true,
 "destination": "web",
 "stack_auto": true
}
```

| Field | Meaning |
|---|---|
| `raw`, `out` | Input RAW and output path without extension |
| `crop` | `[x, y, width, height]` in full-frame pixels from the top-left corner, instead of `auto_crop` |
| `auto_crop`, `ratio` | Crop around the bird, optionally forced to `"3:2"`, `"4:5"` or `"1:1"` |
| `destination` | `web`, `contest`, `print_8x12`, `print_12x18`, `print_16x24`, `print_20x30`: sets the minimum long side checked in the report |
| `stack`, `stack_auto`, `stack_min_iso` | Burst frames to stack, or find burst neighbours automatically above an ISO |
| `look` | Overrides for the look parameters (`bird_ev`, `bird_warm_k`, `env_ev_far`, `env_cool_k`, `global_vibrance`, ...) defined in `Sources/look_pipeline.swift` |

## Options

`stack_burst`: `SEARCH` (tile search radius, default 4), `REJECT` (outlier threshold in noise sigmas, default 3), `DECODER=8` (use RAW decoder version 8), `CROP=x,y,side`, `RAW_OUT=1` (write the linear stack as half-float), `NO_TIFF=1`.

`bird_focus`: `WORKERS` (parallel frames, default 8), `CI_MEMORY_MB`, `DECODER=8`, `VISION_DEVICE` (`ane`, `gpu` or `mix`), `TIMING=1`.

## Licence

MIT
