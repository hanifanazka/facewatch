# facewatch source — components & relations

## Components (13 source files in `src/`)

| # | File | Role / Key structs |
|---|------|-------------------|
| 1 | `main.rs` | CLI (`Cli`, `Command`), entry (`main`), `register`, `run`, `analyze_image`, `Chain::new` wiring |
| 2 | `face.rs` | Shared types (`Embedding`, `FaceEmbeddings`, `FaceMatches`, `DetectedFace`, `Kps`, `Rgb`), helpers (`fill_nchw`, `fill_nchw_bgr`, `load_session`, `norm_crop`), `resize_frame`, `Array3U8` |
| 3 | `elements.rs` | Pipeline (`Stage<I,O>`), detectors (`Detector`: `Scrfd`/`Yunet`), recognizers (`Recognizer_`: `AuraFace`/`SFace`), `RecognitionElement`, `MatchElement` (takes `Recognizer`), `OverlaySink`, `Chain` construction / linking |
| 4 | `gallery.rs` | `GalleryEntry` (`name`, `auraface` [alias `embedding`], `sface`), `Gallery` (load/save, `register` dual-embeddings, `match_faces` per-`Recognizer`, `summary`) |
| 5 | `models.rs` | Embedded `Models` (from `build.rs` meta): `scrfd`, `yunet`, `auraface`, `sface` |
| 6 | `aura.rs` | `AuraFace` session (`load`, `embed` with `fill_nchw(127.5)`, L2-normalized 512-d output) |
| 7 | `sface.rs` | `SFace` session (`load`, `embed` with manual BGR + `(x-127.5)/127.5`, L2-normalized 128-d output `fc1`) |
| 8 | `scrfd.rs` | `Scrfd` detector (`load`, `detect` with `fill_nchw_bgr` at 640) |
| 9 | `yunet.rs` | `Yunet` detector (`load`, `detect`, dynamic-shape 320/640, `fill_nchw_bgr`) |
| 10 | `align.rs` | Landmark-based crop (`norm_crop` → 112×112 RGB) |
| 11 | `draw.rs` | Overlay drawing (`draw_rect`, `draw_text`, `fill_rect`, `draw_kps`, `Rgb`) |
| 12 | `profile.rs` | Timing (`enable`, `time`, `record`, `frame_done`, `report`) — stages: detect, align, embed, draw, publish, total |
| 13 | `rtsp.rs` | RTSP publisher (`RtspPublisher`) |

## Dependency graph (uses / calls direction)

- `main` → `elements`, `models`, `gallery`, `aura`, `sface`, `scrfd`, `face`, `align`, `rtsp`
- `elements` → `align`, `aura`/`sface`, `draw`, `face`, `gallery`, `models`, `profile`, `scrfd`, `yunet`
- `gallery` → `face` (`Embedding`, `Match`)
- `models` → `build.rs` meta (`include!`)
- `aura` / `sface` → `face` (`Array3U8`, `Embedding`, `fill_nchw` / manual blob, `load_session`)
- `scrfd` / `yunet` → `face` (`load_session`, `fill_nchw_bgr`)
- `align` → `face`
- `draw` → `face` (`Rgb`)
- `profile` → standalone (used by `elements`, `main`)
- `rtsp` → standalone (used by `main`)

## Pipeline flow (`Chain::new` order)

`main`/`run` builds:
1. `Detector::load` (`scrfd` or `yunet`)
2. `RecognitionElement::new` (`AuraFace` or `SFace` via `Recognizer_`)
3. `MatchElement::new` (`Gallery` + threshold + `Recognizer`)
4. `OverlaySink::new` (`OverlayOptions` + display channel)

Linked via `link_upstream`: `detection` → `recognition` → `matching` → `overlay`

## Backward-compat / dual-store

- `GalleryEntry` fields: `name`, `auraface` (`#[serde(alias = "embedding")]` for old JSON), `sface` (`#[serde(default)]`)
- `register()` writes both `auraface` (512-d) and `sface` (128-d)
- `match_faces()` selects embedding space by `Recognizer::AuraFace` or `SFace`; skips entries missing that space
