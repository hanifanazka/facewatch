# facewatch

A webcam face-recognition binary that drives **pupi**'s push-pipeline
end-to-end: live webcam frames flow through YuNet (default) or SCRFD detection,
AuraFace recognition, gallery matching, and an annotated overlay — which is
then published over RTSP to a **mediamtx** server for any client to watch.

```
webcam (GStreamer) ──► Detection ──► Recognition ──► Matching ──► OverlaySink ──► appsrc
       RGB frame      YuNet/SCRFD    112×112 → 512-d   cosine vs   boxes/labels │ x264enc (ultrafast,
                                                         gallery     fps, verbose  │ zerolatency, 2000 kbps)
                                                                   ┌──────────────┘
                                                                   ▼
                                                             rtspclientsink ──► mediamtx :8554 (tcp)
```


## Setup

Requires Rust (edition 2024). `pupi` is a local path dependency:

```toml
[dependencies]
pupi = { path = "../pupi" }
ort = "2.0.0-rc.13"   # builds ONNX Runtime (download-binaries) at compile time
```

Build (first build compiles ONNX Runtime, so expect a long wait):

```sh
cargo build --release
```

### Models

Three ONNX models are **downloaded and SHA-256 verified at build time** and then
**embedded into the binary**: `build.rs` ensures `scrfd_10g_bnkps.onnx`,
`face_detection_yunet_2026may.onnx`, and `glintr100.onnx` exist in `models/`
(fetching them when absent — SCRFD/AuraFace from
`https://huggingface.co/fal/AuraFace-v1`, YuNet from the OpenCV model zoo —
and checking every file against the exact published hash before the crate
compiles), and `src/models.rs` packages the verified files with
`include_bytes!`. The app performs no downloads or checksum logic at runtime.

The models directory defaults to `models/` next to `Cargo.toml`; override it
with `FACEWATCH_MODELS_DIR=/path` when building.

> **Why AuraFace (commercial licensing).** Recognition is done by AuraFace
> (`glintr100.onnx`), a ResNet100 + ArcFace-loss model published under an
> **Apache-2.0** license and trained on commercially available data for
> commercial use. We verified this is not insightface's identically-named
> non-commercial model: the file is byte-identical to fal's published blob and
> produces embeddings essentially orthogonal to insightface's Glint360K-R100
> (cos ≈ 0.02), i.e. it is fal's own checkpoint.
>
> The detector (`scrfd_10g_bnkps.onnx`, SCRFD architecture) is also published
> through the same AuraFace repo under its Apache-2.0 model card, but its ONNX
> export stamp (`pytorch 1.6`, 2021) shows it is the original insightface
> export re-hosted — insightface's own pretrained models carry a
> "non-commercial research only" notice. The **default detector is now YuNet**
> `face_detection_yunet_2026may.onnx` from the OpenCV model zoo, Apache-2.0,
> no insightface lineage), which removes this concern entirely; SCRFD remains
> embedded behind `--detector scrfd` for A/B comparison. All Rust dependencies
> are permissive (MIT / Apache-2.0); no GPL/AGPL code is linked.

| Model | Input | Output |
|---|---|---|
| `scrfd_10g_bnkps.onnx` (~16 MiB) | `[1,3,640,640]` f32, `(x-127.5)/128`, RGB | 9× rank-2 `[count,…]` (score/bbox/kps, strides 8/16/32) |
| `face_detection_yunet_2026may.onnx` (~224 KiB) | `[1,3,-1,-1]` f32 dynamic H/W (640 default; any multiple of 32), BGR, raw `[0,255]`, no mean | 12× (cls/obj/bbox/kps, strides 8/16/32) |
| `glintr100.onnx` (~248 MiB) | `[1,3,112,112]` f32, `(x-127.5)/127.5` (no baked-in Sub/Mul) | `[1,512]` |

**Model integrity.** `build.rs` verifies every model file against the exact
SHA-256 fal publishes (HF LFS blob IDs) before embedding; a missing,
truncated, or tampered file abort the download-and-verify step *at build
time*, so the bytes packaged in the binary are always the published
artifacts:

```
scrfd_10g_bnkps.onnx  5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91
face_detection_yunet_2026may.onnx  ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0
glintr100.onnx        a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60
```

## Usage

Global flags come **before** the subcommand:

```sh
# Register a face under a name (from a photo, or one webcam snapshot if no image given)
facewatch register alice photo.jpg

# Live webcam recognition, published to mediamtx (--frames N stops after N frames)
facewatch run
facewatch run --frames 20 --verbose

# Faster detection at lower input resolution (yuNet now uses the dynamic-shape
# 2026may export, so it accepts 320 too — not just scrfd)
facewatch run --input-size 320

# ^C stops any live run gracefully between frames (and prints the --profile summary)

# Publish to a different endpoint, or process headlessly without streaming
facewatch run --rtsp rtsp://127.0.0.1:8554/cam1
facewatch run --no-rtsp

# Analyze a single still image through the same chain
facewatch image photo.jpg
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--detector NAME` | `yunet` | Detector backend: `yunet` (OpenCV zoo, default) or `scrfd` |
| `--threads N` | logical CPUs | ONNX intra-op threads per model session |
| `--input-size N` | `640` | Detector input size: `320` (faster) or `640`; the dynamic-shape yuNet (2026may) and scrfd both accept any multiple of 32 |
| `--threshold F` | `0.40` | Minimum cosine similarity for a gallery match |
| `--gallery PATH` | `~/.facewatch/gallery.json` | Gallery JSON path |
| `--rtsp URL` | `rtsp://127.0.0.1:8554/facewatch` | Publish the annotated stream to this RTSP URL |
| `--no-rtsp` | off | Do not publish to RTSP (headless processing) |
| `--save-dir DIR` | — | Save annotated frames (`frame_000001.png`, …) |
| `--save-every N` | `1` | Save every Nth annotated frame |
| `--verbose` | off | Print one JSON line per processed frame (boxes, kps, matches) |

## Streaming with mediamtx

`facewatch run` encodes the annotated frames (via the exact pipeline
`appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=ultrafast tune=zerolatency bitrate=2000 cabac=false dct8x8=false key-int-max=1 ! rtspclientsink protocols=tcp`) and
publishes them to the `--rtsp` URL — by default `rtsp://127.0.0.1:8554/facewatch`.

The extra `x264enc` flags are for decoder compatibility on openSUSE builds,
where `libopenh264` is the *only* H.264 decoder (ffmpeg ships without the
native h264 decoder there):

- `cabac=false dct8x8=false` — OpenH264 can't decode High/Main; with these off
  (plus zerolatency's `bframes=0`) x264 emits a Baseline stream it accepts.
- `key-int-max=1` — every frame is an IDR, so clients joining mid-stream
  (between keyframes) start decoding on the first frame instead of erroring on
  a GOP of P-slices until the next keyframe.

1. **Install the GStreamer plugins** (openSUSE Leap 16):

   ```sh
   sudo zypper install gstreamer-rtsp-server-devel      # rtspclientsink (OSS)
   sudo zypper addrepo --refresh https://ftp.gwdg.de/pub/linux/misc/packman/suse/openSUSE_Leap_16.0/ packman
   sudo zypper --gpg-auto-import-keys refresh
   sudo zypper install gstreamer-plugins-ugly-codecs    # x264enc + libx264 (Packman)
   gst-inspect-1.0 x264enc && gst-inspect-1.0 rtspclientsink   # verify
   ```

   > Packman's `x264enc` lives in the `gstreamer-plugins-ugly-codecs`
   > subpackage, not the OSS-repo `gstreamer-plugins-ugly`.
   > Note that the openSUSE OSS `gstreamer-plugins-ugly` has no x264.

2. **Run mediamtx** (binary and config kept in `vendor/mediamtx/` — pass the config
   explicitly, otherwise it looks for `mediamtx.yml` in the current directory):

   ```sh
   ./vendor/mediamtx/mediamtx ./vendor/mediamtx/mediamtx.yml
   ```

3. **Run facewatch** in another terminal:

   ```sh
   cargo run --release -- run
   ```

4. **Watch it** with `mpv` (low-latency profile, as configured):

   ```sh
   mpv --profile=low-latency rtsp://127.0.0.1:8554/facewatch
   ```

   While publishing, the stream is listed in mediamtx's API at
   `http://127.0.0.1:9997/v3/paths/get/facewatch`.

## Pipeline details

Frame acquisition keeps **only the newest frame**: the source appsink runs with
`max-buffers=1, drop=true, sync=false`, so when detection/recognition fall
behind the camera, older frames are discarded at the source and the pipeline
always processes the freshest frame instead of a growing stale backlog.

Frames from GStreamer are forced to `RGB` via a `capsfilter` on the appsink, so
no channel swapping happens anywhere in the app. All preprocessing mirrors the
insightface reference implementations:

- **Detection** — selectable via `--detector`:
  - **YuNet (default)** — letterbox resize to 640×640 (top-left, zero-padded),
    BGR `[0,255]` blob (no mean), decode the 12 per-stride outputs
    (cls/obj/bbox/kps, strides 8/16/32) with `score = √(cls·obj)`, keep
    `score ≥ 0.9`, NMS @ IoU 0.3. Mirrors OpenCV's `FaceDetectorYN`; no
    landmark reorder is needed (YuNet's order coincides with the app's).
  - **SCRFD** — letterbox resize to 640×640, `(x-127.5)/128`, anchors at
    `(col·stride, row·stride)` duplicated ×2, decode `score/bbox/kps` for
    strides 8/16/32, NMS @ IoU 0.4, keep `score ≥ 0.5`.
  - Both map boxes/kps back to source-frame pixels.
- **Alignment** — five-landmark similarity transform (`SimilarityTransform`
  Umeyama equivalent) onto the canonical ArcFace 112×112 template, then
  `warpAffine`-style bilinear warp with zero border (`warpAffine` samples
  `M⁻¹·(x,y)`, exactly like OpenCV).
- **Recognition (AuraFace)** — `(x-127.5)/127.5`, L2-normalized 512-d
  embedding.
- **Matching** — cosine similarity against the gallery, best match at/above
  `--threshold`.

### Validation

`facewatch` was validated against an independent `onnxruntime` + `skimage`
reference using the same ONNX files:

- SCRFD boxes agree to ≤ 0.12 px, landmarks to ≤ 0.07 px (interpolation-only
  differences vs OpenCV).
- The alignment matrix agrees with `skimage.transform.SimilarityTransform` to
  ~3.5e-5.
- Embeddings agree at cosine **0.9997** between the Rust pipeline and the
  Python reference; registering and re-analyzing the same photo matches at
  cosine **0.99999**.