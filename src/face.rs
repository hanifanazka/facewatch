//! Payload types that flow between pupi pipeline stages, plus small helpers
//! for converting between `ndarray::Array3<u8>` frames and `image::RgbImage`.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use image::RgbImage;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// Five 2D landmark points: left eye, right eye, nose, left mouth, right mouth.
pub type Kps = [[f32; 2]; 5];

/// A single detected face, in source-frame pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectedFace {
    /// `[x1, y1, x2, y2]` bounding box.
    pub bbox: [f32; 4],
    /// Detection confidence (0..1).
    pub score: f32,
    /// Five facial landmarks.
    pub kps: Kps,
}

/// Output of the detection stage: a source frame plus the faces found in it.
#[derive(Clone, Debug)]
pub struct FaceFrame {
    /// The RGB source frame shared with downstream stages.
    pub src: Arc<Array3U8>,
    pub faces: Vec<DetectedFace>,
}

/// A normalized face embedding (512 floats for AuraFace, L2-normalized).
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Vec<f32>);

/// Output of the recognition stage: an embedding for every detected face.
#[derive(Clone, Debug)]
pub struct FaceEmbeddings {
    pub src: Arc<Array3U8>,
    pub faces: Vec<DetectedFace>,
    pub embeddings: Vec<Embedding>,
}

/// The result of matching one embedding against the gallery.
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    /// The gallery subject name.
    pub name: String,
    /// Cosine similarity against the gallery entry.
    pub score: f32,
}

/// Output of the matching stage: a best gallery match (or `None`) per face.
#[derive(Clone, Debug)]
pub struct FaceMatches {
    pub src: Arc<Array3U8>,
    pub faces: Vec<DetectedFace>,
    pub matches: Vec<Option<Match>>,
}

/// Fresh, unaligned RGB frame payload (`(height, width, 3)`, row-major).
pub type Array3U8 = ndarray::Array3<u8>;

/// Converts an `Array3<u8>` frame to an `RgbImage`, copying the pixel data.
pub fn arr3_to_rgb(frame: &Array3U8) -> Option<RgbImage> {
    let height = frame.shape()[0];
    let width = frame.shape()[1];
    RgbImage::from_raw(width as u32, height as u32, frame.as_slice()?.to_vec())
}

/// Converts an `RgbImage` to an `Array3<u8>` frame.
pub fn rgb_to_arr3(img: &RgbImage) -> Array3U8 {
    let (width, height) = img.dimensions();
    Array3U8::from_shape_vec((height as usize, width as usize, 3), img.as_raw().to_vec())
        .expect("RGB image buffer is always exactly width*height*3 bytes")
}

/// A color used when drawing overlays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb([u8; 3]);

impl Rgb {
    pub const RED: Rgb = Rgb([255, 64, 64]);
    pub const GREEN: Rgb = Rgb([64, 255, 64]);
    pub const WHITE: Rgb = Rgb([255, 255, 255]);
    pub const BLACK: Rgb = Rgb([0, 0, 0]);

    pub fn array(self) -> [u8; 3] {
        self.0
    }
}

/// Copies an RGB frame into the top-left of a zero-initialized `NCHW` float
/// blob, normalizing pixels as `(pixel - 127.5) / denom`. `out` must be
/// `(1, 3, H, W)`; the frame is written at the top-left and the rest stays
/// zero, which is what both ONNX models expect as their input.
pub fn fill_nchw(frame: &Array3U8, denom: f32, out: &mut Array4<f32>) {
    let (height, width) = (frame.shape()[0], frame.shape()[1]);
    for y in 0..height {
        for x in 0..width {
            out[[0, 0, y, x]] = (frame[[y, x, 0]] as f32 - 127.5) / denom;
            out[[0, 1, y, x]] = (frame[[y, x, 1]] as f32 - 127.5) / denom;
            out[[0, 2, y, x]] = (frame[[y, x, 2]] as f32 - 127.5) / denom;
        }
    }
}

/// Resizes a frame to `(new_width, new_height)` using bilinear filtering and
/// returns a new frame.
pub fn resize_frame(frame: &Array3U8, new_width: usize, new_height: usize) -> Array3U8 {
    let img = arr3_to_rgb(frame).expect("frame has a valid RGB layout");
    let resized = image::imageops::resize(
        &img,
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::Triangle,
    );
    rgb_to_arr3(&resized)
}

/// Letterbox-resizes `frame` into `size`x`size` (aspect-fit, top-left,
/// zero-padded) and returns the resized frame plus the per-axis scale factors
/// `(scale_x, scale_y)` needed to map detections back to source pixels.
///
/// Shared by the SCRFD and YuNet detectors, which use the same framing.
pub fn letterbox(frame: &Array3U8, size: usize) -> (Array3U8, f32, f32) {
    let (src_h, src_w) = (frame.shape()[0], frame.shape()[1]);

    // Fit the src aspect ratio inside `size`x`size`, top-left. Sources taller
    // than wide fix the height; wider sources fix the width.
    let im_ratio = src_h as f32 / src_w as f32;
    let (new_w, new_h) = if im_ratio > 1.0 {
        ((size as f32 / im_ratio).floor().max(1.0) as usize, size)
    } else {
        (size, (size as f32 * im_ratio).floor().max(1.0) as usize)
    };
    let scale_x = new_w as f32 / src_w as f32;
    let scale_y = new_h as f32 / src_h as f32;

    (resize_frame(frame, new_w, new_h), scale_x, scale_y)
}

/// Maps boxes/landmarks from letterbox coordinates back to source-frame
/// pixels (per-axis scale division), in place. Shared by both detectors.
pub fn unletterbox(dets: &mut [(f32, [f32; 4], Kps)], scale_x: f32, scale_y: f32) {
    for det in dets {
        det.1 = [
            det.1[0] / scale_x,
            det.1[1] / scale_y,
            det.1[2] / scale_x,
            det.1[3] / scale_y,
        ];
        for k in det.2.iter_mut() {
            k[0] /= scale_x;
            k[1] /= scale_y;
        }
    }
}

/// Builds an ONNX session from embedded model bytes with the shared runtime
/// settings: Level3 graph optimization and `threads` intra-op threads.
pub fn load_session(model_bytes: &[u8], name: &str, threads: usize) -> Result<Session> {
    // ort's SessionBuilder returns `Error<SessionBuilder>`, which is not
    // `Send + Sync` and therefore cannot be `?`-converted into anyhow;
    // stringify those errors explicitly. `Session` itself is fine to
    // `?`/`.context`.
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("ort: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("ort: {e}"))?;
    let session = builder
        .commit_from_memory(model_bytes)
        .with_context(|| format!("failed to load {name} model"))?;

    anyhow::ensure!(
        session.inputs().len() == 1,
        "{name} model must have exactly one input"
    );
    Ok(session)
}

/// Detector input size used when `--input-size` is not given.
pub const DEFAULT_DETECT_SIZE: usize = 640;

/// Serializes the heavyweight ONNX-session tests across the crate: each loads
/// ~300 MiB of embedded models, so parallel `cargo test` threads would
/// multiply resident memory. Model-backed tests take this lock for the whole
/// test function.
#[cfg(test)]
pub(crate) static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Errors if the model is exported with a *fixed* input size that differs from
/// `size` (dynamic-shape models such as YuNet accept any multiple of 32; the
/// bundled SCRFD export may be fixed at 640). Inspects the session input's
/// tensor shape; dynamic dimensions are `-1` and are skipped.
pub fn ensure_fixed_input_fits(session: &Session, size: usize, what: &str) -> Result<()> {
    let Some(shape) = session.inputs()[0].dtype().tensor_shape() else {
        return Ok(());
    };
    let dims: Vec<i64> = shape.iter().copied().collect();
    if dims.len() == 4 {
        for &d in &dims[2..] {
            if d > 0 && d != size as i64 {
                anyhow::bail!(
                    "{what} input is fixed at {d}x{d}; --input-size {size} needs a \
                     dynamic-shape model"
                );
            }
        }
    }
    Ok(())
}

/// Greedy non-maximum suppression over score-sorted candidate faces, shared by
/// the SCRFD and YuNet detectors. IoU (with the `+1` area convention) above
/// `nms_thresh` suppresses the lower-scoring box. Keeps the (usually small)
/// post-threshold candidate list accurate.
pub fn non_max_suppression(
    mut dets: Vec<(f32, [f32; 4], Kps)>,
    nms_thresh: f32,
) -> Vec<DetectedFace> {
    dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = dets.len();
    if n == 0 {
        return Vec::new();
    }

    let mut keep = Vec::with_capacity(n);
    let mut order: Vec<usize> = (0..n).collect();

    while let Some(&i) = order.first() {
        keep.push(i);
        let order_slice = &order[1..];
        let (_, bi, _) = &dets[i];
        let area_i = (bi[2] - bi[0] + 1.0) * (bi[3] - bi[1] + 1.0);
        let mut survived = Vec::with_capacity(order_slice.len());
        for &j in order_slice {
            let (_, bj, _) = &dets[j];
            let xx1 = bi[0].max(bj[0]);
            let yy1 = bi[1].max(bj[1]);
            let xx2 = bi[2].min(bj[2]);
            let yy2 = bi[3].min(bj[3]);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let area_j = (bj[2] - bj[0] + 1.0) * (bj[3] - bj[1] + 1.0);
            let ovr = inter / (area_i + area_j - inter);
            if ovr <= nms_thresh {
                survived.push(j);
            }
        }
        order = survived;
    }

    keep.into_iter()
        .map(|i| {
            let (score, bbox, kps) = dets[i];
            DetectedFace { bbox, score, kps }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    fn det(score: f32, bbox: [f32; 4]) -> (f32, [f32; 4], Kps) {
        (score, bbox, [[0.0; 2]; 5])
    }

    #[test]
    fn rgb_constants_and_array_roundtrip() {
        assert_eq!(Rgb::RED.array(), [255, 64, 64]);
        assert_eq!(Rgb::GREEN.array(), [64, 255, 64]);
        assert_eq!(Rgb::WHITE.array(), [255, 255, 255]);
        assert_eq!(Rgb::BLACK.array(), [0, 0, 0]);
        assert_eq!(Rgb(Rgb::RED.array()), Rgb::RED);
    }

    #[test]
    fn arr3_to_rgb_and_back_roundtrip() {
        let frame = Array3U8::from_shape_fn((3, 4, 3), |(y, x, c)| (y * 4 + x + c) as u8);
        let img = arr3_to_rgb(&frame).expect("contiguous frame converts");
        let back = rgb_to_arr3(&img);
        assert_eq!(back.shape(), &[3, 4, 3]);
        assert_eq!(back, frame);
    }

    #[test]
    fn non_contiguous_views_report_no_slice() {
        // arr3_to_rgb hands off frame.as_slice(); a strided view (here, a
        // column subsample of the middle dim) has no contiguous slice, which
        // is the case the Option return guards against.
        let owned = Array3U8::from_shape_fn((4, 4, 3), |_| 7u8);
        let view = owned.slice(s![.., 0..2, ..]);
        assert!(view.as_slice().is_none());
    }

    #[test]
    fn fill_nchw_normalizes_top_left_and_zeros_the_rest() {
        let mut src = Array3U8::from_shape_fn((2, 2, 3), |_| 255u8);
        src[[0, 0, 1]] = 0;
        let mut blob = Array4::<f32>::zeros((1, 3, 4, 4));
        fill_nchw(&src, 127.5, &mut blob);

        // (255 - 127.5) / 127.5 == 1.0; (0 - 127.5) / 127.5 == -1.0.
        assert_eq!(blob[[0, 0, 0, 0]], 1.0);
        assert_eq!(blob[[0, 1, 0, 0]], -1.0);
        assert_eq!(blob[[0, 2, 0, 0]], 1.0);
        // Pixels past the 2x2 source stay zero.
        assert_eq!(blob[[0, 0, 2, 0]], 0.0);
        assert_eq!(blob[[0, 0, 0, 2]], 0.0);
        assert_eq!(blob[[0, 1, 3, 3]], 0.0);
    }

    #[test]
    fn resize_frame_keeps_solid_color_and_dims() {
        let frame = Array3U8::from_shape_fn((2, 2, 3), |(_, _, c)| if c == 0 { 255 } else { 0 });
        let out = resize_frame(&frame, 8, 4);
        assert_eq!(out.shape(), &[4, 8, 3]);
        assert!(out.iter().step_by(3).all(|&v| v == 255), "R channel stays red");
        assert!(out.iter().skip(1).step_by(3).all(|&v| v == 0));
        assert!(out.iter().skip(2).step_by(3).all(|&v| v == 0));
    }

    #[test]
    fn letterbox_portrait_fits_height() {
        // 40x20 (h x w), im_ratio 2.0 -> 50x100 at size 100. The letterbox
        // output is the aspect-fit resized frame itself (all content); the
        // zero padding only appears in the size x size detection blob.
        let frame = Array3U8::from_shape_fn((40, 20, 3), |(_, _, c)| if c == 0 { 200 } else { 0 });
        let (out, sx, sy) = letterbox(&frame, 100);
        assert_eq!(out.shape(), &[100, 50, 3]);
        assert!((sx - 2.5).abs() < 1e-6);
        assert!((sy - 2.5).abs() < 1e-6);
        // Full-height content, no padding inside the returned frame.
        assert_eq!(out[[10, 49, 0]], 200);
        assert_eq!(out[[99, 0, 0]], 200);

        // Placed top-left in a 100x100 blob the right half stays zero —
        // exactly how the detectors feed the network.
        let mut blob = Array4::<f32>::zeros((1, 3, 100, 100));
        fill_nchw(&out, 127.5, &mut blob);
        assert!(blob[[0, 0, 10, 49]] > 0.0, "content in the last filled column");
        assert_eq!(blob[[0, 0, 10, 50]], 0.0, "padding begins at col 50");
        assert_eq!(blob[[0, 0, 10, 99]], 0.0, "far right is padding");
    }

    #[test]
    fn letterbox_landscape_fits_width() {
        // 20x40 (h x w), im_ratio 0.5 -> 100x50 at size 100.
        let frame = Array3U8::from_shape_fn((20, 40, 3), |(_, _, c)| if c == 0 { 200 } else { 0 });
        let (out, sx, sy) = letterbox(&frame, 100);
        assert_eq!(out.shape(), &[50, 100, 3]);
        assert!((sx - 2.5).abs() < 1e-6);
        assert!((sy - 2.5).abs() < 1e-6);
        assert_eq!(out[[49, 10, 0]], 200);

        // Bottom rows of the blob stay zero.
        let mut blob = Array4::<f32>::zeros((1, 3, 100, 100));
        fill_nchw(&out, 127.5, &mut blob);
        assert!(blob[[0, 0, 49, 10]] > 0.0, "content in the last filled row");
        assert_eq!(blob[[0, 0, 50, 10]], 0.0, "padding begins at row 50");
        assert_eq!(blob[[0, 0, 99, 10]], 0.0, "far bottom is padding");
    }

    #[test]
    fn letterbox_degenerate_small_frames_stay_1px() {
        let one = Array3U8::from_shape_fn((1, 1, 3), |_| 1u8);
        let (out, ..) = letterbox(&one, 32);
        assert_eq!(out.shape(), &[32, 32, 3]);
    }

    #[test]
    fn unletterbox_maps_boxes_and_kps_back_to_source() {
        let mut dets = vec![(
            0.9,
            [10.0, 20.0, 30.0, 40.0],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        )];
        unletterbox(&mut dets, 2.0, 4.0);
        let (_, bbox, kps) = dets[0];
        assert_eq!(bbox, [5.0, 5.0, 15.0, 10.0]);
        assert_eq!(kps[0], [0.5, 0.5]);
        assert_eq!(kps[2], [2.5, 1.5]);
        assert_eq!(kps[4], [4.5, 2.5]);
    }

    #[test]
    fn nms_empty_input_returns_empty() {
        assert!(non_max_suppression(Vec::new(), 0.4).is_empty());
    }

    #[test]
    fn nms_keeps_a_single_candidate() {
        let out = non_max_suppression(vec![det(0.9, [0.0, 0.0, 10.0, 10.0])], 0.4);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].bbox, [0.0, 0.0, 10.0, 10.0]);
        assert_eq!(out[0].score, 0.9);
    }

    #[test]
    fn nms_suppresses_overlap_above_threshold() {
        let dets = vec![
            det(0.9, [0.0, 0.0, 10.0, 10.0]),
            det(0.8, [1.0, 1.0, 11.0, 11.0]), // IoU ~0.70 > 0.4
        ];
        let out = non_max_suppression(dets, 0.4);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].score, 0.9);
    }

    #[test]
    fn nms_keeps_disjoint_boxes() {
        let dets = vec![
            det(0.9, [0.0, 0.0, 10.0, 10.0]),
            det(0.8, [20.0, 0.0, 30.0, 10.0]),
        ];
        let out = non_max_suppression(dets, 0.4);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn nms_keeps_both_when_iou_below_threshold() {
        // 10px box and a 12px box whose intersection IoU is 0.25.
        let dets = vec![
            det(0.9, [0.0, 0.0, 10.0, 10.0]),
            det(0.8, [5.0, 0.0, 16.0, 11.0]),
        ];
        let out = non_max_suppression(dets, 0.4);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn nms_returns_scores_sorted_descending() {
        let dets = vec![
            det(0.5, [0.0, 0.0, 10.0, 10.0]),
            det(0.9, [20.0, 0.0, 30.0, 10.0]),
            det(0.7, [0.0, 20.0, 10.0, 30.0]),
        ];
        let out = non_max_suppression(dets, 0.4);
        let scores: Vec<f32> = out.iter().map(|d| d.score).collect();
        assert_eq!(scores, vec![0.9, 0.7, 0.5]);
    }

    #[test]
    fn nms_ties_keep_both_and_preserve_kps() {
        let mut dets = vec![
            det(0.9, [0.0, 0.0, 10.0, 10.0]),
            det(0.9, [20.0, 0.0, 30.0, 10.0]),
        ];
        dets[0].2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let out = non_max_suppression(dets, 0.4);
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|d| d.kps == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]));
    }
}