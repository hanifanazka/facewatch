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
/// settings: Level3 graph optimization and 2 intra-op threads.
pub fn load_session(model_bytes: &[u8], name: &str) -> Result<Session> {
    // ort's SessionBuilder returns `Error<SessionBuilder>`, which is not
    // `Send + Sync` and therefore cannot be `?`-converted into anyhow;
    // stringify those errors explicitly. `Session` itself is fine to
    // `?`/`.context`.
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("ort: {e}"))?
        .with_intra_threads(2)
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