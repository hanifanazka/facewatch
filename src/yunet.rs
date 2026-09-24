//! YuNet face detector (OpenCV zoo `face_detection_yunet_2026may.onnx`), the
//! dynamic-shape re-export of the 2023mar model (same weights; outputs at 640
//! are byte-identical, verified by probe), decoded to mirror OpenCV's
//! `FaceDetectorYN` postprocess (`modules/objdetect/src/face_detect.cpp`),
//! which is the reference consumer of this model.
//!
//! Contract (verified empirically against the OpenCV reference pipeline):
//! - one input `input`, dynamic `(1, 3, H, W)` f32 NCHW (H = W =
//!   `--input-size`, default 640; the graph is fully convolutional, any
//!   multiple of 32 works), BGR channel order, raw pixel values in `[0, 255]`
//!   with no mean subtraction; the graph has no baked-in normalization, so the
//!   blob must be fed exactly like `cv::dnn::blobFromImage` defaults do;
//! - 12 outputs, three per stride `s in {8, 16, 32}`:
//!   `cls_s` / `obj_s` `(1, N, 1)`, `bbox_s` `(1, N, 4)`, `kps_s` `(1, N, 10)`,
//!   where `N = (S/s)^2` cells for input size `S`;
//! - per-cell decode: `score = sqrt(clamp(cls) * clamp(obj))`; box
//!   `cx = (c + tx) * s`, `cy = (r + ty) * s`, `w = exp(tw) * s`,
//!   `h = exp(th) * s`; landmarks the same `(offset + index) * s` form;
//! - landmark order `[right eye, left eye, nose, right mouth, left mouth]` is
//!   subject-relative but happens to equal the app's image-left-first
//!   `ARCFACE_DST` pairing (subject right == image left), so no remap is
//!   needed; the recognition gate (Azka ~0.735 on t1.png) verifies this.
//! - scores below 0.9 and NMS (IoU 0.3) suppress candidates, matching the
//!   `FaceDetectorYN` defaults.

use anyhow::{Context, Result};
use ndarray::{Array4, ArrayViewD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;

use crate::face::{
    Array3U8, DetectedFace, Kps, ensure_fixed_input_fits, letterbox, load_session,
    non_max_suppression, unletterbox,
};

/// Detection confidence threshold (FaceDetectorYN default).
pub const DET_THRESHOLD: f32 = 0.9;
/// NMS IoU threshold (FaceDetectorYN default).
pub const NMS_THRESHOLD: f32 = 0.3;

const STRIDES: [usize; 3] = [8, 16, 32];

/// Output positions of the 12 per-stride tensors, resolved by name once at
/// load time (the names are part of the YuNet export contract).
#[derive(Clone, Copy)]
struct OutIdx {
    cls: [usize; 3],
    obj: [usize; 3],
    bbox: [usize; 3],
    kps: [usize; 3],
}

/// A ready-to-run YuNet detection session.
pub struct Yunet {
    session: Session,
    input_name: String,
    input_size: usize,
    out: OutIdx,
    det_thresh: f32,
    nms_thresh: f32,
}

impl Yunet {
    /// Builds a detection session from the embedded model bytes.
    pub fn load(model_bytes: &[u8], threads: usize, input_size: usize) -> Result<Self> {
        let session = load_session(model_bytes, "YuNet", threads)?;
        ensure_fixed_input_fits(&session, input_size, "YuNet")?;

        anyhow::ensure!(
            session.outputs().len() == 12,
            "YuNet model must have 12 outputs (cls/obj/bbox/kps per stride), got {}",
            session.outputs().len()
        );

        // Resolve each output's position by its canonical name.
        let outputs = session.outputs();
        let pos_of = |name: &str| outputs.iter().position(|o| o.name() == name);

        let mut out = OutIdx {
            cls: [0; 3],
            obj: [0; 3],
            bbox: [0; 3],
            kps: [0; 3],
        };
        for (i, &stride) in STRIDES.iter().enumerate() {
            let name = |k: &str| format!("{k}_{stride}");
            out.cls[i] = pos_of(&name("cls")).with_context(|| {
                format!("YuNet output 'cls_{stride}' not found; unexpected export variant")
            })?;
            out.obj[i] = pos_of(&name("obj")).with_context(|| {
                format!("YuNet output 'obj_{stride}' not found; unexpected export variant")
            })?;
            out.bbox[i] = pos_of(&name("bbox")).with_context(|| {
                format!("YuNet output 'bbox_{stride}' not found; unexpected export variant")
            })?;
            out.kps[i] = pos_of(&name("kps")).with_context(|| {
                format!("YuNet output 'kps_{stride}' not found; unexpected export variant")
            })?;
        }

        let input_name = session.inputs()[0].name().to_owned();
        Ok(Self {
            session,
            input_name,
            input_size,
            out,
            det_thresh: DET_THRESHOLD,
            nms_thresh: NMS_THRESHOLD,
        })
    }

    /// Runs detection on an RGB frame. Coordinates are in source-frame pixels.
    pub fn detect(&mut self, frame: &Array3U8) -> Result<Vec<DetectedFace>> {
        let input_size = self.input_size;
        let (resized, scale_x, scale_y) = letterbox(frame, input_size);

        // BGR channels, raw [0, 255], no mean subtraction (mirrors
        // blobFromImage defaults) — fill channels swapped into NCHW.
        let mut blob = Array4::<f32>::zeros((1, 3, input_size, input_size));
        fill_nchw_bgr(&resized, &mut blob);

        let tensor = Tensor::from_array(blob)?;

        // Run + decode inside a scope: `outputs` borrows `self.session`
        // mutably, so it must be dropped before NMS below.
        let mut scores_all = {
            let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;

            let mut scores_all: Vec<(f32, [f32; 4], Kps)> = Vec::new();
            for (i, &stride) in STRIDES.iter().enumerate() {
                let cls = outputs[self.out.cls[i]].try_extract_array::<f32>()?;
                let obj = outputs[self.out.obj[i]].try_extract_array::<f32>()?;
                let bbox = outputs[self.out.bbox[i]].try_extract_array::<f32>()?;
                let kps = outputs[self.out.kps[i]].try_extract_array::<f32>()?;
                scores_all.extend(decode_yunet_stride(
                    input_size,
                    self.det_thresh,
                    stride,
                    &cls,
                    &obj,
                    &bbox,
                    &kps,
                ));
            }
            scores_all
        };

        // Map back to source coordinates (per-axis scales).
        unletterbox(&mut scores_all, scale_x, scale_y);

        Ok(non_max_suppression(scores_all, self.nms_thresh))
    }
}

/// Decodes the cls/obj/bbox/kps tensors of one YuNet stride into candidate
/// detections at letterbox scale: `score = sqrt(clamp(cls) * clamp(obj))`,
/// box `(c + t) * stride` with `exp`-width, landmarks the same offset+scale
/// form. Pure (no session), mirroring OpenCV `FaceDetectorYN` postprocess.
fn decode_yunet_stride(
    input_size: usize,
    det_thresh: f32,
    stride: usize,
    cls: &ArrayViewD<f32>,
    obj: &ArrayViewD<f32>,
    bbox: &ArrayViewD<f32>,
    kps: &ArrayViewD<f32>,
) -> Vec<(f32, [f32; 4], Kps)> {
    let cells = input_size / stride;
    let mut dets = Vec::new();
    for r in 0..cells {
        for c in 0..cells {
            let cell = r * cells + c;
            let cls_score = cls
                .get(IxDyn(&[0, cell, 0]))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let obj_score = obj
                .get(IxDyn(&[0, cell, 0]))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let score = (cls_score * obj_score).sqrt();
            if score < det_thresh {
                continue;
            }

            let stride_f = stride as f32;
            let tx = bbox.get(IxDyn(&[0, cell, 0])).copied().unwrap_or(0.0);
            let ty = bbox.get(IxDyn(&[0, cell, 1])).copied().unwrap_or(0.0);
            let tw = bbox.get(IxDyn(&[0, cell, 2])).copied().unwrap_or(0.0);
            let th = bbox.get(IxDyn(&[0, cell, 3])).copied().unwrap_or(0.0);

            let cx = (c as f32 + tx) * stride_f;
            let cy = (r as f32 + ty) * stride_f;
            let w = tw.exp() * stride_f;
            let h = th.exp() * stride_f;
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Identical offset+scale decode per landmark; order is kept
            // as-is (see module docs: no remap needed).
            let mut landmarks = [[0.0f32; 2]; 5];
            for k in 0..5 {
                let px = (c as f32 + kps.get(IxDyn(&[0, cell, k * 2])).copied().unwrap_or(0.0))
                    * stride_f;
                let py = (r as f32 + kps.get(IxDyn(&[0, cell, k * 2 + 1])).copied().unwrap_or(0.0))
                    * stride_f;
                landmarks[k] = [px, py];
            }

            dets.push((score, [x1, y1, x2, y2], landmarks));
        }
    }
    dets
}

/// Copies an RGB frame into the top-left of a zero-initialized `NCHW` float
/// blob in **BGR** order with raw `[0, 255]` values and no mean subtraction,
/// matching `cv::dnn::blobFromImage` defaults for the YuNet ONNX graph.
fn fill_nchw_bgr(frame: &Array3U8, out: &mut Array4<f32>) {
    let (height, width) = (frame.shape()[0], frame.shape()[1]);
    for y in 0..height {
        for x in 0..width {
            out[[0, 0, y, x]] = frame[[y, x, 2]] as f32;
            out[[0, 1, y, x]] = frame[[y, x, 1]] as f32;
            out[[0, 2, y, x]] = frame[[y, x, 0]] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Models;
    use ndarray::ArrayD;

    #[test]
    fn thresholds_match_facedetectoryn_defaults() {
        assert_eq!(DET_THRESHOLD, 0.9);
        assert_eq!(NMS_THRESHOLD, 0.3);
        assert_eq!(STRIDES, [8, 16, 32]);
    }

    #[test]
    fn fill_nchw_bgr_swaps_channels_without_normalizing() {
        // RGB pixel (r, g, b) must land in BGR order: channel 0 = b, 1 = g, 2 = r.
        let mut frame = Array3U8::zeros((1, 1, 3));
        frame[[0, 0, 0]] = 10; // r
        frame[[0, 0, 1]] = 20; // g
        frame[[0, 0, 2]] = 30; // b
        let mut out = Array4::<f32>::zeros((1, 3, 4, 4));
        fill_nchw_bgr(&frame, &mut out);
        assert_eq!(out[[0, 0, 0, 0]], 30.0, "channel 0 is B");
        assert_eq!(out[[0, 1, 0, 0]], 20.0, "channel 1 is G");
        assert_eq!(out[[0, 2, 0, 0]], 10.0, "channel 2 is R");
        // Raw [0,255] values: no mean subtraction, no scaling.
        assert_eq!(out[[0, 0, 0, 0]], 30.0);
        // Rest of the blob stays zero.
        assert_eq!(out[[0, 0, 1, 0]], 0.0);
    }

    #[test]
    fn load_rejects_garbage_bytes() {
        assert!(Yunet::load(b"this is not an onnx model", 1, 640).is_err());
    }

    /// Builds synthetic cls/obj/bbox/kps tensors for one stride, with a single
    /// candidate encoded at row `r`, column `c` of the stride grid.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn synthetic_outputs(
        input_size: usize,
        stride: usize,
        r: usize,
        c: usize,
        cls: f32,
        obj: f32,
        b: [f32; 4],
        k: [f32; 10],
    ) -> (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>, ArrayD<f32>) {
        let cells = input_size / stride;
        let cell = r * cells + c;
        let mut cls_a = ArrayD::<f32>::zeros(vec![1, cells * cells, 1]);
        let mut obj_a = ArrayD::<f32>::zeros(vec![1, cells * cells, 1]);
        let mut bbox_a = ArrayD::<f32>::zeros(vec![1, cells * cells, 4]);
        let mut kps_a = ArrayD::<f32>::zeros(vec![1, cells * cells, 10]);
        cls_a[[0, cell, 0]] = cls;
        obj_a[[0, cell, 0]] = obj;
        for (i, v) in b.iter().enumerate() {
            bbox_a[[0, cell, i]] = *v;
        }
        for (i, v) in k.iter().enumerate() {
            kps_a[[0, cell, i]] = *v;
        }
        (cls_a, obj_a, bbox_a, kps_a)
    }

    #[test]
    fn decode_maps_one_cell_to_expected_box_and_kps() {
        // input 64, stride 16 -> 4x4 grid; cell r=1, c=2 -> cell 6.
        // cls 0.81 * obj 1.0 -> score 0.9.
        let (cls, obj, bbox, kps) = synthetic_outputs(
            64,
            16,
            1,
            2,
            0.81,
            1.0,
            [0.5, -0.25, 0.0, f32::ln(2.0)],
            [0.25, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let out = decode_yunet_stride(64, 0.5, 16, &cls.view(), &obj.view(), &bbox.view(), &kps.view());
        assert_eq!(out.len(), 1);
        let (s, box_, landmarks) = out[0];
        assert!((s - 0.9).abs() < 1e-6, "sqrt(0.81*1.0) = 0.9, got {s}");
        // cx = (2+0.5)*16 = 40; cy = (1-0.25)*16 = 12; w = exp(0)*16 = 16; h = exp(ln2)*16 = 32.
        assert_eq!(box_, [32.0, -4.0, 48.0, 28.0]);
        // kps[0] = ((2 + 0.25)*16, (1 - 0.125)*16) = (36, 14).
        assert_eq!(landmarks[0], [36.0, 14.0]);
    }

    #[test]
    fn decode_clamps_scores_and_applies_threshold() {
        // cls/obj beyond [0,1] clamp to 1.0; 1.5 * 1.5 -> 1.0 -> score 1.0.
        let (cls, obj, bbox, kps) = synthetic_outputs(64, 16, 0, 0, 1.5, 1.5, [0.0; 4], [0.0; 10]);
        let out = decode_yunet_stride(64, 0.9, 16, &cls.view(), &obj.view(), &bbox.view(), &kps.view());
        assert_eq!(out[0].0, 1.0);

        // 0.85^2 sqrt = 0.85 < 0.9 -> skipped.
        let (cls, obj, bbox, kps) = synthetic_outputs(64, 16, 0, 0, 0.85, 0.85, [0.0; 4], [0.0; 10]);
        let out = decode_yunet_stride(64, 0.9, 16, &cls.view(), &obj.view(), &bbox.view(), &kps.view());
        assert!(out.is_empty());
    }

    #[test]
    fn real_model_loads_at_320_and_640_and_detects_without_panic() {
        let _g = crate::face::MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models = Models::embedded();
        for size in [320, 640] {
            let mut det = Yunet::load(models.yunet, 1, size).expect("yunet loads at dynamic size");
            let frame = Array3U8::from_shape_fn((360, size, 3), |(_, _, _)| 90u8);
            let faces = det.detect(&frame).expect("detect runs");
            for f in &faces {
                assert!(f.bbox.iter().all(|v| v.is_finite()), "finite box: {:?}", f.bbox);
                assert!(f.kps.iter().all(|p| p.iter().all(|v| v.is_finite())));
                assert!((0.0..=1.0).contains(&f.score));
            }
        }
    }
}