//! SCRFD face detector, decoding mirrored from insightface's `scrfd.py`
//! (2 anchors per cell, strides 8/16/32, top-left zero-padded letterbox).

use anyhow::Result;
use ndarray::{Array4, ArrayViewD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;

use crate::face::{
    Array3U8, DetectedFace, Kps, ensure_fixed_input_fits, fill_nchw, letterbox, load_session,
    non_max_suppression, unletterbox,
};

/// Detection confidence threshold, matching insightface defaults.
pub const DET_THRESHOLD: f32 = 0.5;
/// NMS IoU threshold, matching insightface defaults.
pub const NMS_THRESHOLD: f32 = 0.4;

const STRIDES: [usize; 3] = [8, 16, 32];
const NUM_ANCHORS: usize = 2;

/// A ready-to-run SCRFD detection session.
pub struct Scrfd {
    session: Session,
    input_name: String,
    input_size: usize,
    det_thresh: f32,
    nms_thresh: f32,
}

impl Scrfd {
    /// Builds a detection session from the embedded model bytes.
    pub fn load(model_bytes: &[u8], threads: usize, input_size: usize) -> Result<Self> {
        let session = load_session(model_bytes, "SCRFD", threads)?;
        ensure_fixed_input_fits(&session, input_size, "SCRFD")?;

        anyhow::ensure!(session.outputs().len() == 9, "SCRFD model must have 9 outputs (score/bbox/kps)");

        let input_name = session.inputs()[0].name().to_owned();
        Ok(Self {
            session,
            input_name,
            input_size,
            det_thresh: DET_THRESHOLD,
            nms_thresh: NMS_THRESHOLD,
        })
    }

    /// Runs detection on an RGB frame. Coordinates are in source-frame pixels.
    pub fn detect(&mut self, frame: &Array3U8) -> Result<Vec<DetectedFace>> {
        let input_size = self.input_size;
        let (resized, scale_x, scale_y) = letterbox(frame, input_size);

        // Build the NCHW float blob: (x - 127.5) / 128.0, RGB already.
        let mut blob = Array4::<f32>::zeros((1, 3, input_size, input_size));
        fill_nchw(&resized, 128.0, &mut blob);

        let tensor = Tensor::from_array(blob)?;

        // Run + decode inside a scope: `outputs` borrows `self.session`
        // mutably, so it must be dropped before we call `self.nms` below.
        let mut scores_all = {
            let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;

            let mut scores_all: Vec<(f32, [f32; 4], Kps)> = Vec::new();
            for (idx, stride) in STRIDES.iter().enumerate() {
                let score_view = outputs[idx].try_extract_array::<f32>()?;
                let bbox_view = outputs[idx + 3].try_extract_array::<f32>()?;
                let kps_view = outputs[idx + 6].try_extract_array::<f32>()?;
                scores_all.extend(decode_scrfd_stride(
                    input_size,
                    self.det_thresh,
                    *stride,
                    &score_view,
                    &bbox_view,
                    &kps_view,
                ));
            }
            scores_all
        };

        // Map back to source coordinates (per-axis scales).
        unletterbox(&mut scores_all, scale_x, scale_y);

        Ok(non_max_suppression(scores_all, self.nms_thresh))
    }
}

/// Decodes the score/bbox/kps tensors of one SCRFD stride into candidate
/// detections at letterbox scale, keeping only scores at/above `det_thresh`.
/// Pure (no session), so the decode math is unit-testable with synthetic
/// tensors. Mirrors insightface's `scrfd.py` per-anchor decode.
fn decode_scrfd_stride(
    input_size: usize,
    det_thresh: f32,
    stride: usize,
    score: &ArrayViewD<f32>,
    bbox: &ArrayViewD<f32>,
    kps: &ArrayViewD<f32>,
) -> Vec<(f32, [f32; 4], Kps)> {
    let height = input_size / stride;
    let width = input_size / stride;
    let count = height * width * NUM_ANCHORS;

    let mut dets = Vec::new();
    for a in 0..count {
        // Scores are rank-2 `[count, 1]`: index `[a, 0]`.
        let s = score
            .get(IxDyn(&[a, 0]))
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        if s < det_thresh {
            continue;
        }

        // Anchor center: (col * stride, row * stride), duplicated per anchor.
        let anchor = a / NUM_ANCHORS;
        let row = anchor / width;
        let col = anchor % width;
        let (cx, cy) = ((col * stride) as f32, (row * stride) as f32);

        let x1 = cx - bbox.get(IxDyn(&[a, 0])).copied().unwrap_or(0.0) * stride as f32;
        let y1 = cy - bbox.get(IxDyn(&[a, 1])).copied().unwrap_or(0.0) * stride as f32;
        let x2 = cx + bbox.get(IxDyn(&[a, 2])).copied().unwrap_or(0.0) * stride as f32;
        let y2 = cy + bbox.get(IxDyn(&[a, 3])).copied().unwrap_or(0.0) * stride as f32;

        let mut kps_out = [[0.0f32; 2]; 5];
        for k in 0..5 {
            let px = cx + kps.get(IxDyn(&[a, k * 2])).copied().unwrap_or(0.0) * stride as f32;
            let py = cy + kps.get(IxDyn(&[a, k * 2 + 1])).copied().unwrap_or(0.0) * stride as f32;
            kps_out[k] = [px, py];
        }

        dets.push((s, [x1, y1, x2, y2], kps_out));
    }
    dets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Models;
    use ndarray::ArrayD;

    #[test]
    fn thresholds_match_insightface_defaults() {
        assert_eq!(DET_THRESHOLD, 0.5);
        assert_eq!(NMS_THRESHOLD, 0.4);
        assert_eq!(STRIDES, [8, 16, 32]);
        assert_eq!(NUM_ANCHORS, 2);
    }

    #[test]
    fn load_rejects_garbage_bytes() {
        assert!(Scrfd::load(b"this is not an onnx model", 1, 640).is_err());
    }

    /// Builds (score, bbox, kps) synthetic tensors for one stride of a given
    /// input size, with a single candidate encoded at anchor index `a`.
    #[allow(clippy::type_complexity)]
    fn synthetic_outputs(
        input_size: usize,
        stride: usize,
        a: usize,
        s: f32,
        b: [f32; 4],
        k: [f32; 10],
    ) -> (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>) {
        let count = (input_size / stride) * (input_size / stride) * NUM_ANCHORS;
        let mut score = ArrayD::<f32>::zeros(vec![count, 1]);
        let mut bbox = ArrayD::<f32>::zeros(vec![count, 4]);
        let mut kps = ArrayD::<f32>::zeros(vec![count, 10]);
        score[[a, 0]] = s;
        for (i, v) in b.iter().enumerate() {
            bbox[[a, i]] = *v;
        }
        for (i, v) in k.iter().enumerate() {
            kps[[a, i]] = *v;
        }
        (score, bbox, kps)
    }

    #[test]
    fn decode_maps_one_anchor_to_expected_box_and_kps() {
        // input 64, stride 8 -> 8x8 grid; anchor 7 -> anchor 3 -> row 0, col 3.
        // Anchor center = (24, 0).
        let (score, bbox, kps) = synthetic_outputs(
            64,
            8,
            7,
            0.9,
            [2.0, 1.0, 3.0, 1.5],
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let out = decode_scrfd_stride(64, 0.5, 8, &score.view(), &bbox.view(), &kps.view());
        assert_eq!(out.len(), 1);
        let (s, box_, kps_out) = out[0];
        assert_eq!(s, 0.9);
        // x1 = 24 - 2*8 = 8; y1 = 0 - 1*8 = -8; x2 = 24 + 3*8 = 48; y2 = 0 + 1.5*8 = 12.
        assert_eq!(box_, [8.0, -8.0, 48.0, 12.0]);
        // kps[0] = (24 + 1*8, 0 + 0.5*8) = (32, 4).
        assert_eq!(kps_out[0], [32.0, 4.0]);
    }

    #[test]
    fn decode_skips_candidates_below_threshold() {
        let (score, bbox, kps) = synthetic_outputs(64, 8, 0, 0.4, [0.0; 4], [0.0; 10]);
        let out = decode_scrfd_stride(64, 0.5, 8, &score.view(), &bbox.view(), &kps.view());
        assert!(out.is_empty(), "score 0.4 < 0.5 is skipped");
    }

    #[test]
    fn decode_clamps_negative_scores_below_threshold() {
        // A negative reported score is clamped to 0 and therefore skipped at
        // the default 0.5 confidence threshold (never produces a detection).
        let (score, bbox, kps) = synthetic_outputs(64, 8, 0, -0.1, [0.0; 4], [0.0; 10]);
        let out = decode_scrfd_stride(64, 0.5, 8, &score.view(), &bbox.view(), &kps.view());
        assert!(out.is_empty(), "clamped score falls below the threshold");
    }

    #[test]
    fn decode_handles_second_anchor_of_a_cell() {
        // a = 1 -> anchor 0 -> row 0, col 0; center (0,0). Box fully offsets.
        let (score, bbox, kps) = synthetic_outputs(64, 8, 1, 0.9, [1.0, 2.0, 3.0, 4.0], [0.0; 10]);
        let out = decode_scrfd_stride(64, 0.5, 8, &score.view(), &bbox.view(), &kps.view());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, [-8.0, -16.0, 24.0, 32.0]);
    }

    #[test]
    fn real_model_loads_at_320_and_640_and_detects_without_panic() {
        // The bundled SCRFD export declares a fully dynamic input
        // ([1, 3, -1, -1]), so ensure_fixed_input_fits passes and both sizes
        // load. (The "SCRFD input is fixed at 640x640" message in
        // ensure_fixed_input_fits only fires for fixed-shape exports.)
        let _g = crate::face::MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for size in [320usize, 640usize] {
            let mut det =
                Scrfd::load(Models::embedded().scrfd, 1, size).unwrap_or_else(|e| panic!("scrfd loads at {size}: {e}"));
            let frame = Array3U8::from_shape_fn((size / 2, size, 3), |_| 100u8);
            let faces = det.detect(&frame).expect("detect runs");
            for f in &faces {
                assert!(f.bbox.iter().all(|v| v.is_finite()), "finite box: {:?}", f.bbox);
                assert!(f.kps.iter().all(|p| p.iter().all(|v| v.is_finite())));
                assert!((0.0..=1.0).contains(&f.score));
            }
        }
    }
}