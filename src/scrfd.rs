//! SCRFD face detector, decoding mirrored from insightface's `scrfd.py`
//! (2 anchors per cell, strides 8/16/32, top-left zero-padded letterbox).

use anyhow::Result;
use ndarray::{Array4, IxDyn};
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

            // Decode.
            let mut scores_all: Vec<(f32, [f32; 4], Kps)> = Vec::new();
            for (idx, stride) in STRIDES.iter().enumerate() {
                let score_view = outputs[idx].try_extract_array::<f32>()?;
                let bbox_view = outputs[idx + 3].try_extract_array::<f32>()?;
                let kps_view = outputs[idx + 6].try_extract_array::<f32>()?;

                let height = input_size / stride;
                let width = input_size / stride;
                let count = height * width * NUM_ANCHORS;

                for a in 0..count {
                    // Scores are rank-2 `[count, 1]`: index `[a, 0]`.
                    let s = score_view
                        .get(IxDyn(&[a, 0]))
                        .copied()
                        .unwrap_or(0.0)
                        .max(0.0);
                    if s < self.det_thresh {
                        continue;
                    }

                    // Anchor center: (col * stride, row * stride), duplicated per anchor.
                    let anchor = a / NUM_ANCHORS;
                    let row = anchor / width;
                    let col = anchor % width;
                    let (cx, cy) = ((col * stride) as f32, (row * stride) as f32);

                    let x1 = cx - bbox_view.get(IxDyn(&[a, 0])).copied().unwrap_or(0.0) * *stride as f32;
                    let y1 = cy - bbox_view.get(IxDyn(&[a, 1])).copied().unwrap_or(0.0) * *stride as f32;
                    let x2 = cx + bbox_view.get(IxDyn(&[a, 2])).copied().unwrap_or(0.0) * *stride as f32;
                    let y2 = cy + bbox_view.get(IxDyn(&[a, 3])).copied().unwrap_or(0.0) * *stride as f32;

                    let mut kps = [[0.0f32; 2]; 5];
                    for k in 0..5 {
                        let px = cx + kps_view.get(IxDyn(&[a, k * 2])).copied().unwrap_or(0.0) * *stride as f32;
                        let py = cy + kps_view.get(IxDyn(&[a, k * 2 + 1])).copied().unwrap_or(0.0) * *stride as f32;
                        kps[k] = [px, py];
                    }

                    scores_all.push((
                        s,
                        [x1, y1, x2, y2],
                        kps,
                    ));
                }
            }
            scores_all
        };

        // Map back to source coordinates (per-axis scales).
        unletterbox(&mut scores_all, scale_x, scale_y);

        Ok(non_max_suppression(scores_all, self.nms_thresh))
    }
}