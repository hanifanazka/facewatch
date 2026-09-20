//! SCRFD face detector, decoding mirrored from insightface's `scrfd.py`
//! (2 anchors per cell, strides 8/16/32, top-left zero-padded letterbox).

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array4, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::face::{Array3U8, DetectedFace, Kps, resize_frame};

/// SCRFD input resolution (width, height).
pub const INPUT_SIZE: usize = 640;
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
    det_thresh: f32,
    nms_thresh: f32,
}

impl Scrfd {
    /// Loads a SCRFD ONNX model.
    pub fn load(path: &Path) -> Result<Self> {
        // See aura.rs: ort's SessionBuilder returns `Error<SessionBuilder>`,
        // which is not `Send + Sync`, so stringify those errors explicitly.
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("ort: {e}"))?;
        let session = builder
            .commit_from_file(path)
            .context("failed to load SCRFD model")?;

        anyhow::ensure!(session.inputs().len() == 1, "SCRFD model must have exactly one input");
        anyhow::ensure!(session.outputs().len() == 9, "SCRFD model must have 9 outputs (score/bbox/kps)");

        let input_name = session.inputs()[0].name().to_owned();
        Ok(Self {
            session,
            input_name,
            det_thresh: DET_THRESHOLD,
            nms_thresh: NMS_THRESHOLD,
        })
    }

    /// Runs detection on an RGB frame. Coordinates are in source-frame pixels.
    pub fn detect(&mut self, frame: &Array3U8) -> Result<Vec<DetectedFace>> {
        let src_h = frame.shape()[0];
        let src_w = frame.shape()[1];

        // Letterbox resize: fit the src aspect ratio inside 640x640, top-left.
        let im_ratio = src_h as f32 / src_w as f32;
        // model_ratio = 640/640 = 1.0; when the image is taller than wide we
        // fix the height, otherwise we fix the width.
        let (new_w, new_h) = if im_ratio > 1.0 {
            (INPUT_SIZE, (INPUT_SIZE as f32 / im_ratio).floor().max(1.0) as usize)
        } else {
            ((INPUT_SIZE as f32 * im_ratio).floor().max(1.0) as usize, INPUT_SIZE)
        };
        let det_scale = new_h as f32 / src_h as f32;

        let resized = resize_frame(frame, new_w, new_h);

        // Build the NCHW float blob: (x - 127.5) / 128.0, RGB already.
        let mut blob = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized[[y, x, 0]];
                let g = resized[[y, x, 1]];
                let b = resized[[y, x, 2]];
                blob[[0, 0, y, x]] = (p as f32 - 127.5) / 128.0;
                blob[[0, 1, y, x]] = (g as f32 - 127.5) / 128.0;
                blob[[0, 2, y, x]] = (b as f32 - 127.5) / 128.0;
            }
        }

        let tensor = Tensor::from_array(blob)?;

        // Run + decode inside a scope: `outputs` borrows `self.session`
        // mutably, so it must be dropped before we call `self.nms` below.
        let scores_all = {
            let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;

            // Decode.
            let mut scores_all: Vec<(f32, [f32; 4], Kps)> = Vec::new();
            for (idx, stride) in STRIDES.iter().enumerate() {
                let score_view = outputs[idx].try_extract_array::<f32>()?;
                let bbox_view = outputs[idx + 3].try_extract_array::<f32>()?;
                let kps_view = outputs[idx + 6].try_extract_array::<f32>()?;

                let height = INPUT_SIZE / stride;
                let width = INPUT_SIZE / stride;
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

        // Map back to source coordinates.
        let mut scores_all = scores_all;
        for det in &mut scores_all {
            det.1 = [
                det.1[0] / det_scale,
                det.1[1] / det_scale,
                det.1[2] / det_scale,
                det.1[3] / det_scale,
            ];
            for k in det.2.iter_mut() {
                k[0] /= det_scale;
                k[1] /= det_scale;
            }
        }

        Ok(self.nms(scores_all))
    }

    /// Standard NMS over the score-sorted candidates (stays accurate for the
    /// (usually small) number of post-threshold boxes).
    fn nms(&self, mut dets: Vec<(f32, [f32; 4], Kps)>) -> Vec<DetectedFace> {
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
                if ovr <= self.nms_thresh {
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
}