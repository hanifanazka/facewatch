//! YuNet face detector (OpenCV zoo `face_detection_yunet_2023mar.onnx`),
//! decoded to mirror OpenCV's `FaceDetectorYN` postprocess
//! (`modules/objdetect/src/face_detect.cpp`), which is the reference consumer
//! of this model.
//!
//! Contract (verified empirically against the OpenCV reference pipeline):
//! - one input `input`, fixed `(1, 3, 640, 640)` f32 NCHW, BGR channel order,
//!   raw pixel values in `[0, 255]` with no mean subtraction; the graph has no
//!   baked-in normalization, so the blob must be fed exactly like
//!   `cv::dnn::blobFromImage` defaults do;
//! - 12 outputs, three per stride `s in {8, 16, 32}`:
//!   `cls_s` / `obj_s` `(1, N, 1)`, `bbox_s` `(1, N, 4)`, `kps_s` `(1, N, 10)`,
//!   where `N = (640/s)^2` cells;
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
use ndarray::{Array4, IxDyn};
use ort::session::Session;
use ort::value::Tensor;

use crate::face::{Array3U8, DetectedFace, Kps, letterbox, load_session, non_max_suppression, unletterbox};

/// YuNet fixed input resolution (square).
pub const INPUT_SIZE: usize = 640;
/// Detection confidence threshold (FaceDetectorYN default).
pub const DET_THRESHOLD: f32 = 0.9;
/// NMS IoU threshold (FaceDetectorYN default).
pub const NMS_THRESHOLD: f32 = 0.3;

const STRIDES: [usize; 3] = [8, 16, 32];

/// Output positions of the 12 per-stride tensors, resolved by name once at
/// load time (the names are part of the 2023mar export contract).
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
    out: OutIdx,
    det_thresh: f32,
    nms_thresh: f32,
}

impl Yunet {
    /// Builds a detection session from the embedded model bytes.
    pub fn load(model_bytes: &[u8]) -> Result<Self> {
        let session = load_session(model_bytes, "YuNet")?;

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
            out,
            det_thresh: DET_THRESHOLD,
            nms_thresh: NMS_THRESHOLD,
        })
    }

    /// Runs detection on an RGB frame. Coordinates are in source-frame pixels.
    pub fn detect(&mut self, frame: &Array3U8) -> Result<Vec<DetectedFace>> {
        let (resized, scale_x, scale_y) = letterbox(frame, INPUT_SIZE);

        // BGR channels, raw [0, 255], no mean subtraction (mirrors
        // blobFromImage defaults) — fill channels swapped into NCHW.
        let mut blob = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
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

                let cells = INPUT_SIZE / stride;
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
                        if score < self.det_thresh {
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

                        // Identical offset+scale decode per landmark; order is
                        // kept as-is (see module docs: no remap needed).
                        let mut landmarks = [[0.0f32; 2]; 5];
                        for k in 0..5 {
                            let px =
                                (c as f32 + kps.get(IxDyn(&[0, cell, k * 2])).copied().unwrap_or(0.0))
                                    * stride_f;
                            let py =
                                (r as f32 + kps.get(IxDyn(&[0, cell, k * 2 + 1])).copied().unwrap_or(0.0))
                                    * stride_f;
                            landmarks[k] = [px, py];
                        }

                        scores_all.push((score, [x1, y1, x2, y2], landmarks));
                    }
                }
            }
            scores_all
        };

        // Map back to source coordinates (per-axis scales).
        unletterbox(&mut scores_all, scale_x, scale_y);

        Ok(non_max_suppression(scores_all, self.nms_thresh))
    }
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