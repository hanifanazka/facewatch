//! SFace recognition: 112x112 aligned BGR crop -> 128-d
//! L2-normalized embedding. Preprocessing mirrors OpenCV's
//! FaceRecognizerSF: BGR channel order, normalized (pixel - 127.5) / 127.5.

use anyhow::{Context as _, Result};
use ndarray::{Array4, IxDyn};
use ort::session::Session;
use ort::value::Tensor;

use crate::face::{Array3U8, Embedding, load_session};

/// Input patch size for SFace (square 112x112).
pub const INPUT_SIZE: usize = 112;

/// A ready-to-run SFace recognition session.
pub struct SFace {
    session: Session,
    input_name: String,
}

impl SFace {
    /// Builds a recognition session from the embedded model bytes.
    pub fn load(model_bytes: &[u8], threads: usize) -> Result<Self> {
        let session = load_session(model_bytes, "SFace recognition", threads)?;

        anyhow::ensure!(
            session.outputs().len() == 1,
            "recognition model must have exactly one output"
        );

        let input_name = session.inputs()[0].name().to_owned();
        Ok(Self {
            session,
            input_name,
        })
    }

    /// Computes the L2-normalized embedding for an aligned `112x112` RGB crop.
    /// Converts RGB to BGR and normalizes (pixel - 127.5) / 127.5.
    pub fn embed(&mut self, crop: &Array3U8) -> Result<Embedding> {
        anyhow::ensure!(
            crop.shape() == [INPUT_SIZE, INPUT_SIZE, 3],
            "recognition crop must be 112x112x3 RGB, got {:?}",
            crop.shape()
        );

        let mut blob = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                // BGR order with (pixel - 127.5) / 127.5 normalization
                blob[[0, 0, y, x]] = (crop[[y, x, 2]] as f32 - 127.5) / 127.5;
                blob[[0, 1, y, x]] = (crop[[y, x, 1]] as f32 - 127.5) / 127.5;
                blob[[0, 2, y, x]] = (crop[[y, x, 0]] as f32 - 127.5) / 127.5;
            }
        }

        let tensor = Tensor::from_array(blob)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let view = outputs[0].try_extract_array::<f32>()?;

        // Flatten the (1, D) output and L2-normalize.
        let dim = view.len();
        let mut values = Vec::with_capacity(dim);
        for i in 0..dim {
            values.push(view.get(IxDyn(&[0, i])).copied().unwrap_or(0.0));
        }

        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut values {
                *v /= norm;
            }
        }

        Ok(Embedding(values))
    }
}