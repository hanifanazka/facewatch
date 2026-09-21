//! AuraFace (glintr100) recognition: 112x112 aligned RGB crop -> 512-d
//! L2-normalized embedding. Preprocessing mirrors insightface's
//! `ArcFaceONNX.get_feat` for a model without baked-in normalization:
//! `(pixel - 127.5) / 127.5` on the RGB channels (no channel swap needed).

use anyhow::{Context, Result};
use ndarray::{Array4, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::face::{Array3U8, Embedding};

/// Input patch size for AuraFace (square 112x112).
pub const INPUT_SIZE: usize = 112;

/// A ready-to-run AuraFace recognition session.
pub struct AuraFace {
    session: Session,
    input_name: String,
}

impl AuraFace {
    /// Builds a recognition session from the embedded model bytes.
    pub fn load(model_bytes: &[u8]) -> Result<Self> {
        // ort's SessionBuilder returns `Error<SessionBuilder>`, which is not
        // `Send + Sync` and therefore cannot be `?`-converted into anyhow;
        // stringify those errors explicitly. `Session` itself is fine to
        // `?`/`.context`.
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("ort: {e}"))?;
        let session = builder
            .commit_from_memory(model_bytes)
            .context("failed to load AuraFace model")?;

        anyhow::ensure!(
            session.inputs().len() == 1,
            "recognition model must have exactly one input"
        );
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
    pub fn embed(&mut self, crop: &Array3U8) -> Result<Embedding> {
        anyhow::ensure!(
            crop.shape() == [INPUT_SIZE, INPUT_SIZE, 3],
            "recognition crop must be 112x112x3 RGB, got {:?}",
            crop.shape()
        );

        let mut blob = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                for c in 0..3 {
                    let v = crop[[y, x, c]] as f32;
                    blob[[0, c, y, x]] = (v - 127.5) / 127.5;
                }
            }
        }

        let tensor = Tensor::from_array(blob)?;
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;
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