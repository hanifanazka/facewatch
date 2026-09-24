//! SFace recognition: 112x112 aligned BGR crop -> 128-d
//! L2-normalized embedding. Preprocessing mirrors OpenCV's
//! FaceRecognizerSF: BGR channel order, normalized (pixel - 127.5) / 127.5.

use anyhow::Result;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Models;

    #[test]
    fn input_size_is_square_112() {
        assert_eq!(INPUT_SIZE, 112);
    }

    #[test]
    fn load_rejects_garbage_bytes() {
        assert!(SFace::load(b"this is not an onnx model", 1).is_err());
        assert!(SFace::load(b"", 1).is_err());
    }

    #[test]
    fn real_model_embeds_are_128d_l2_normalized_and_deterministic() {
        let _g = crate::face::MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut sface = SFace::load(Models::embedded().sface, 1).expect("model loads");

        let crop = Array3U8::from_shape_fn((INPUT_SIZE, INPUT_SIZE, 3), |(y, x, _)| {
            ((y * 31 + x * 7) % 256) as u8
        });
        let a = sface.embed(&crop).expect("embed blank crop");
        assert_eq!(a.0.len(), 128);

        let norm: f32 = a.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "L2-normalized, got norm {norm}");

        let b = sface.embed(&crop).expect("embed again");
        let cos: f32 = a.0.iter().zip(&b.0).map(|(x, y)| x * y).sum();
        assert!(cos > 0.9999, "deterministic embedding, cosine {cos}");
    }

    #[test]
    fn real_model_channel_order_matters() {
        // SFace expects BGR; feeding the same pixels with R/B swapped must
        // produce a clearly different embedding. The crop is a smooth
        // vertical ramp with distinct per-channel offsets — low-frequency and
        // chromatic, so the swap is not a no-op (flat colored fields already
        // differ strongly: e.g. red vs blue cosine ~0.77 in probes).
        let _g = crate::face::MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut sface = SFace::load(Models::embedded().sface, 1).expect("model loads");

        let crop = Array3U8::from_shape_fn((INPUT_SIZE, INPUT_SIZE, 3), |(y, _, c)| {
            ((y * 2 + c * 40) % 256) as u8
        });
        let mut swapped = crop.clone();
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                swapped[[y, x, 0]] = crop[[y, x, 2]];
                swapped[[y, x, 2]] = crop[[y, x, 0]];
            }
        }
        let a = sface.embed(&crop).expect("canonical BGR input");
        let b = sface.embed(&swapped).expect("channel-swapped input");
        let cos: f32 = a.0.iter().zip(&b.0).map(|(x, y)| x * y).sum();
        assert!(cos < 0.99, "BGR vs RGB must differ, cosine {cos}");
        // Sanity: identical inputs still give identical embeddings (0.9999+).
        let again = sface.embed(&crop).expect("embed again");
        let cos2: f32 = a.0.iter().zip(&again.0).map(|(x, y)| x * y).sum();
        assert!(cos2 > 0.9999, "identical input is deterministic, cosine {cos2}");
    }

    #[test]
    fn real_model_rejects_off_shape_crop() {
        let _g = crate::face::MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut sface = SFace::load(Models::embedded().sface, 1).expect("model loads");
        let crop = Array3U8::zeros((112, 112, 2));
        let err = sface.embed(&crop).unwrap_err();
        assert!(err.to_string().contains("112x112x3"), "{err}");
    }
}