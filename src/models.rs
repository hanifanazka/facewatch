//! ONNX models, embedded in the binary at compile time.
//!
//! `build.rs` guarantees the files in the models directory (default
//! `<crate>/models`, override with `FACEWATCH_MODELS_DIR`) exist and pass the
//! SHA-256 check *before* compilation, then generates `model_meta.rs` below —
//! metadata consts plus the `include_bytes!` byte slices — so the embedded
//! bytes are always the verified artifacts. No download or checksum logic
//! runs at runtime.

include!(concat!(env!("OUT_DIR"), "/model_meta.rs"));

/// The two embedded models as byte slices.
#[derive(Clone, Copy, Debug)]
pub struct Models {
    pub scrfd: &'static [u8],
    pub auraface: &'static [u8],
}

impl Models {
    /// Static models embedded by `build.rs` (compile-time download + SHA-256
    /// verification before the embed happens).
    pub fn embedded() -> Models {
        Models {
            scrfd: SCRFD_BYTES,
            auraface: AURAFACE_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_models_have_the_verified_sizes() {
        // build.rs already hash-verified the exact bytes that were embedded;
        // this guards the generated include/const plumbing at dev time.
        assert_eq!(SCRFD_BYTES.len(), SCRFD_SIZE);
        assert_eq!(AURAFACE_BYTES.len(), AURAFACE_SIZE);
    }
}