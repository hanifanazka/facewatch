//! Build-time model provisioning.
//!
//! Guarantees that `scrfd_10g_bnkps.onnx`, `face_detection_yunet_2023mar.onnx`,
//! and `glintr100.onnx` exist in `models/` **before** the crate compiles:
//! missing, truncated, or tampered files are downloaded and verified against
//! the exact SHA-256 published for each artifact. The verified files are then
//! embedded into the binary via `include_bytes!` in `src/models.rs`, so the
//! app performs no downloads or checksum logic at runtime.
//!
//! The models directory defaults to `<crate>/models` and can be overridden
//! with the `FACEWATCH_MODELS_DIR` environment variable.

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// Canonical model metadata; `src/models.rs` gets these as generated consts
/// (via `OUT_DIR/model_meta.rs`) so the embed name and the checksums used at
/// compile time can never drift from the ones verified by this script.
struct ModelSpec {
    prefix: &'static str,
    name: &'static str,
    url: &'static str,
    size: u64,
    sha256: &'static str,
}

const MODEL_SPECS: &[ModelSpec] = &[
    ModelSpec {
        prefix: "SCRFD",
        name: "scrfd_10g_bnkps.onnx",
        url: "https://huggingface.co/fal/AuraFace-v1/resolve/main/scrfd_10g_bnkps.onnx",
        size: 16_923_827,
        sha256: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    },
    // OpenCV zoo YuNet (2023mar), kept behind Git LFS in opencv_zoo, so the
    // raw bytes come from the media host rather than a raw-URL pointer file.
    ModelSpec {
        prefix: "YUNET",
        name: "face_detection_yunet_2023mar.onnx",
        url: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        size: 232_589,
        sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    ModelSpec {
        prefix: "AURAFACE",
        name: "glintr100.onnx",
        url: "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
        size: 260_694_151,
        sha256: "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
    },
];

fn main() -> Result<()> {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").context("CARGO_MANIFEST_DIR not set")?,
    );
    let models_dir = env::var_os("FACEWATCH_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("models"));

    // Make the resolved directory available to `include_bytes!` in the crate.
    println!("cargo:rustc-env=FACEWATCH_MODELS_DIR={}", models_dir.display());

    fs::create_dir_all(&models_dir)
        .with_context(|| format!("failed to create models dir {}", models_dir.display()))?;

    for spec in MODEL_SPECS {
        let path = models_dir.join(spec.name);
        // Re-verify whenever the file changes or disappears on disk.
        println!("cargo:rerun-if-changed={}", path.display());
        ensure_model(&path, spec)
            .with_context(|| format!("failed to prepare model {} for embedding", path.display()))?;
    }

    write_model_meta().context("failed to write model metadata for the crate")?;
    Ok(())
}

/// Ensures `path` holds exactly the published artifact: correct size *and*
/// SHA-256. Downloads from `url` otherwise, re-verifying the download before
/// it is accepted. Any mismatch aborts the build.
fn ensure_model(path: &Path, spec: &ModelSpec) -> Result<()> {
    if let Ok(meta) = fs::metadata(path) {
        let digest = sha256_of(path).with_context(|| format!("failed to hash {}", path.display()))?;
        if meta.len() == spec.size && digest == spec.sha256 {
            eprintln!(
                "facewatch: verified model {} ({} MiB)",
                path.display(),
                meta.len() / (1024 * 1024)
            );
            return Ok(());
        }
        eprintln!(
            "facewatch: model {} fails size/SHA-256 check (size {} / {} bytes, sha256 {digest}); re-downloading",
            path.display(),
            meta.len(),
            spec.size
        );
    }

    let tmp = path.with_extension("part");
    eprintln!(
        "facewatch: downloading {} ({} MiB) from {} ...",
        spec.name,
        spec.size / (1024 * 1024),
        spec.url
    );
    let response = ureq::get(spec.url)
        .call()
        .with_context(|| format!("HTTP request failed for {}", spec.url))?;
    let mut reader = response.into_body().into_reader();
    let mut out = File::create(&tmp)
        .with_context(|| format!("failed to create {}", tmp.display()))?;

    let mut downloaded: u64 = 0;
    let mut last_reported: u64 = 0;
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("read failed while downloading {}", spec.name))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .with_context(|| format!("write failed while downloading {}", spec.name))?;
        downloaded += n as u64;
        // Coarse progress report so the ~248 MiB download doesn't look hung.
        let mb = downloaded / (64 * 1024 * 1024);
        if mb != last_reported {
            last_reported = mb;
            eprintln!("  {} MiB ...", downloaded / (1024 * 1024));
        }
    }
    out.flush().ok();
    drop(out);

    if downloaded != spec.size {
        fs::remove_file(&tmp).ok();
        anyhow::bail!(
            "download of {} truncated: got {downloaded} bytes, expected {}",
            spec.name,
            spec.size
        );
    }

    // Never accept a download that does not match the published artifact.
    let digest = sha256_of(&tmp)?;
    if digest != spec.sha256 {
        fs::remove_file(&tmp).ok();
        anyhow::bail!(
            "download of {} failed SHA-256 verification\n  got  {digest}\n  want {}",
            spec.name,
            spec.sha256
        );
    }

    fs::rename(&tmp, path)
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    eprintln!("facewatch: saved model {}", path.display());
    Ok(())
}

/// Writes `$OUT_DIR/model_meta.rs` with metadata consts and the embedded
/// byte slices (`include_bytes!`) for each model, all generated from the
/// canonical `MODEL_SPECS` list so the embed and the verified checksums can
/// never drift.
fn write_model_meta() -> Result<()> {
    let out_dir = env::var_os("OUT_DIR").context("OUT_DIR not set")?;
    let dest = Path::new(&out_dir).join("model_meta.rs");
    let mut s = String::from("// Generated by build.rs; DO NOT EDIT.\n");
    for spec in MODEL_SPECS {
        s.push_str(&format!(
            "#[allow(dead_code)] pub const {}_FILE: &str = {:?};\n",
            spec.prefix, spec.name
        ));
        s.push_str(&format!(
            "#[allow(dead_code)] pub const {}_URL: &str = {:?};\n",
            spec.prefix, spec.url
        ));
        s.push_str(&format!(
            "#[allow(dead_code)] pub const {}_SIZE: usize = {};\n",
            spec.prefix, spec.size
        ));
        s.push_str(&format!(
            "#[allow(dead_code)] pub const {}_SHA256: &str = {:?};\n",
            spec.prefix, spec.sha256
        ));
        s.push_str(&format!(
            "pub static {}_BYTES: &[u8] = include_bytes!(concat!(env!(\"FACEWATCH_MODELS_DIR\"), \"/\", {:?}));\n",
            spec.prefix, spec.name
        ));
    }
    fs::write(&dest, s).with_context(|| format!("failed to write {}", dest.display()))?;
    Ok(())
}

/// Hex SHA-256 of a file, computed by streaming.
fn sha256_of(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}