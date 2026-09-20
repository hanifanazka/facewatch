//! ONNX model discovery and download.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// SCRFD detection model (from fal/AuraFace-v1).
pub const SCRFD_URL: &str =
    "https://huggingface.co/fal/AuraFace-v1/resolve/main/scrfd_10g_bnkps.onnx";
/// AuraFace recognition model.
pub const AURAFACE_URL: &str =
    "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx";

/// Expected file sizes (bytes); the download is re-attempted when a file is
/// missing or truncated.
pub const SCRFD_SIZE: u64 = 16_923_827;
pub const AURAFACE_SIZE: u64 = 260_694_151;

/// SHA-256 of the exact files fal publishes under the apache-2.0 AuraFace
/// card (verified against the HF LFS blob IDs). Every local file is checked
/// against these before use.
pub const SCRFD_SHA256: &str = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91";
pub const AURAFACE_SHA256: &str =
    "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60";

/// Resolved paths to the two ONNX models.
#[derive(Clone, Debug)]
pub struct Models {
    pub scrfd: PathBuf,
    pub auraface: PathBuf,
}

/// Locates (downloading when absent) the two models inside `dir`.
pub fn ensure_models(dir: &Path) -> Result<Models> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create models dir {}", dir.display()))?;

    let scrfd = ensure_file(dir, "scrfd_10g_bnkps.onnx", SCRFD_URL, SCRFD_SIZE, SCRFD_SHA256)
        .with_context(|| format!("failed to prepare SCRFD model in {}", dir.display()))?;
    let auraface = ensure_file(
        dir,
        "glintr100.onnx",
        AURAFACE_URL,
        AURAFACE_SIZE,
        AURAFACE_SHA256,
    )
    .with_context(|| format!("failed to prepare AuraFace model in {}", dir.display()))?;

    Ok(Models { scrfd, auraface })
}

/// Ensures `name` exists in `dir` with the expected content (size *and*
/// SHA-256), downloading from `url` otherwise.
fn ensure_file(
    dir: &Path,
    name: &str,
    url: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<PathBuf> {
    let path = dir.join(name);
    if let Ok(meta) = std::fs::metadata(&path) {
        let digest = sha256_of(&path)?;
        if meta.len() >= expected_size && digest == expected_sha256 {
            eprintln!(
                "using model: {} ({} MiB)",
                path.display(),
                meta.len() / (1024 * 1024)
            );
            return Ok(path);
        }
        eprintln!(
            "model {} fails the size/SHA-256 check (sha256 {digest}); re-downloading",
            path.display()
        );
    }

    let tmp = dir.join(format!("{name}.part"));
    eprintln!(
        "downloading {name} ({} MiB) from {url} ...",
        expected_size / (1024 * 1024)
    );
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP request failed for {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut out = std::fs::File::create(&tmp)
        .with_context(|| format!("failed to create {}", tmp.display()))?;

    let mut downloaded: u64 = 0;
    let mut last_reported: u64 = 0;
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("read failed while downloading {name}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .with_context(|| format!("write failed while downloading {name}"))?;
        downloaded += n as u64;
        // Coarse progress report so long downloads don't look hung.
        let mb = downloaded / (64 * 1024 * 1024);
        if mb != last_reported {
            last_reported = mb;
            eprintln!("  {} MiB ...", downloaded / (1024 * 1024));
        }
    }
    out.flush().ok();

    if downloaded < expected_size {
        anyhow::bail!(
            "download of {name} truncated: got {downloaded} bytes, expected ~{expected_size}"
        );
    }

    // Never accept a download that does not match the published artifact.
    let digest = sha256_of(&tmp)?;
    if digest != expected_sha256 {
        std::fs::remove_file(&tmp).ok();
        anyhow::bail!(
            "download of {name} failed SHA-256 verification\n  got  {digest}\n  want {expected_sha256}"
        );
    }

    std::fs::rename(&tmp, &path).with_context(|| {
        format!("failed to finalize {}", path.display())
    })?;
    eprintln!("saved model: {}", path.display());
    Ok(path)
}

/// Hex SHA-256 of a file, computed by streaming.
fn sha256_of(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
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