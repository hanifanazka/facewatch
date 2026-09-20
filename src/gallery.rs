//! Face gallery: named embeddings matched by cosine similarity.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::face::{Embedding, Match};

/// One registered subject.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GalleryEntry {
    pub name: String,
    /// L2-normalized embedding.
    pub embedding: Vec<f32>,
}

/// The set of known subjects.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Gallery {
    pub entries: Vec<GalleryEntry>,
}

impl Gallery {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read gallery {}", path.display()))?;
        let gallery = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse gallery {}", path.display()))?;
        Ok(gallery)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, bytes)
            .with_context(|| format!("failed to write gallery {}", path.display()))?;
        Ok(())
    }

    /// Adds (or replaces) a subject by name.
    pub fn register(&mut self, name: &str, embedding: Embedding) {
        let entry = GalleryEntry {
            name: name.to_owned(),
            embedding: embedding.0,
        };
        if let Some(existing) = self.entries.iter_mut().find(|e| e.name == name) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Computes cosine similarity between two embeddings.
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            dot += x * y;
        }
        dot
    }

    /// Returns the best gallery match per embedding, or `None` when the best
    /// similarity is below `threshold`.
    pub fn match_faces(
        &self,
        embeddings: &[Embedding],
        threshold: f32,
    ) -> Vec<Option<Match>> {
        embeddings
            .iter()
            .map(|emb| {
                let mut best: Option<(&str, f32)> = None;
                for entry in &self.entries {
                    let sim = Self::cosine(&emb.0, &entry.embedding);
                    if best.map(|(_, s)| sim > s).unwrap_or(true) {
                        best = Some((entry.name.as_str(), sim));
                    }
                }
                best.and_then(|(name, score)| {
                    if score >= threshold {
                        Some(Match {
                            name: name.to_owned(),
                            score,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Short human-readable summary of a gallery.
    pub fn summary(&self) -> String {
        if self.entries.is_empty() {
            "(empty)".to_owned()
        } else {
            self.entries
                .iter()
                .map(|e| e.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

/// Loads a gallery, falling back to per-name direct entries if the path points
/// at an older flat map format.
pub fn load_gallery_any(path: &Path) -> Result<Gallery> {
    if !path.exists() {
        return Ok(Gallery::default());
    }
    match Gallery::load(path) {
        Ok(g) => Ok(g),
        Err(first) => {
            // Try legacy `{"name": [embedding...]}` map format.
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read gallery {}", path.display()))?;
            let map: HashMap<String, Vec<f32>> =
                serde_json::from_str(&raw).with_context(|| {
                    format!("gallery {} is neither list nor map format", path.display())
                })?;
            let _ = first;
            Ok(Gallery {
                entries: map
                    .into_iter()
                    .map(|(name, embedding)| GalleryEntry { name, embedding })
                    .collect(),
            })
        }
    }
}