//! Face gallery: named embeddings matched by cosine similarity.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::face::{Embedding, Match};

/// Recognition backend used for matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Recognizer {
    AuraFace,
    SFace,
}

/// One registered subject.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GalleryEntry {
    pub name: String,
    /// AuraFace L2-normalized embedding (512-d).
    #[serde(alias = "embedding", default)]
    pub auraface: Vec<f32>,
    /// SFace L2-normalized embedding (128-d).
    #[serde(default)]
    pub sface: Vec<f32>,
}

impl GalleryEntry {
    /// Returns the embedding for the given recognizer.
    pub fn embedding(&self, recognizer: Recognizer) -> &[f32] {
        match recognizer {
            Recognizer::AuraFace => &self.auraface,
            Recognizer::SFace => &self.sface,
        }
    }

    /// Checks if the entry has an embedding for the given recognizer.
    pub fn has_embedding(&self, recognizer: Recognizer) -> bool {
        !self.embedding(recognizer).is_empty()
    }
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

    /// Adds (or replaces) a subject by name with both embeddings.
    pub fn register(&mut self, name: &str, auraface: Embedding, sface: Embedding) {
        let entry = GalleryEntry {
            name: name.to_owned(),
            auraface: auraface.0,
            sface: sface.0,
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
    /// similarity is below `threshold`. Uses the specified recognizer's embedding space.
    pub fn match_faces(
        &self,
        embeddings: &[Embedding],
        threshold: f32,
        recognizer: Recognizer,
    ) -> Vec<Option<Match>> {
        embeddings
            .iter()
            .map(|emb| {
                let mut best: Option<(&str, f32)> = None;
                for entry in &self.entries {
                    if !entry.has_embedding(recognizer) {
                        continue;
                    }
                    let sim = Self::cosine(&emb.0, entry.embedding(recognizer));
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
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}