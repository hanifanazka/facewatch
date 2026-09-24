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

#[cfg(test)]
mod tests {
    use super::*;

    fn e(v: Vec<f32>) -> Embedding {
        Embedding(v)
    }

    #[test]
    fn register_stores_both_embeddings() {
        let mut g = Gallery::default();
        g.register("A", e(vec![1.0, 2.0, 3.0]), e(vec![4.0, 5.0, 6.0]));

        let entry = &g.entries[0];
        assert_eq!(entry.auraface, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.sface, vec![4.0, 5.0, 6.0]);
        assert_eq!(entry.embedding(Recognizer::AuraFace), &[1.0, 2.0, 3.0]);
        assert_eq!(entry.embedding(Recognizer::SFace), &[4.0, 5.0, 6.0]);
        assert!(entry.has_embedding(Recognizer::AuraFace));
        assert!(entry.has_embedding(Recognizer::SFace));
    }

    #[test]
    fn register_replaces_existing_entry() {
        let mut g = Gallery::default();
        g.register("A", e(vec![1.0, 2.0]), e(vec![3.0, 4.0]));
        g.register("A", e(vec![5.0, 6.0]), e(vec![7.0, 8.0]));

        assert_eq!(g.entries.len(), 1);
        assert_eq!(g.entries[0].auraface, vec![5.0, 6.0]);
        assert_eq!(g.entries[0].sface, vec![7.0, 8.0]);
    }

    #[test]
    fn match_faces_uses_requested_recognizer_space() {
        let mut g = Gallery::default();
        g.register("A", e(vec![1.0, 0.0]), e(vec![0.0, 1.0]));
        g.register("B", e(vec![0.9, 0.1]), e(vec![0.1, 0.9]));

        let match_a = g.match_faces(&[e(vec![1.0, 0.0])], 0.0, Recognizer::AuraFace)[0].clone().unwrap();
        assert_eq!(match_a.name, "A");
        assert_eq!(match_a.score, 1.0);

        let match_s = g.match_faces(&[e(vec![0.0, 1.0])], 0.0, Recognizer::SFace)[0].clone().unwrap();
        assert_eq!(match_s.name, "A");
        assert_eq!(match_s.score, 1.0);
    }

    #[test]
    fn match_faces_skips_entries_missing_recognizer_space() {
        let mut g = Gallery::default();
        g.register("A", e(vec![1.0, 0.0]), e(vec![]));
        g.register("B", e(vec![0.0, 1.0]), e(vec![1.0, 0.0]));

        let match_s = g.match_faces(&[e(vec![1.0, 0.0])], 0.0, Recognizer::SFace)[0].clone().unwrap();
        assert_eq!(match_s.name, "B");
        assert_eq!(match_s.score, 1.0);
    }

    #[test]
    fn match_faces_threshold_blocks_match() {
        let mut g = Gallery::default();
        g.register("A", e(vec![1.0, 0.0]), e(vec![0.0, 1.0]));

        let matches = g.match_faces(&[e(vec![1.0, 0.0])], 1.1, Recognizer::AuraFace);
        assert!(matches[0].is_none());
    }

    #[test]
    fn legacy_embedding_alias_loads_as_auraface() {
        let json = r#"{"entries":[{"name":"A","embedding":[1.0, 2.0, 3.0]}]}"#;
        let g: Gallery = serde_json::from_str(json).unwrap();
        let entry = &g.entries[0];
        assert_eq!(entry.auraface, vec![1.0, 2.0, 3.0]);
        assert!(entry.sface.is_empty());
        assert_eq!(entry.embedding(Recognizer::AuraFace), &[1.0, 2.0, 3.0]);
        assert!(!entry.has_embedding(Recognizer::SFace));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut g = Gallery::default();
        g.register("A", e(vec![1.0, 2.0]), e(vec![3.0, 4.0]));
        let path = std::env::temp_dir().join(format!("facewatch-test-{}.json", std::process::id()));
        g.save(&path).unwrap();
        let loaded = Gallery::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.entries[0].auraface, vec![1.0, 2.0]);
        assert_eq!(loaded.entries[0].sface, vec![3.0, 4.0]);
    }

    #[test]
    fn summary_formats_names() {
        let mut g = Gallery::default();
        g.register("A", e(vec![]), e(vec![]));
        g.register("B", e(vec![]), e(vec![]));
        assert_eq!(g.summary(), "A, B");
        assert_eq!(Gallery::default().summary(), "(empty)");
    }

    #[test]
    fn cosine_computes_dot_product() {
        assert_eq!(Gallery::cosine(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
        assert_eq!(Gallery::cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }
}
