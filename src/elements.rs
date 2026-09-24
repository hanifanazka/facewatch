//! pupil pipeline elements: detection, recognition, matching, and an overlay
//! terminal sink that draws results, saves annotated frames, and streams them
//! to the display channel.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::Result;
use pupi::{Buffer, Caps, Element, SinkPad, SourcePad, TypeCap};

use crate::align;
use crate::aura::AuraFace;
use crate::draw;
use crate::face::{
    Array3U8, DetectedFace, FaceEmbeddings, FaceFrame, FaceMatches, Rgb, arr3_to_rgb, rgb_to_arr3,
};
use crate::gallery::{Gallery, Recognizer};
use crate::models::Models;
use crate::profile;
use crate::scrfd::Scrfd;
use crate::sface::SFace;
use crate::yunet::Yunet;

/// Selectable face-detection backend; keeps both ONNX detectors embedded for
/// A/B comparison (`--detector scrfd` vs the default `yunet`).
pub enum Detector {
    Scrfd(Scrfd),
    Yunet(Yunet),
}

impl Detector {
    /// Loads the requested backend (`"scrfd"` or, by default, `"yunet"`) from
    /// the embedded model bytes.
    pub fn load(kind: &str, models: &Models, threads: usize, input_size: usize) -> Result<Self> {
        match kind {
            "scrfd" => Ok(Detector::Scrfd(Scrfd::load(models.scrfd, threads, input_size)?)),
            "yunet" => Ok(Detector::Yunet(Yunet::load(models.yunet, threads, input_size)?)),
            other => anyhow::bail!(
                "unknown detector {other:?} (expected \"yunet\" or \"scrfd\")"
            ),
        }
    }

    /// Runs detection on an RGB frame, delegating to the active backend.
    pub fn detect(&mut self, frame: &Array3U8) -> Result<Vec<DetectedFace>> {
        match self {
            Detector::Scrfd(d) => d.detect(frame),
            Detector::Yunet(d) => d.detect(frame),
        }
    }
}

/// Selectable face-recognition backend; keeps both ONNX recognizers embedded for
/// A/B comparison (`--recognizer sface` vs the default `auraface`).
pub enum Recognizer_ {
    AuraFace(AuraFace),
    SFace(SFace),
}

impl Recognizer_ {
    /// Loads the requested backend (`"auraface"` or `"sface"`) from
    /// the embedded model bytes.
    pub fn load(kind: &str, models: &Models, threads: usize) -> Result<Self> {
        match kind {
            "auraface" => Ok(Recognizer_::AuraFace(AuraFace::load(models.auraface, threads)?)),
            "sface" => Ok(Recognizer_::SFace(SFace::load(models.sface, threads)?)),
            other => anyhow::bail!(
                "unknown recognizer {other:?} (expected \"auraface\" or \"sface\")"
            ),
        }
    }

    /// Runs recognition on an aligned RGB crop, delegating to the active backend.
    pub fn embed(&mut self, crop: &Array3U8) -> Result<crate::face::Embedding> {
        match self {
            Recognizer_::AuraFace(a) => a.embed(crop),
            Recognizer_::SFace(s) => s.embed(crop),
        }
    }
}

/// Creates a sink pad accepting `I` and a source pad offering `O` linked by a
/// synchronous transform callback.
fn make_pads<I: 'static, O: 'static>(
    mut transform: impl FnMut(&Buffer) -> Result<Buffer, String> + Send + 'static,
) -> (Arc<Mutex<SinkPad>>, Arc<Mutex<SourcePad>>) {
    let src = Arc::new(Mutex::new(SourcePad::new(Caps {
        formats: vec![TypeCap::of::<O>()],
    })));
    let src_clone = Arc::clone(&src);
    let sink = Arc::new(Mutex::new(SinkPad::new(
        Caps {
            formats: vec![TypeCap::of::<I>()],
        },
        |_| Ok(()),
        move |buffer: Buffer| {
            let out = transform(&buffer)?;
            src_clone
                .lock()
                .map_err(|_| "element source pad mutex poisoned".to_owned())?
                .push(out);
            Ok(())
        },
    )));
    (sink, src)
}

/// A one-sink/one-source pipeline stage: the sink pad accepts `I`, the source
/// pad offers `O`, and a synchronous transform turns each input buffer into
/// the stage's output.
pub struct Stage<I: 'static, O: 'static> {
    sink: Arc<Mutex<SinkPad>>,
    src: Arc<Mutex<SourcePad>>,
    _marker: PhantomData<(I, O)>,
}

impl<I: 'static, O: 'static> Stage<I, O> {
    /// Builds a stage whose source pad offers `O`, driven by a synchronous
    /// transform from each `I` input buffer. The concrete pipeline stages
    /// wrap this in their named `new` constructors below.
    pub fn from_transform(
        transform: impl FnMut(&Buffer) -> Result<Buffer, String> + Send + 'static,
    ) -> Self {
        let (sink, src) = make_pads::<I, O>(transform);
        Self {
            sink,
            src,
            _marker: PhantomData,
        }
    }
}

impl<I: 'static, O: 'static> Element for Stage<I, O> {
    fn src_pad(&self, name: &str) -> Option<&Arc<Mutex<SourcePad>>> {
        (name == "src").then_some(&self.src)
    }
    fn sink_pad(&self, name: &str) -> Option<&Arc<Mutex<SinkPad>>> {
        (name == "sink").then_some(&self.sink)
    }
}

/// Stage 1: `Array3<u8>` RGB frame -> `FaceFrame` (detections).
pub type DetectionElement = Stage<Array3U8, FaceFrame>;

impl Stage<Array3U8, FaceFrame> {
    pub fn new(detector: Detector) -> Self {
        let mut detector = detector;
        Stage::from_transform(move |buffer| {
            let frame = buffer
                .downcast_ref::<Array3U8>()
                .ok_or("expected Array3<u8> frame")?
                .clone();
            let faces = profile::time("detect", || detector.detect(&frame)).map_err(|e| e.to_string())?;
            Ok(Buffer::new(FaceFrame {
                src: Arc::new(frame),
                faces,
            }))
        })
    }
}

/// Stage 2: `FaceFrame` -> `FaceEmbeddings` (aligned crops embedded).
pub type RecognitionElement = Stage<FaceFrame, FaceEmbeddings>;

impl Stage<FaceFrame, FaceEmbeddings> {
    pub fn new(recognizer: Recognizer_) -> Self {
        let mut recognizer = recognizer;
        Stage::from_transform(move |buffer| {
            let frame = buffer
                .downcast_ref::<FaceFrame>()
                .ok_or("expected FaceFrame")?
                .clone();
            let mut embeddings = Vec::with_capacity(frame.faces.len());
            for face in &frame.faces {
                let crop = profile::time("align", || align::norm_crop(&frame.src, &face.kps, 112));
                let embedding =
                    profile::time("embed", || recognizer.embed(&crop)).map_err(|e| e.to_string())?;
                embeddings.push(embedding);
            }
            Ok(Buffer::new(FaceEmbeddings {
                src: frame.src.clone(),
                faces: frame.faces.clone(),
                embeddings,
            }))
        })
    }
}

/// Stage 3: `FaceEmbeddings` -> `FaceMatches` (cosine similarity vs gallery).
pub type MatchElement = Stage<FaceEmbeddings, FaceMatches>;

impl Stage<FaceEmbeddings, FaceMatches> {
    pub fn new(gallery: Arc<Mutex<Gallery>>, threshold: f32, recognizer: Recognizer) -> Self {
        Stage::from_transform(move |buffer| {
            let faces = buffer
                .downcast_ref::<FaceEmbeddings>()
                .ok_or("expected FaceEmbeddings")?
                .clone();
            let matches = gallery
                .lock()
                .map_err(|_| "gallery mutex poisoned".to_owned())?
                .match_faces(&faces.embeddings, threshold, recognizer);
            Ok(Buffer::new(FaceMatches {
                src: faces.src.clone(),
                faces: faces.faces.clone(),
                matches,
            }))
        })
    }
}

/// Options for the terminal overlay stage.
#[derive(Clone, Debug)]
pub struct OverlayOptions {
    /// Optional directory for saved annotated frames.
    pub save_dir: Option<std::path::PathBuf>,
    /// Save frequency (defaults to every frame).
    pub save_every: usize,
    /// Log a JSON line per frame.
    pub verbose: bool,
}

/// Stage 4 (terminal): draws boxes/labels on the frame, optionally saves and
/// forwards it. Has only a sink pad.
pub type OverlaySink = Stage<FaceMatches, ()>;

impl Stage<FaceMatches, ()> {
    pub fn new(options: OverlayOptions, display_tx: Option<Sender<Array3U8>>) -> Self {
        let mut state = OverlayState::default();
        Stage::from_transform(move |buffer| {
            let matches = buffer
                .downcast_ref::<FaceMatches>()
                .ok_or("expected FaceMatches")?
                .clone();
            profile::time("draw", || render_overlay(&matches, &options, &display_tx, &mut state));
            Ok(Buffer::new(()))
        })
    }
}

/// A fully linked and negotiated detection->recognition->matching->overlay
/// chain, plus the channel carrying annotated frames out for display.
pub struct Chain {
    pub detection: DetectionElement,
    // The remaining stages are only ever reached through the pads linked in
    // `Chain::new`; rustc sees the fields as unread even though dropping them
    // would destroy the pipeline (elements own their sessions and pads).
    #[allow(dead_code)]
    pub recognition: RecognitionElement,
    #[allow(dead_code)]
    pub matching: MatchElement,
    #[allow(dead_code)]
    pub overlay: OverlaySink,
    /// Receives one annotated frame per processed frame.
    pub display_rx: std::sync::mpsc::Receiver<Array3U8>,
}

impl Chain {
    /// Builds the chain from the given models and gallery. `detector` selects
    /// the detection backend (`"scrfd"` or `"yunet"`), `recognizer` selects
    /// the recognition backend (`"auraface"` default or `"sface"`), `threads`
    /// sets the per-session ONNX intra-op thread count, and `input_size` the
    /// detector input resolution (320 or 640).
    pub fn new(
        models: &Models,
        gallery: Arc<Mutex<Gallery>>,
        threshold: f32,
        options: OverlayOptions,
        detector: &str,
        recognizer: &str,
        threads: usize,
        input_size: usize,
    ) -> Result<Self> {
        let detector = Detector::load(detector, models, threads, input_size)?;
        let recognizer_kind = match recognizer {
            "sface" => Recognizer::SFace,
            _ => Recognizer::AuraFace,
        };
        let recognizer = Recognizer_::load(recognizer, models, threads)?;
        let (tx, display_rx) = std::sync::mpsc::channel::<Array3U8>();

        let detection = DetectionElement::new(detector);
        let recognition = RecognitionElement::new(recognizer);
        let matching = MatchElement::new(gallery, threshold, recognizer_kind);
        let overlay = OverlaySink::new(options, Some(tx));

        link_upstream(&detection, &recognition)?;
        link_upstream(&recognition, &matching)?;
        link_upstream(&matching, &overlay)?;
        negotiate_upstream(&detection)?;
        negotiate_upstream(&recognition)?;
        negotiate_upstream(&matching)?;

        Ok(Self {
            detection,
            recognition,
            matching,
            overlay,
            display_rx,
        })
    }

    /// Pushes one RGB frame into the chain (used by still-image mode).
    pub fn push_frame(&self, frame: Array3U8) -> Result<()> {
        let sink = self
            .detection
            .sink_pad("sink")
            .ok_or_else(|| anyhow::anyhow!("detection element has no sink pad"))?;
        let mut sink = sink
            .lock()
            .map_err(|_| anyhow::anyhow!("detection sink pad mutex poisoned"))?;
        sink.push(Buffer::new(frame))
            .map_err(|e| anyhow::anyhow!("detection stage rejected frame: {e}"))
    }
}

/// Links the upstream element's `src` pad to the downstream element's `sink`.
pub fn link_upstream(a: &impl Element, b: &impl Element) -> Result<()> {
    let src = a
        .src_pad("src")
        .ok_or_else(|| anyhow::anyhow!("upstream element has no src pad"))?;
    let sink = b
        .sink_pad("sink")
        .ok_or_else(|| anyhow::anyhow!("downstream element has no sink pad"))?;
    src.lock()
        .map_err(|_| anyhow::anyhow!("source pad mutex poisoned"))?
        .link(Arc::clone(sink));
    Ok(())
}

/// Negotiates caps on the element's `src` pad (panics on incompatibility).
pub fn negotiate_upstream(e: &impl Element) -> Result<()> {
    let src = e
        .src_pad("src")
        .ok_or_else(|| anyhow::anyhow!("element has no src pad"))?;
    src.lock()
        .map_err(|_| anyhow::anyhow!("source pad mutex poisoned"))?
        .negotiate();
    Ok(())
}

struct OverlayState {
    frame_no: u64,
    fps: f32,
    last_fps_sample: Instant,
    fps_count: u64,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            frame_no: 0,
            fps: 0.0,
            last_fps_sample: Instant::now(),
            fps_count: 0,
        }
    }
}

fn render_overlay(
    matches: &FaceMatches,
    options: &OverlayOptions,
    display_tx: &Option<Sender<Array3U8>>,
    state: &mut OverlayState,
) {
    state.frame_no += 1;

    // Frame-rate estimation over a sliding second.
    state.fps_count += 1;
    let now = Instant::now();
    let elapsed = now.duration_since(state.last_fps_sample);
    if elapsed >= Duration::from_secs(1) {
        state.fps = state.fps_count as f32 / elapsed.as_secs_f32();
        state.fps_count = 0;
        state.last_fps_sample = now;
    }

    let mut img = match arr3_to_rgb(&matches.src) {
        Some(img) => img,
        None => return,
    };

    for (i, face) in matches.faces.iter().enumerate() {
        let matched = matches.matches.get(i).cloned().flatten();
        let color = if matched.is_some() { Rgb::GREEN } else { Rgb::RED };
        draw_face(&mut img, face, matched.as_ref(), color);
    }

    let info = format!(
        "facewatch | faces: {} | fps: {:.0}",
        matches.faces.len(),
        state.fps
    );
    let tw = draw::text_width(&info) as i32;
    draw::fill_rect(&mut img, 4, 4, 4 + tw + 3, 4 + 7 + 3, Rgb::BLACK);
    draw::draw_text(&mut img, 6, 6, &info, Rgb::WHITE);

    if options.verbose {
        let faces: Vec<serde_json::Value> = matches
            .faces
            .iter()
            .enumerate()
            .map(|(i, f)| {
                serde_json::json!({
                    "box": f.bbox,
                    "score": f.score,
                    "kps": f.kps,
                    "match": matches.matches.get(i).cloned().flatten().map(|m| {
                        serde_json::json!({ "name": m.name, "score": m.score })
                    }),
                })
            })
            .collect();
        eprintln!(
            "{}",
            serde_json::json!({ "frame": state.frame_no, "faces": faces })
        );
    }

    // Save annotated frames.
    if let Some(dir) = &options.save_dir {
        let every = options.save_every.max(1) as u64;
        if state.frame_no % every == 0 {
            std::fs::create_dir_all(dir).ok();
            let name = dir.join(format!("frame_{:06}.png", state.frame_no));
            if img.save(&name).is_err() {
                eprintln!("failed to save {}", name.display());
            }
        }
    }

    // Stream to the display.
    if let Some(tx) = display_tx {
        let _ = tx.send(rgb_to_arr3(&img));
    }
}

fn draw_face(
    img: &mut image::RgbImage,
    face: &DetectedFace,
    matched: Option<&crate::face::Match>,
    color: Rgb,
) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let b = face.bbox;
    let x1 = b[0].round().clamp(0.0, w as f32 - 1.0) as i32;
    let y1 = b[1].round().clamp(0.0, h as f32 - 1.0) as i32;
    let x2 = b[2].round().clamp(0.0, w as f32 - 1.0) as i32;
    let y2 = b[3].round().clamp(0.0, h as f32 - 1.0) as i32;

    draw::draw_rect(img, x1, y1, x2, y2, color, 2);
    draw::draw_kps(img, &face.kps, color, 2);

    let label = match matched {
        Some(m) => format!("{} {:.2}", m.name, m.score),
        None => format!("Face unknown {:.2}", face.score),
    };
    let tw = draw::text_width(&label) as i32;
    let label_y = (y1 - 12).max(4);
    draw::fill_rect(img, x1, label_y, (x1 + tw + 3).min(w - 1), (label_y + 10).min(h - 1), Rgb::BLACK);
    draw::draw_text(img, x1 + 1, label_y + 1, &label, Rgb::WHITE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::face::Embedding;
    use pupi::Buffer;

    /// A downstream sink pad that records every buffer it receives.
    fn capture_sink<T: 'static>() -> (Arc<Mutex<SinkPad>>, Arc<Mutex<Vec<Buffer>>>) {
        let received: Arc<Mutex<Vec<Buffer>>> = Arc::new(Mutex::new(Vec::new()));
        let recv = Arc::clone(&received);
        let pad = Arc::new(Mutex::new(SinkPad::new(
            Caps {
                formats: vec![TypeCap::of::<T>()],
            },
            |_| Ok(()),
            move |buffer| {
                recv.lock().unwrap().push(buffer);
                Ok(())
            },
        )));
        (pad, received)
    }

    /// Links a downstream capture sink to `stage`, negotiates, and returns the
    /// capture.
    fn link_capture<O: 'static>(
        stage: &impl Element,
    ) -> Arc<Mutex<Vec<Buffer>>> {
        let (sink, received) = capture_sink::<O>();
        let src = stage
            .src_pad("src")
            .expect("stage has a src pad");
        src.lock().unwrap().link(Arc::clone(&sink));
        src.lock().unwrap().negotiate();
        received
    }

    fn push_to<I>(stage: &impl Element, payload: I) -> Result<(), String>
    where
        I: std::any::Any + Send + Sync,
    {
        let sink = stage.sink_pad("sink").expect("stage has a sink pad");
        sink.lock().unwrap().push(Buffer::new(payload))
    }

    struct Padless;

    impl Element for Padless {
        fn src_pad(&self, _: &str) -> Option<&Arc<Mutex<SourcePad>>> {
            None
        }
        fn sink_pad(&self, _: &str) -> Option<&Arc<Mutex<SinkPad>>> {
            None
        }
    }

    #[test]
    fn stage_exposes_only_its_named_pads() {
        let stage: Stage<u8, ()> = Stage::from_transform(|_| Ok(Buffer::new(())));
        assert!(stage.src_pad("src").is_some());
        assert!(stage.sink_pad("sink").is_some());
        assert!(stage.src_pad("anything-else").is_none());
        assert!(stage.sink_pad("anything-else").is_none());
    }

    #[test]
    fn stage_runs_transform_and_pushes_output_downstream() {
        let stage: Stage<u32, String> = Stage::from_transform(|b| {
            let v = b.downcast_ref::<u32>().ok_or("expected u32")?;
            Ok(Buffer::new(v.to_string()))
        });
        let received = link_capture::<String>(&stage);
        push_to(&stage, 42u32).unwrap();
        let buf = received.lock().unwrap().pop().expect("one output buffer");
        assert_eq!(buf.downcast_ref::<String>(), Some(&"42".to_owned()));
    }

    #[test]
    fn stage_rejects_wrong_payload_type() {
        let stage: Stage<u32, ()> = Stage::from_transform(|b| {
            b.downcast_ref::<u32>().ok_or("expected u32")?;
            Ok(Buffer::new(()))
        });
        let err = push_to(&stage, "not a u32".to_owned()).unwrap_err();
        assert_eq!(err, "expected u32");
        // And a transform failure propagates as an Err with its message.
        let failing: Stage<u8, ()> = Stage::from_transform(|_| Err("boom".to_owned()));
        let err = push_to(&failing, 1u8).unwrap_err();
        assert_eq!(err, "boom");
    }

    #[test]
    fn match_element_matches_against_gallery() {
        let mut g = Gallery::default();
        g.register("A", Embedding(vec![1.0, 0.0]), Embedding(vec![]));
        g.register("B", Embedding(vec![0.0, 1.0]), Embedding(vec![]));
        let gallery = Arc::new(Mutex::new(g));
        let stage = MatchElement::new(Arc::clone(&gallery), 0.5, Recognizer::AuraFace);

        let received = link_capture::<FaceMatches>(&stage);
        let src = Arc::new(Array3U8::zeros((4, 4, 3)));
        let payload = FaceEmbeddings {
            src: Arc::clone(&src),
            faces: vec![],
            embeddings: vec![Embedding(vec![1.0, 0.0]), Embedding(vec![0.0, 1.0])],
        };
        push_to(&stage, payload).unwrap();

        let buf = received.lock().unwrap().pop().unwrap();
        let matches = buf.downcast_ref::<FaceMatches>().unwrap();
        assert_eq!(matches.matches.len(), 2);
        let a = matches.matches[0].as_ref().unwrap();
        assert_eq!(a.name, "A");
        assert_eq!(a.score, 1.0);
        let b = matches.matches[1].as_ref().unwrap();
        assert_eq!(b.name, "B");
    }

    #[test]
    fn match_element_applies_threshold_and_recognizer_space() {
        let mut g = Gallery::default();
        g.register("A", Embedding(vec![1.0, 0.0]), Embedding(vec![0.0, 1.0]));
        let gallery = Arc::new(Mutex::new(g));

        // Below threshold -> None.
        let stage = MatchElement::new(Arc::clone(&gallery), 0.95, Recognizer::AuraFace);
        let received = link_capture::<FaceMatches>(&stage);
        let src = Arc::new(Array3U8::zeros((4, 4, 3)));
        let payload = FaceEmbeddings {
            src,
            faces: vec![],
            embeddings: vec![Embedding(vec![0.9, 0.0])], // 0.9 < 0.95
        };
        push_to(&stage, payload).unwrap();
        let buf = received.lock().unwrap().pop().unwrap();
        let matches = buf.downcast_ref::<FaceMatches>().unwrap();
        assert!(matches.matches[0].is_none());

        // SFace space: A has sface [0,1], probe [0,1] matches at 1.0.
        let stage = MatchElement::new(gallery, 0.5, Recognizer::SFace);
        let received = link_capture::<FaceMatches>(&stage);
        let payload = FaceEmbeddings {
            src: Arc::new(Array3U8::zeros((4, 4, 3))),
            faces: vec![],
            embeddings: vec![Embedding(vec![0.0, 1.0])],
        };
        push_to(&stage, payload).unwrap();
        let buf = received.lock().unwrap().pop().unwrap();
        let matches = buf.downcast_ref::<FaceMatches>().unwrap();
        assert_eq!(matches.matches[0].as_ref().unwrap().name, "A");
    }

    #[test]
    fn overlay_sink_emits_annotated_frame_on_the_display_channel() {
        let (tx, rx) = std::sync::mpsc::channel::<Array3U8>();
        let stage = OverlaySink::new(
            OverlayOptions {
                save_dir: None,
                save_every: 1,
                verbose: false,
            },
            Some(tx),
        );

        let src = Array3U8::zeros((8, 8, 3));
        let payload = FaceMatches {
            src: Arc::new(src.clone()),
            faces: vec![DetectedFace {
                bbox: [1.0, 1.0, 6.0, 6.0],
                score: 0.9,
                kps: [[0.0; 2]; 5],
            }],
            matches: vec![None],
        };
        push_to(&stage, payload).unwrap();

        let out = rx.recv_timeout(Duration::from_secs(5)).expect("annotated frame");
        assert_eq!(out.shape(), &[8, 8, 3]);
        let changed = out.iter().zip(src.iter()).filter(|(a, b)| a != b).count();
        assert!(changed > 0, "the red box must be drawn onto the frame");
    }

    #[test]
    fn overlay_sink_saves_every_nth_frame() {
        let dir = std::env::temp_dir().join(format!("facewatch-overlay-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let stage = OverlaySink::new(
            OverlayOptions {
                save_dir: Some(dir.clone()),
                save_every: 2,
                verbose: false,
            },
            None,
        );

        let payload = || FaceMatches {
            src: Arc::new(Array3U8::zeros((4, 4, 3))),
            faces: vec![],
            matches: vec![],
        };
        for _ in 0..3 {
            push_to(&stage, payload()).unwrap();
        }

        assert!(dir.join("frame_000002.png").exists(), "frame 2 saved (2 % 2 == 0)");
        assert!(!dir.join("frame_000001.png").exists());
        assert!(!dir.join("frame_000003.png").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn link_upstream_reports_missing_pads() {
        let err = link_upstream(&Padless, &Padless).unwrap_err();
        assert!(err.to_string().contains("no src pad"), "{err}");
    }

    #[test]
    fn detector_load_rejects_unknown_kind_before_touching_models() {
        let models = Models {
            scrfd: &[],
            yunet: &[],
            auraface: &[],
            sface: &[],
        };
        let err = Detector::load("bogus", &models, 1, 640)
            .err()
            .expect("unknown detector must fail");
        assert!(err.to_string().contains("unknown detector"), "{err}");
    }

    #[test]
    fn recognizer_load_rejects_unknown_kind_before_touching_models() {
        let models = Models {
            scrfd: &[],
            yunet: &[],
            auraface: &[],
            sface: &[],
        };
        let err = Recognizer_::load("bogus", &models, 1)
            .err()
            .expect("unknown recognizer must fail");
        assert!(err.to_string().contains("unknown recognizer"), "{err}");
    }

    #[test]
    fn chain_rejects_unknown_detector() {
        let models = Models {
            scrfd: &[],
            yunet: &[],
            auraface: &[],
            sface: &[],
        };
        let gallery = Arc::new(Mutex::new(Gallery::default()));
        let err = Chain::new(
            &models,
            gallery,
            0.4,
            OverlayOptions {
                save_dir: None,
                save_every: 1,
                verbose: false,
            },
            "bogus",
            "auraface",
            1,
            640,
        )
        .err()
        .expect("unknown detector must fail");
        assert!(err.to_string().contains("unknown detector"), "{err}");
    }

    #[test]
    fn chain_end_to_end_with_real_models_pushes_annotated_frames() {
        // Heavy: loads the embedded ONNX models (both recognizer and detector
        // pairs); serialized via MODEL_LOCK.
        let _g = crate::face::MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let models = Models::embedded();
        let gallery = Arc::new(Mutex::new(Gallery::default()));
        let opts = OverlayOptions {
            save_dir: None,
            save_every: 1,
            verbose: false,
        };

        // yunet + auraface
        let chain = Chain::new(&models, Arc::clone(&gallery), 0.4, opts.clone(), "yunet", "auraface", 1, 640)
            .expect("yunet+auraface chain loads");
        let frame = Array3U8::from_shape_fn((480, 640, 3), |_| 100u8);
        chain.push_frame(frame.clone()).expect("frame accepted");
        let out = chain
            .display_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("annotated frame arrives");
        assert_eq!(out.shape(), &[480, 640, 3]);
        drop(chain);

        // scrfd + sface
        let chain = Chain::new(&models, gallery, 0.4, opts, "scrfd", "sface", 1, 640)
            .expect("scrfd+sface chain loads");
        chain.push_frame(frame).expect("frame accepted");
        let out = chain
            .display_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("annotated frame arrives");
        assert_eq!(out.shape(), &[480, 640, 3]);
    }
}