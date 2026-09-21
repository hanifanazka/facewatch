//! pupil pipeline elements: detection, recognition, matching, and an overlay
//! terminal sink that draws results, saves annotated frames, and streams them
//! to the display channel.

use std::sync::{Arc, Mutex};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::Result;
use pupi::{Buffer, Caps, Element, SinkPad, SourcePad, TypeCap};

use crate::align;
use crate::aura::AuraFace;
use crate::draw;
use crate::face::{
    Array3U8, DetectedFace, FaceEmbeddings, FaceFrame, FaceMatches, Rgb, rgb_to_arr3_ref,
    arr3_to_rgb,
};
use crate::gallery::Gallery;
use crate::scrfd::Scrfd;

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

/// Stage 1: `Array3<u8>` RGB frame -> `FaceFrame` (detections).
pub struct DetectionElement {
    sink: Arc<Mutex<SinkPad>>,
    src: Arc<Mutex<SourcePad>>,
}

impl DetectionElement {
    pub fn new(detector: Scrfd) -> Self {
        let mut detector = detector;
        let (sink, src) = make_pads::<Array3U8, FaceFrame>(move |buffer| {
            let frame = buffer
                .downcast_ref::<Array3U8>()
                .ok_or("expected Array3<u8> frame")?
                .clone();
            let faces = detector.detect(&frame).map_err(|e| e.to_string())?;
            Ok(Buffer::new(FaceFrame {
                src: Arc::new(frame),
                faces,
            }))
        });
        Self { sink, src }
    }
}

impl Element for DetectionElement {
    fn src_pad(&self, name: &str) -> Option<&Arc<Mutex<SourcePad>>> {
        (name == "src").then_some(&self.src)
    }
    fn sink_pad(&self, name: &str) -> Option<&Arc<Mutex<SinkPad>>> {
        (name == "sink").then_some(&self.sink)
    }
}

/// Stage 2: `FaceFrame` -> `FaceEmbeddings` (aligned crops embedded).
pub struct RecognitionElement {
    sink: Arc<Mutex<SinkPad>>,
    src: Arc<Mutex<SourcePad>>,
}

impl RecognitionElement {
    pub fn new(aura: AuraFace) -> Self {
        let mut aura = aura;
        let (sink, src) = make_pads::<FaceFrame, FaceEmbeddings>(move |buffer| {
            let frame = buffer
                .downcast_ref::<FaceFrame>()
                .ok_or("expected FaceFrame")?
                .clone();
            let mut embeddings = Vec::with_capacity(frame.faces.len());
            for face in &frame.faces {
                let crop = align::norm_crop(&frame.src, &face.kps, 112);
                let embedding = aura.embed(&crop).map_err(|e| e.to_string())?;
                embeddings.push(embedding);
            }
            Ok(Buffer::new(FaceEmbeddings {
                src: frame.src.clone(),
                faces: frame.faces.clone(),
                embeddings,
            }))
        });
        Self { sink, src }
    }
}

impl Element for RecognitionElement {
    fn src_pad(&self, name: &str) -> Option<&Arc<Mutex<SourcePad>>> {
        (name == "src").then_some(&self.src)
    }
    fn sink_pad(&self, name: &str) -> Option<&Arc<Mutex<SinkPad>>> {
        (name == "sink").then_some(&self.sink)
    }
}

/// Stage 3: `FaceEmbeddings` -> `FaceMatches` (cosine similarity vs gallery).
pub struct MatchElement {
    sink: Arc<Mutex<SinkPad>>,
    src: Arc<Mutex<SourcePad>>,
}

impl MatchElement {
    pub fn new(gallery: Arc<Mutex<Gallery>>, threshold: f32) -> Self {
        let (sink, src) = make_pads::<FaceEmbeddings, FaceMatches>(move |buffer| {
            let faces = buffer
                .downcast_ref::<FaceEmbeddings>()
                .ok_or("expected FaceEmbeddings")?
                .clone();
            let matches = gallery
                .lock()
                .map_err(|_| "gallery mutex poisoned".to_owned())?
                .match_faces(&faces.embeddings, threshold);
            Ok(Buffer::new(FaceMatches {
                src: faces.src.clone(),
                faces: faces.faces.clone(),
                matches,
            }))
        });
        Self { sink, src }
    }
}

impl Element for MatchElement {
    fn src_pad(&self, name: &str) -> Option<&Arc<Mutex<SourcePad>>> {
        (name == "src").then_some(&self.src)
    }
    fn sink_pad(&self, name: &str) -> Option<&Arc<Mutex<SinkPad>>> {
        (name == "sink").then_some(&self.sink)
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
pub struct OverlaySink {
    sink: Arc<Mutex<SinkPad>>,
    _src: Arc<Mutex<SourcePad>>,
}

impl OverlaySink {
    pub fn new(options: OverlayOptions, display_tx: Option<Sender<Array3U8>>) -> Self {
        let (sink, src) = make_pads::<FaceMatches, ()>(move |buffer| {
            let matches = buffer
                .downcast_ref::<FaceMatches>()
                .ok_or("expected FaceMatches")?
                .clone();
            render_overlay(&matches, &options, &display_tx);
            Ok(Buffer::new(()))
        });
        Self { sink, _src: src }
    }
}

impl Element for OverlaySink {
    fn src_pad(&self, name: &str) -> Option<&Arc<Mutex<SourcePad>>> {
        (name == "src").then_some(&self._src)
    }
    fn sink_pad(&self, name: &str) -> Option<&Arc<Mutex<SinkPad>>> {
        (name == "sink").then_some(&self.sink)
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
    /// Builds the chain from the given models and gallery.
    pub fn new(
        models: &crate::models::Models,
        gallery: Arc<Mutex<Gallery>>,
        threshold: f32,
        options: OverlayOptions,
    ) -> Result<Self> {
        let detector = Scrfd::load(models.scrfd)?;
        let aura = AuraFace::load(models.auraface)?;
        let (tx, display_rx) = std::sync::mpsc::channel::<Array3U8>();

        let detection = DetectionElement::new(detector);
        let recognition = RecognitionElement::new(aura);
        let matching = MatchElement::new(gallery, threshold);
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

fn render_overlay(matches: &FaceMatches, options: &OverlayOptions, display_tx: &Option<Sender<Array3U8>>) {
    let mut state = overlay_state();
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
        let _ = tx.send(rgb_to_arr3_ref(&img));
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

/// Per-process overlay state shared between (synchronous) pipeline frames.
static OVERLAY_STATE: std::sync::OnceLock<std::sync::Mutex<OverlayState>> =
    std::sync::OnceLock::new();

fn overlay_state() -> std::sync::MutexGuard<'static, OverlayState> {
    OVERLAY_STATE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
