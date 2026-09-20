//! facewatch: webcam face recognition on the pupi push-pipeline, using SCRFD
//! (detection) + AuraFace (recognition) through ONNX Runtime (ort).

mod align;
mod aura;
mod draw;
mod elements;
mod face;
mod gallery;
mod models;
mod rtsp;
mod scrfd;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use pupi::{GstreamRunner, GstreamRunnerError};

use crate::aura::AuraFace;
use crate::elements::{Chain, OverlayOptions};
use crate::face::Array3U8;
use crate::gallery::{Gallery, load_gallery_any};
use crate::models::{Models, ensure_models};
use crate::rtsp::RtspPublisher;
use crate::scrfd::Scrfd;

/// Default gallery location: `$HOME/.facewatch/gallery.json`.
fn default_gallery() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut path = home.unwrap_or_else(|| PathBuf::from("."));
    path.push(".facewatch");
    path.push("gallery.json");
    path
}

#[derive(Parser)]
#[command(
    name = "facewatch",
    version,
    about = "Webcam face recognition on the pupi push-pipeline (SCRFD + AuraFace, ONNX Runtime)."
)]
struct Cli {
    /// Directory that contains (or will receive) the ONNX models.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// Minimum cosine similarity required for a gallery match.
    #[arg(long, default_value_t = 0.40)]
    threshold: f32,

    /// Gallery JSON path (defaults to ~/.facewatch/gallery.json).
    #[arg(long)]
    gallery: Option<PathBuf>,

    /// Publish the annotated stream to this RTSP URL (mediamtx).
    #[arg(long, default_value = "rtsp://127.0.0.1:8554/facewatch")]
    rtsp: String,

    /// Do not publish to RTSP (process frames headlessly).
    #[arg(long)]
    no_rtsp: bool,

    /// Save annotated frames to this directory.
    #[arg(long)]
    save_dir: Option<PathBuf>,

    /// Save every Nth annotated frame (with --save-dir).
    #[arg(long, default_value_t = 1)]
    save_every: usize,

    /// Log a JSON line per processed frame to stderr.
    #[arg(long)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Register a face in the gallery from a still image (or one webcam snapshot).
    Register {
        /// Name to associate with the registered face.
        name: String,
        /// Optional image file; uses a webcam snapshot when omitted.
        image: Option<PathBuf>,
    },
    /// Run live webcam face recognition.
    Run {
        /// Stop after processing this many frames (~0.5-2 fps on this box).
        #[arg(long)]
        frames: Option<u64>,
    },
    /// Recognize faces in a single image file.
    Image {
        /// Image file to analyze.
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let models = ensure_models(&cli.models_dir)?;
    let gallery_path = cli.gallery.clone().unwrap_or_else(default_gallery);
    let gallery = Arc::new(Mutex::new(load_gallery_any(&gallery_path)?));

    if cli.verbose {
        let names = gallery
            .lock()
            .map(|g| g.summary())
            .unwrap_or_else(|_| "(unavailable)".to_owned());
        eprintln!("gallery {}: {}", gallery_path.display(), names);
    }

    match &cli.command {
        Command::Register { name, image } => {
            register(
                name,
                image.as_ref(),
                &models,
                &gallery_path,
                &gallery,
                cli.save_dir.as_ref(),
            )
        }
        Command::Run { frames } => run(&cli, &models, &gallery, *frames),
        Command::Image { path } => analyze_image(&cli, &models, &gallery, path),
    }
}

/// Loads an image file as an RGB `Array3<u8>` frame.
fn load_image_frame(path: &PathBuf) -> Result<Array3U8> {
    let img = image::open(path).with_context(|| format!("failed to open image {}", path.display()))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let frame =
        Array3U8::from_shape_vec((h as usize, w as usize, 3), rgb.into_raw())
            .context("image buffer layout is invalid")?;
    Ok(frame)
}

/// Grabs one frame from the default webcam.
fn webcam_snapshot() -> Result<Array3U8> {
    let runner = GstreamRunner::webcam()?;
    runner.start_pipeline()?;
    let mut frame: Option<Array3U8> = None;
    for _ in 0..60 {
        match runner.pull_frame() {
            Ok(Some(f)) => {
                frame = Some(f);
                break;
            }
            Ok(None) => break,
            Err(_) => continue,
        }
    }
    runner.stop().ok();
    frame.ok_or_else(|| anyhow!("no webcam frame captured (is a camera available?)"))
}

/// Registers the first (best-scoring) detected face under `name`.
fn register(
    name: &str,
    image: Option<&PathBuf>,
    models: &Models,
    gallery_path: &PathBuf,
    gallery: &Arc<Mutex<Gallery>>,
    save_dir: Option<&PathBuf>,
) -> Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("registration name must not be empty");
    }

    let mut detector = Scrfd::load(&models.scrfd)?;
    let mut aura = AuraFace::load(&models.auraface)?;

    let frame = match image {
        Some(path) => load_image_frame(path)?,
        None => webcam_snapshot()?,
    };

    let faces = detector.detect(&frame)?;
    anyhow::ensure!(!faces.is_empty(), "no face detected in the source frame");
    let best = faces
        .iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        .expect("faces is non-empty");

    let crop = align::norm_crop(&frame, &best.kps, 112);
    if let Some(dir) = save_dir {
        std::fs::create_dir_all(dir).ok();
        if let Some(img) = face::arr3_to_rgb(&crop) {
            let path = dir.join(format!("crop_{name}.png"));
            if img.save(&path).is_ok() {
                eprintln!("saved alignment crop to {}", path.display());
            }
        }
    }
    let embedding = aura.embed(&crop)?;

    let mut gallery = gallery.lock().map_err(|_| anyhow!("gallery mutex poisoned"))?;
    gallery.register(name, embedding);
    gallery.save(gallery_path)?;

    eprintln!(
        "registered '{name}' (score {:.3}, box {:?}) in {}",
        best.score,
        best.bbox,
        gallery_path.display()
    );
    Ok(())
}

/// Builds the linked chain and drives it from the default webcam.
fn run(
    cli: &Cli,
    models: &Models,
    gallery: &Arc<Mutex<Gallery>>,
    frames: Option<u64>,
) -> Result<()> {
    let chain = Chain::new(models, Arc::clone(gallery), cli.threshold, overlay_opts(cli))?;

    let runner = GstreamRunner::webcam()?;
    elements::link_upstream(&runner, &chain.detection)?;
    elements::negotiate_upstream(&runner)?;

    eprintln!("driving pipeline from GStreamer webcam source");
    if cli.save_dir.is_some() {
        eprintln!("saving annotated frames to {}", cli.save_dir.as_ref().unwrap().display());
    }

    runner.start_pipeline()?;
    let mut publisher: Option<RtspPublisher> = None;

    let mut processed: u64 = 0;
    let result = loop {
        match runner.pull_frame() {
            Ok(Some(_)) => {}
            Ok(None) => break Ok(()), // end of stream
            Err(GstreamRunnerError::PullSample(_)) => continue,
            Err(e) => break Err(anyhow!("frame pull failed: {e}")),
        }
        processed += 1;

        // Drain the annotated-frame channel (keeping the newest frame) and
        // publish it over RTSP. The publisher is created lazily from the
        // first frame's dimensions.
        let mut latest = None;
        while let Ok(frame) = chain.display_rx.try_recv() {
            latest = Some(frame);
        }
        if let (Some(frame), false) = (&latest, cli.no_rtsp) {
            let (h, w, _) = frame.dim();
            if publisher.is_none() {
                let mut p = RtspPublisher::new(&cli.rtsp, w as u32, h as u32)?;
                p.start()?;
                publisher = Some(p);
            }
            publisher
                .as_mut()
                .expect("publisher was just created")
                .push_frame(&frame)?;
        }

        if !cli.verbose && processed % 90 == 1 {
            eprintln!("{processed} frames processed");
        }

        if let Some(limit) = frames {
            if processed >= limit {
                break Ok(());
            }
        }
    };
    if let Some(p) = publisher.as_mut() {
        p.stop();
    }
    runner.stop().ok();
    result
}

/// Runs the chain once on a still image.
fn analyze_image(
    cli: &Cli,
    models: &Models,
    gallery: &Arc<Mutex<Gallery>>,
    path: &PathBuf,
) -> Result<()> {
    let chain = Chain::new(models, Arc::clone(gallery), cli.threshold, overlay_opts(cli))?;
    let frame = load_image_frame(path)?;
    let (frame_w, frame_h) = (frame.shape()[1], frame.shape()[0]);

    eprintln!("analyzing {} ({}x{})", path.display(), frame_w, frame_h);
    chain.push_frame(frame)?;
    Ok(())
}

fn overlay_opts(cli: &Cli) -> OverlayOptions {
    OverlayOptions {
        save_dir: cli.save_dir.clone(),
        save_every: cli.save_every,
        verbose: cli.verbose,
    }
}