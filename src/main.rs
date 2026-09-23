//! facewatch: webcam face recognition on the pupi push-pipeline, using YuNet
//! (default) or SCRFD detection + AuraFace recognition through ONNX Runtime
//! (ort).

mod align;
mod aura;
mod draw;
mod elements;
mod face;
mod gallery;
mod models;
mod profile;
mod rtsp;
mod scrfd;
mod yunet;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use pupi::{GstreamRunner, GstreamRunnerError};

use crate::aura::AuraFace;
use crate::elements::{Chain, OverlayOptions};
use crate::face::Array3U8;
use crate::gallery::Gallery;
use crate::models::Models;
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
    about = "Webcam face recognition on the pupi push-pipeline (YuNet/SCRFD + AuraFace, ONNX Runtime)."
)]
struct Cli {
    /// Minimum cosine similarity required for a gallery match.
    #[arg(long, default_value_t = 0.40)]
    threshold: f32,

    /// Face detector backend: "yunet" (default) or "scrfd".
    #[arg(long, default_value = "yunet")]
    detector: String,

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

    /// Print a per-frame stage timing summary (mean/p50/p90/max) when a run ends.
    #[arg(long)]
    profile: bool,

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
    /// Run face recognition from the webcam, a media URI, or a still image.
    Run {
        /// Frame source: a file/RTSP URI (e.g. `file:///path/v.mp4`), a bare
        /// local path, `image:<path>` to loop a still image, or the webcam by
        /// default.
        #[arg(long)]
        source: Option<String>,

        /// Stop after processing this many frames (~0.5-2 fps on this box).
        /// Applies to live sources too: the webcam and URI paths otherwise run
        /// until ^C (or end of stream for files).
        #[arg(long)]
        frames: Option<u64>,
    },
    /// Recognize faces in a single image file.
    Image {
        /// Image file to analyze.
        path: PathBuf,
    },
}

/// Set when the process receives SIGINT (^C). The run loop checks it between
/// frames, so even an unbounded live session stops cleanly and prints its
/// `--profile` report instead of being killed without one.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Graceful ^C handling: the loop breaks as if it hit its natural end.
    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .context("failed to install ^C handler")?;

    // The ONNX models are embedded in the binary by build.rs (verified via
    // SHA-256 at compile time), so there is no runtime download or lookup.
    let models = Models::embedded();
    let gallery_path = cli.gallery.clone().unwrap_or_else(default_gallery);
    let gallery = Arc::new(Mutex::new(Gallery::load(&gallery_path)?));

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
        Command::Run { frames, source } => run(&cli, &models, &gallery, *frames, source),
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

    let mut detector = Scrfd::load(models.scrfd)?;
    let mut aura = AuraFace::load(models.auraface)?;

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

/// Where the next frame comes from while driving the chain.
enum FrameSource {
    /// Live GStreamer source (webcam or file/RTSP URI); each `pull_frame`
    /// pushes synchronously into the chain.
    Gst(GstreamRunner),
    /// A single still image looped through the chain (no GStreamer needed).
    Stills(Array3U8),
}

/// Turns a `--source` argument into a URI for `uridecodebin`: already-URIs
/// pass through, bare paths become absolute `file://` URIs.
fn to_uri(source: &str) -> String {
    if source.contains("://") {
        return source.to_owned();
    }
    let path = Path::new(source);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|dir| dir.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    format!("file://{}", path.display())
}

/// Builds the linked chain and drives it from the webcam, a `--source` URI, or
/// a looped still image. With `--profile`, per-frame stage timings are
/// accumulated and a summary is printed when the run ends.
fn run(
    cli: &Cli,
    models: &Models,
    gallery: &Arc<Mutex<Gallery>>,
    frames: Option<u64>,
    source: &Option<String>,
) -> Result<()> {
    if cli.profile {
        profile::enable();
    }

    let chain = Chain::new(
        models,
        Arc::clone(gallery),
        cli.threshold,
        overlay_opts(cli),
        &cli.detector,
    )?;

    let source = match source {
        Some(s) if s.starts_with("image:") => {
            let path = s.trim_start_matches("image:");
            let frame = load_image_frame(&PathBuf::from(path))?;
            anyhow::ensure!(
                frames.is_some(),
                "a still-image source needs --frames so the loop can end"
            );
            eprintln!("driving pipeline from still image {path} (looped)");
            FrameSource::Stills(frame)
        }
        Some(s) => {
            let uri = to_uri(s);
            // Decode as fast as possible: the runner's appsink keeps only the
            // newest frame (max-buffers=1, drop=true, sync=false), so a slow
            // chain consumes the freshest frame instead of a stale backlog.
            let runner = GstreamRunner::uri(&uri)?;
            eprintln!("driving pipeline from {uri}");
            FrameSource::Gst(runner)
        }
        None => {
            eprintln!("driving pipeline from GStreamer webcam source");
            FrameSource::Gst(GstreamRunner::webcam()?)
        }
    };

    if let FrameSource::Gst(runner) = &source {
        elements::link_upstream(runner, &chain.detection)?;
        elements::negotiate_upstream(runner)?;
        runner.start_pipeline()?;
    }

    if cli.save_dir.is_some() {
        eprintln!(
            "saving annotated frames to {}",
            cli.save_dir.as_ref().unwrap().display()
        );
    }

    let mut publisher: Option<RtspPublisher> = None;

    let mut processed: u64 = 0;
    let result = loop {
        let iter_start = Instant::now();

        // Bound the run: `--frames` applies to every source (the webcam and
        // file-URI paths have no natural end), and a ^C interrupt stops the
        // loop between frames so the --profile report still prints.
        if INTERRUPTED.load(Ordering::SeqCst)
            || matches!(frames, Some(limit) if processed >= limit)
        {
            break Ok(());
        }

        // Frame acquisition. On GStreamer sources the whole synchronous chain
        // runs inside pull_frame with each element profiled individually; on
        // stills the push happens explicitly below. Acquisition itself is not
        // a profiled stage — `total` covers the whole iteration.
        match &source {
            FrameSource::Gst(runner) => match runner.pull_frame() {
                Ok(Some(_)) => {}
                Ok(None) => break Ok(()), // end of stream
                Err(GstreamRunnerError::PullSample(_)) => continue,
                Err(e) => break Err(anyhow!("frame pull failed: {e}")),
            },
            FrameSource::Stills(frame) => {
                chain.push_frame(frame.clone())?;
            }
        }
        processed += 1;

        // Drain the annotated-frame channel (keeping the newest frame) and
        // publish it over RTSP. The publisher is created lazily from the
        // first frame's dimensions.
        profile::time("publish", || {
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
            Ok::<(), anyhow::Error>(())
        })?;

        profile::record("total", iter_start.elapsed());
        profile::frame_done();

        if !cli.verbose && processed % 90 == 1 {
            eprintln!("{processed} frames processed");
        }
    };
    if let Some(p) = publisher.as_mut() {
        p.stop();
    }
    if let FrameSource::Gst(runner) = &source {
        runner.stop().ok();
    }

    if cli.profile {
        eprintln!("{}", profile::report());
    }
    result
}

/// Runs the chain once on a still image.
fn analyze_image(
    cli: &Cli,
    models: &Models,
    gallery: &Arc<Mutex<Gallery>>,
    path: &PathBuf,
) -> Result<()> {
    let chain = Chain::new(
        models,
        Arc::clone(gallery),
        cli.threshold,
        overlay_opts(cli),
        &cli.detector,
    )?;
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