//! facewatch: webcam face recognition on the pupi push-pipeline, using YuNet
//! (default) or SCRFD detection + AuraFace recognition through ONNX Runtime
//! (ort).

mod align;
mod aura;
mod sface;
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
use crate::sface::SFace;

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

    /// Face recognition backend: "auraface" (default) or "sface".
    #[arg(long, default_value = "auraface")]
    recognizer: String,

    /// ONNX intra-op threads per model session (default: logical CPU count).
    #[arg(long)]
    threads: Option<usize>,

    /// Detector input size in pixels: 320 (faster, lower resolution) or 640 (default).
    #[arg(long, default_value_t = crate::face::DEFAULT_DETECT_SIZE)]
    input_size: usize,

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
    validate_input_size(cli.input_size)?;
    let threads = model_threads(&cli);

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
                threads,
                cli.input_size,
            )
        }
        Command::Run { frames, source } => {
            run(&cli, &models, &gallery, *frames, source, threads)
        }
        Command::Image { path } => analyze_image(&cli, &models, &gallery, path, threads),
    }
}

/// Effective ONNX intra-op thread count per model session: `--threads`, else
/// the logical CPU count (fallback 2 when it cannot be determined).
fn model_threads(cli: &Cli) -> usize {
    cli.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
    })
}

/// Rejects detector input sizes the embedded/pipeline supports (320 or 640).
fn validate_input_size(input_size: usize) -> Result<()> {
    anyhow::ensure!(
        matches!(input_size, 320 | 640),
        "--input-size must be 320 or 640, got {input_size}"
    );
    Ok(())
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
    threads: usize,
    input_size: usize,
) -> Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("registration name must not be empty");
    }

    let mut detector = Scrfd::load(models.scrfd, threads, input_size)?;
    let mut aura = AuraFace::load(models.auraface, threads)?;
    let mut sface = SFace::load(models.sface, threads)?;

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
    let embedding_aura = aura.embed(&crop)?;
    let embedding_sface = sface.embed(&crop)?;

    let mut gallery = gallery.lock().map_err(|_| anyhow!("gallery mutex poisoned"))?;
    gallery.register(name, embedding_aura, embedding_sface);
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
    threads: usize,
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
        &cli.recognizer,
        threads,
        cli.input_size,
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
    threads: usize,
) -> Result<()> {
    let chain = Chain::new(
        models,
        Arc::clone(gallery),
        cli.threshold,
        overlay_opts(cli),
        &cli.detector,
        &cli.recognizer,
        threads,
        cli.input_size,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::sync::Mutex;

    // Tests that mutate process-global state (HOME, or the profile flag) must
    // not race with each other or with the app's other tests.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn run_cli(args: &[&str]) -> Cli {
        let mut argv = vec!["facewatch"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).expect("CLI parses")
    }

    #[test]
    fn cli_config_is_debug_assertable() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_parses_defaults() {
        let cli = run_cli(&["run"]);
        assert_eq!(cli.threshold, 0.40);
        assert_eq!(cli.detector, "yunet");
        assert_eq!(cli.recognizer, "auraface");
        assert_eq!(cli.input_size, crate::face::DEFAULT_DETECT_SIZE);
        assert_eq!(cli.save_every, 1);
        assert_eq!(cli.rtsp, "rtsp://127.0.0.1:8554/facewatch");
        assert!(!cli.no_rtsp);
        assert!(!cli.verbose);
        assert!(!cli.profile);
        assert!(matches!(cli.command, Command::Run { frames: None, source: None }));
    }

    #[test]
    fn cli_parses_global_flags_and_run_options() {
        let cli = run_cli(&[
            "--threshold",
            "0.7",
            "--detector",
            "scrfd",
            "--recognizer",
            "sface",
            "--threads",
            "4",
            "--input-size",
            "320",
            "--gallery",
            "/tmp/g.json",
            "--no-rtsp",
            "--save-dir",
            "/tmp/saved",
            "--save-every",
            "5",
            "--verbose",
            "--profile",
            "run",
            "--frames",
            "10",
            "--source",
            "image:/tmp/photo.jpg",
        ]);
        assert_eq!(cli.threshold, 0.7);
        assert_eq!(cli.detector, "scrfd");
        assert_eq!(cli.recognizer, "sface");
        assert_eq!(cli.threads, Some(4));
        assert_eq!(cli.input_size, 320);
        assert_eq!(cli.gallery, Some(PathBuf::from("/tmp/g.json")));
        assert!(cli.no_rtsp);
        assert_eq!(cli.save_dir, Some(PathBuf::from("/tmp/saved")));
        assert_eq!(cli.save_every, 5);
        assert!(cli.verbose);
        assert!(cli.profile);
        match cli.command {
            Command::Run { frames, source } => {
                assert_eq!(frames, Some(10));
                assert_eq!(source.as_deref(), Some("image:/tmp/photo.jpg"));
            }
            _ => panic!("expected Run subcommand"),
        }
    }

    #[test]
    fn cli_parses_register_and_image_subcommands() {
        match run_cli(&["register", "bob", "photo.jpg"]).command {
            Command::Register { name, image } => {
                assert_eq!(name, "bob");
                assert_eq!(image, Some(PathBuf::from("photo.jpg")));
            }
            _ => panic!("expected Register"),
        }
        match run_cli(&["register", "bob"]).command {
            Command::Register { image, .. } => assert!(image.is_none()),
            _ => panic!("expected Register"),
        }
        match run_cli(&["image", "x.png"]).command {
            Command::Image { path } => assert_eq!(path, PathBuf::from("x.png")),
            _ => panic!("expected Image"),
        }
    }

    #[test]
    fn validate_input_size_accepts_supported_sizes() {
        validate_input_size(320).unwrap();
        validate_input_size(640).unwrap();
        let err = validate_input_size(512).unwrap_err();
        assert!(err.to_string().contains("--input-size must be 320 or 640"), "{err}");
    }

    #[test]
    fn to_uri_passes_through_uris_and_builds_file_uris() {
        assert_eq!(to_uri("rtsp://127.0.0.1:8554/cam"), "rtsp://127.0.0.1:8554/cam");
        assert_eq!(to_uri("file:///abs/v.mp4"), "file:///abs/v.mp4");
        assert_eq!(to_uri("/abs/path.mp4"), "file:///abs/path.mp4");
        let expected = format!("file://{}", std::env::current_dir().unwrap().join("rel.mp4").display());
        assert_eq!(to_uri("rel.mp4"), expected);
    }

    #[test]
    fn default_gallery_uses_home_dot_facewatch() {
        let _g = ENV_LOCK.lock().unwrap();
        let old = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", "/home/facewatch-tester") };
        let p = default_gallery();
        assert_eq!(p, PathBuf::from("/home/facewatch-tester/.facewatch/gallery.json"));
        match old {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }

    #[test]
    fn default_gallery_falls_back_to_cwd_when_home_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        let old = std::env::var_os("HOME");
        unsafe { std::env::remove_var("HOME") };
        let p = default_gallery();
        assert_eq!(p, PathBuf::from(".").join(".facewatch").join("gallery.json"));
        match old {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => {}
        }
    }

    #[test]
    fn model_threads_uses_explicit_or_available_parallelism() {
        let with_threads = run_cli(&["--threads", "3", "run"]);
        assert_eq!(model_threads(&with_threads), 3);
        let default = run_cli(&["run"]);
        let expected = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        assert_eq!(model_threads(&default), expected);
    }

    #[test]
    fn load_image_frame_roundtrips_a_png() {
        let img = image::RgbImage::from_fn(2, 2, |x, y| {
            if x == 0 && y == 0 {
                image::Rgb([255u8, 0, 0])
            } else {
                image::Rgb([0u8, 0, 0])
            }
        });
        let path = std::env::temp_dir().join(format!("facewatch-frame-{}.png", std::process::id()));
        img.save(&path).expect("saves a png");
        let frame = load_image_frame(&path).expect("loads the png");
        assert_eq!(frame.shape(), &[2, 2, 3]);
        assert_eq!(frame[[0, 0, 0]], 255);
        assert_eq!(frame[[0, 0, 1]], 0);
        assert_eq!(frame[[0, 0, 2]], 0);
        assert_eq!(frame[[1, 1, 0]], 0);
        std::fs::remove_file(&path).ok();

        let err = load_image_frame(&PathBuf::from("/nonexistent/facewatch-missing.png"))
            .unwrap_err();
        assert!(err.to_string().contains("failed to open image"), "{err}");
    }

    #[test]
    fn overlay_opts_copies_cli_fields() {
        let cli = run_cli(&["--save-dir", "/tmp/out", "--save-every", "3", "--verbose", "run"]);
        let opts = overlay_opts(&cli);
        assert_eq!(opts.save_dir, Some(PathBuf::from("/tmp/out")));
        assert_eq!(opts.save_every, 3);
        assert!(opts.verbose);
    }

    #[test]
    fn register_rejects_empty_names_before_loading_models() {
        // The empty-name guard runs before any model load, so a dummy Models
        // is enough to exercise it.
        let models = Models {
            scrfd: &[],
            yunet: &[],
            auraface: &[],
            sface: &[],
        };
        let gallery_path = std::env::temp_dir().join(format!("facewatch-gal-{}.json", std::process::id()));
        let gallery = Arc::new(Mutex::new(Gallery::default()));
        let err = register(
            "   ",
            None,
            &models,
            &gallery_path,
            &gallery,
            None,
            1,
            640,
        )
        .unwrap_err();
        assert!(err.to_string().contains("must not be empty"), "{err}");
    }
}