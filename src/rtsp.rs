//! RTSP publishing: annotated RGB frames are pushed into an `appsrc` and
//! encoded/published through the operator-specified pipeline:
//!
//! ```text
//! appsrc name=mysource ! videoconvert ! video/x-raw,format=I420 !
//! x264enc speed-preset=ultrafast tune=zerolatency bitrate=2000
//!        cabac=false dct8x8=false key-int-max=1 !
//! rtspclientsink protocols=tcp location=<url> name=sink
//! ```
//!
//! Baseline compatibility: this machine's ffmpeg/mpv builds expose
//! `libopenh264` as their *only* H.264 decoder, and OpenH264 cannot decode
//! High/Main profile. x264 picks its SPS profile from the enabled feature set,
//! so `cabac=false dct8x8=false` (plus `tune=zerolatency`'s bframes=0) makes
//! x264 emit a Baseline stream that OpenH264 accepts.
//!
//! `key-int-max=1` makes every frame an IDR. The stream runs at ~1 fps, so a
//! client joining between keyframes would otherwise sit through a GOP of
//! forward-only slices that OpenH264 refuses (it cannot start mid-GOP, unlike
//! ffmpeg's native decoder) — with all-intra encoding any client starts
//! decoding on the very first frame.
//!
//! The sink targets a mediamtx (or any RTSP server) endpoint; clients watch
//! the stream with e.g. `mpv --profile=low-latency <url>`.

use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;

use crate::face::Array3U8;

/// Publishes annotated frames to an RTSP server (e.g. mediamtx).
pub struct RtspPublisher {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    url: String,
    started: bool,
    start: Instant,
}

impl RtspPublisher {
    /// Builds the `appsrc -> x264 -> rtspclientsink` pipeline for `url`
    /// (for example `rtsp://127.0.0.1:8554/facewatch`) and a `width` x
    /// `height` RGB stream. The pipeline is not started until [`start`]
    /// (Self::start) is called.
    pub fn new(url: &str, width: u32, height: u32) -> Result<Self> {
        gst::init().context("failed to initialize GStreamer")?;

        let pipeline_desc = format!(
            "appsrc name=mysource ! videoconvert ! video/x-raw,format=I420 ! \
             x264enc speed-preset=ultrafast tune=zerolatency bitrate=2000 \
             cabac=false dct8x8=false key-int-max=1 ! \
             rtspclientsink protocols=tcp location=\"{url}\" name=sink"
        );
        let pipeline = gst::parse::launch(&pipeline_desc)
            .with_context(|| format!("failed to build RTSP pipeline: {pipeline_desc}"))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow!("parsed RTSP pipeline is not a gst::Pipeline"))?;

        let appsrc = pipeline
            .by_name("mysource")
            .ok_or_else(|| anyhow!("RTSP pipeline has no 'mysource' appsrc"))?
            .downcast::<gst_app::AppSrc>()
            .map_err(|_| anyhow!("'mysource' is not an AppSrc"))?;

        // Annotated frames arrive as row-major RGB. Nominal 30 fps drives
        // x264 rate control; real cadence is carried by the PTS we stamp.
        let caps: gst::Caps = format!(
            "video/x-raw,format=RGB,width={width},height={height},framerate=30/1"
        )
        .parse()
        .context("invalid RGB caps string")?;
        appsrc.set_caps(Some(&caps));
        appsrc.set_format(gst::Format::Time);
        appsrc.set_stream_type(gst_app::AppStreamType::Stream);

        Ok(Self {
            pipeline,
            appsrc,
            url: url.to_owned(),
            started: false,
            start: Instant::now(),
        })
    }

    /// Starts the pipeline; does nothing if already started.
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| anyhow!("failed to set RTSP pipeline to Playing: {e}"))?;
        let (result, current, _) = self
            .pipeline
            .state(gst::ClockTime::from_seconds(15));
        result.map_err(|e| anyhow!("RTSP pipeline state change failed: {e}"))?;
        if current != gst::State::Playing {
            return Err(anyhow!(
                "RTSP pipeline did not reach Playing (current: {current:?}); is the RTSP server ({}) up?",
                self.url
            ));
        }
        self.started = true;
        self.drain_bus_messages();
        eprintln!(
            "publishing to {} (x264enc ultrafast/zerolatency baseline, 2000 kbps, tcp)",
            self.url
        );
        Ok(())
    }

    /// Pushes one RGB frame into the `appsrc`, stamping the current wall time
    /// as PTS so playback reflects real cadence.
    pub fn push_frame(&mut self, frame: &Array3U8) -> Result<()> {
        if !self.started {
            return Ok(());
        }
        self.drain_bus_messages();

        let data = frame
            .as_slice()
            .ok_or_else(|| anyhow!("frame is not contiguous"))?;
        let mut buffer = gst::Buffer::from_mut_slice(data.to_vec());
        let pts = gst::ClockTime::from_nseconds(self.start.elapsed().as_nanos() as u64);
        let buffer_mut = buffer
            .get_mut()
            .ok_or_else(|| anyhow!("buffer is not writable"))?;
        buffer_mut.set_pts(Some(pts));

        self.appsrc.push_buffer(buffer).map(|_| ()).map_err(|flow| {
            anyhow!(
                "appsrc rejected frame: {flow:?} (is the RTSP server still up?)"
            )
        })
    }

    /// Stops the pipeline; does nothing if already stopped.
    pub fn stop(&mut self) {
        if self.started {
            let _ = self.pipeline.set_state(gst::State::Null);
            self.started = false;
        }
    }

    /// Non-blockingly drains the pipeline bus, printing errors/warnings.
    fn drain_bus_messages(&mut self) {
        use gst::MessageView;
        if let Some(bus) = self.pipeline.bus() {
            while let Some(msg) = bus.timed_pop(gst::ClockTime::ZERO) {
                match msg.view() {
                    MessageView::Error(err) => {
                        eprintln!("RTSP pipeline error: {}", err.error());
                        if let Some(debug) = err.debug() {
                            eprintln!("  {debug}");
                        }
                    }
                    MessageView::Warning(warn) => {
                        eprintln!("RTSP pipeline warning: {}", warn.error());
                    }
                    MessageView::Eos(_) => eprintln!("RTSP pipeline reached end of stream"),
                    _ => {}
                }
            }
        }
    }
}

impl Drop for RtspPublisher {
    fn drop(&mut self) {
        self.stop();
    }
}