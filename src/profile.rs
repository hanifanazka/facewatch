//! Lightweight per-stage profiling for the synchronous facewatch pipeline.
//!
//! Enabled with `--profile`. Pipeline stages run on a single thread, so a
//! thread-local map of stage names to accumulated durations is enough:
//! [`time`]/[`record`] add to the current frame, [`frame_done`] closes it,
//! and [`report`] prints mean/p50/p90/max across all frames.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static FRAMES: RefCell<Vec<BTreeMap<&'static str, Duration>>> = RefCell::new(Vec::new());
    static CUR: RefCell<BTreeMap<&'static str, Duration>> = RefCell::new(BTreeMap::new());
}

/// Turns profiling on.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Adds `d` to stage `stage` of the current frame.
pub fn record(stage: &'static str, d: Duration) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    CUR.with(|cur| {
        let mut cur = cur.borrow_mut();
        *cur.entry(stage).or_insert(Duration::ZERO) += d;
    });
}

/// Runs `f`, timing it as stage `stage` (summed across calls within a frame).
pub fn time<R>(stage: &'static str, f: impl FnOnce() -> R) -> R {
    if !ENABLED.load(Ordering::Relaxed) {
        return f();
    }
    let start = Instant::now();
    let result = f();
    record(stage, start.elapsed());
    result
}

/// Closes the current frame and starts the next one.
pub fn frame_done() {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    FRAMES.with(|frames| {
        let cur = CUR.with(|c| c.replace(BTreeMap::new()));
        if !cur.is_empty() {
            frames.borrow_mut().push(cur);
        }
    });
}

/// Formats a mean/p50/p90/max summary over all recorded frames.
pub fn report() -> String {
    FRAMES.with(|frames| {
        let frames = frames.borrow();
        if frames.is_empty() {
            return "(no frames profiled)".to_owned();
        }

        let mut stages: BTreeSet<&'static str> = BTreeSet::new();
        let mut total_all = Duration::ZERO;
        for frame in frames.iter() {
            stages.extend(frame.keys().copied());
            total_all += frame.values().sum::<Duration>();
        }

        let mut out = format!("profile summary over {} frames:\n", frames.len());
        out += &format!(
            "{:<8} {:>9} {:>9} {:>9} {:>9} {:>7}\n",
            "stage", "mean", "p50", "p90", "max", "share"
        );
        for stage in stages {
            let mut vals: Vec<Duration> = frames
                .iter()
                .filter_map(|f| f.get(stage))
                .copied()
                .collect();
            vals.sort_unstable();
            if vals.is_empty() {
                continue;
            }
            let mean = vals.iter().sum::<Duration>() / vals.len() as u32;
            let p50 = vals[vals.len() / 2];
            let p90 = vals[(vals.len() * 9) / 10];
            let max = vals[vals.len() - 1];
            let total = vals.iter().sum::<Duration>();
            let share = if total_all.is_zero() {
                0.0
            } else {
                total.as_secs_f64() / total_all.as_secs_f64() * 100.0
            };
            out += &format!(
                "{stage:<8} {:>8.1}ms {:>8.1}  {:>8.1}  {:>8.1}  {:>5.1}%\n",
                mean.as_secs_f64() * 1e3,
                p50.as_secs_f64() * 1e3,
                p90.as_secs_f64() * 1e3,
                max.as_secs_f64() * 1e3,
                share,
            );
        }
        out
    })
}