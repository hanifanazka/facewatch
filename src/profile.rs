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

/// Canonical pipeline order of the profiled stages, as they run per frame.
/// Each chain element is timed individually; `align`/`embed` only run
/// when a face is detected, and stages that never run are omitted from the
/// report. Frame acquisition is not a profiled stage — `total` covers the
/// whole iteration.
const STAGE_ORDER: [&str; 6] = ["detect", "align", "embed", "draw", "publish", "total"];

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static FRAMES: RefCell<Vec<BTreeMap<&'static str, Duration>>> = RefCell::new(Vec::new());
    static CUR: RefCell<BTreeMap<&'static str, Duration>> = RefCell::new(BTreeMap::new());
}

/// Turns profiling on.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Turns profiling off and drops any recorded frames (used by tests to keep
/// the global flag isolated; harmless when profiling was already off).
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
    FRAMES.with(|frames| frames.borrow_mut().clear());
    CUR.with(|cur| cur.borrow_mut().clear());
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

        let mut seen: BTreeSet<&'static str> = BTreeSet::new();
        for frame in frames.iter() {
            seen.extend(frame.keys().copied());
        }
        // Wall-clock anchor for the `share` column: the summed `total` stage
        // across all frames (`total` is recorded every frame).
        let total_sum: Duration = frames
            .iter()
            .map(|f| f.get("total").copied().unwrap_or_default())
            .sum();

        let mut out = format!("profile summary over {} frames:\n", frames.len());
        out += &format!(
            "{:<8} {:>9} {:>9} {:>9} {:>9} {:>7}\n",
            "stage", "mean", "p50", "p90", "max", "share"
        );
        for stage in STAGE_ORDER {
            if !seen.contains(stage) {
                continue;
            }
            let mut vals: Vec<Duration> = frames
                .iter()
                .filter_map(|f| f.get(stage))
                .copied()
                .collect();
            vals.sort_unstable();
            let mean = vals.iter().sum::<Duration>() / vals.len() as u32;
            let p50 = vals[vals.len() / 2];
            let p90 = vals[(vals.len() * 9) / 10];
            let max = vals[vals.len() - 1];
            let total = vals.iter().sum::<Duration>();
            // Share of the whole run's wall-clock time (`total` reads 100%).
            let share = if total_sum.is_zero() {
                0.0
            } else {
                total.as_secs_f64() / total_sum.as_secs_f64() * 100.0
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

#[cfg(test)]
mod tests {
    use super::*;

    // `ENABLED` is process-global while `FRAMES`/`CUR` are thread-local, so
    // tests that touch the flag serialize on this lock to avoid stepping on
    // each other when running in parallel threads.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn disabled_is_a_noop_and_reports_no_frames() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        assert_eq!(time("detect", || 7), 7);
        record("detect", ms(5));
        frame_done();
        assert!(report().contains("(no frames profiled)"));
    }

    #[test]
    fn records_accumulate_within_a_frame() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        record("detect", ms(1));
        record("detect", ms(2));
        frame_done();
        let text = report();
        assert!(text.contains("profile summary over 1 frames"));
        assert!(text.contains("3.0ms"), "detect mean is the sum of both records: {text}");
        disable();
    }

    #[test]
    fn frame_done_starts_a_new_frame() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        record("detect", ms(4));
        frame_done();
        record("detect", ms(2));
        frame_done();
        let text = report();
        assert!(text.contains("profile summary over 2 frames"));
        assert!(text.contains("3.0ms"), "detect mean is 3.0ms across two frames: {text}");
        disable();
    }

    #[test]
    fn empty_frames_are_not_pushed() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        frame_done(); // nothing recorded yet
        assert!(report().contains("(no frames profiled)"));
        disable();
    }

    #[test]
    fn report_computes_mean_p50_p90_max() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        // One sample per frame: [1, 1, 10] -> mean 4.0, p50 1.0, p90 10.0,
        // max 10.0.
        record("detect", ms(1));
        frame_done();
        record("detect", ms(1));
        frame_done();
        record("detect", ms(10));
        frame_done();
        let text = report();
        assert!(text.contains("4.0ms"), "{text}");
        assert!(text.contains("1.0"), "{text}");
        assert!(text.contains("10.0"), "{text}");
        disable();
    }

    #[test]
    fn report_lists_stages_in_canonical_order_and_only_seen_ones() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        record("embed", ms(1));
        record("total", ms(2));
        frame_done();
        let text = report();
        let embed = text.find("embed").expect("embed stage present");
        let total = text.find("total").expect("total stage present");
        assert!(embed < total);
        assert!(!text.contains("detect"), "unseen stage omitted:\n{text}");
        assert!(!text.contains("align"), "unseen stage omitted:\n{text}");
        disable();
    }

    #[test]
    fn total_reports_full_share_and_other_stages_proportional() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        record("detect", ms(10));
        record("publish", ms(30));
        record("total", ms(40));
        frame_done();
        let text = report();
        assert!(text.contains("100.0%"), "total share is 100%:\n{text}");
        assert!(text.contains("25.0%"), "detect share 10/40:\n{text}");
        assert!(text.contains("75.0%"), "publish share 30/40:\n{text}");
        disable();
    }

    #[test]
    fn zero_total_sum_does_not_divide_by_zero() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        record("detect", ms(5));
        frame_done();
        let text = report();
        assert!(text.contains("0.0%"), "share guards zero total:\n{text}");
        disable();
    }

    #[test]
    fn disable_drops_all_recorded_frames() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable();
        enable();
        record("detect", ms(1));
        frame_done();
        assert!(!report().contains("(no frames profiled)"));
        disable();
        assert!(report().contains("(no frames profiled)"));
    }
}