//! Live display window (minifb). The window is `!Send`, so it lives on the
//! main thread while pipeline callbacks feed it frames through a channel.

use anyhow::{Context, Result};
use minifb::{Key, ScaleMode, Window, WindowOptions};

use crate::face::{Array3U8, arr3_to_rgba32};

/// A minifb window that renders `Array3<u8>` RGB frames.
pub struct Viewer {
    window: Window,
    width: usize,
    height: usize,
}

impl Viewer {
    /// Opens a window sized to the first shown frame (`None` dimensions open
    /// it once a frame arrives).
    pub fn new(title: &str, width: usize, height: usize) -> Result<Self> {
        let window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: true,
                scale_mode: ScaleMode::AspectRatioStretch,
                ..Default::default()
            },
        )
        .with_context(|| format!("failed to open display window {title}"))?;
        Ok(Self {
            window,
            width,
            height,
        })
    }

    /// Draws a frame. Returns `false` when the window was closed (or the user
    /// pressed Escape), signalling the caller to stop showing frames.
    pub fn update(&mut self, frame: &Array3U8) -> Result<bool> {
        let pixels = arr3_to_rgba32(frame);
        self.width = frame.shape()[1];
        self.height = frame.shape()[0];
        self.window
            .update_with_buffer(&pixels, self.width, self.height)
            .with_context(|| "failed to update display window")?;

        Ok(self.window.is_open() && !self.window.is_key_down(Key::Escape))
    }

    /// Blocks briefly, pumping window events. Returns `false` when closed.
    pub fn pump(&mut self) -> Result<bool> {
        std::thread::sleep(std::time::Duration::from_millis(16));
        Ok(self.window.is_open() && !self.window.is_key_down(Key::Escape))
    }
}