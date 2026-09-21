//! Payload types that flow between pupi pipeline stages, plus small helpers
//! for converting between `ndarray::Array3<u8>` frames and `image::RgbImage`.

use std::sync::Arc;

use image::RgbImage;
use ndarray::Array4;

/// Five 2D landmark points: left eye, right eye, nose, left mouth, right mouth.
pub type Kps = [[f32; 2]; 5];

/// A single detected face, in source-frame pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectedFace {
    /// `[x1, y1, x2, y2]` bounding box.
    pub bbox: [f32; 4],
    /// Detection confidence (0..1).
    pub score: f32,
    /// Five facial landmarks.
    pub kps: Kps,
}

/// Output of the detection stage: a source frame plus the faces found in it.
#[derive(Clone, Debug)]
pub struct FaceFrame {
    /// The RGB source frame shared with downstream stages.
    pub src: Arc<Array3U8>,
    pub faces: Vec<DetectedFace>,
}

/// A normalized face embedding (512 floats for AuraFace, L2-normalized).
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Vec<f32>);

/// Output of the recognition stage: an embedding for every detected face.
#[derive(Clone, Debug)]
pub struct FaceEmbeddings {
    pub src: Arc<Array3U8>,
    pub faces: Vec<DetectedFace>,
    pub embeddings: Vec<Embedding>,
}

/// The result of matching one embedding against the gallery.
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    /// The gallery subject name.
    pub name: String,
    /// Cosine similarity against the gallery entry.
    pub score: f32,
}

/// Output of the matching stage: a best gallery match (or `None`) per face.
#[derive(Clone, Debug)]
pub struct FaceMatches {
    pub src: Arc<Array3U8>,
    pub faces: Vec<DetectedFace>,
    pub matches: Vec<Option<Match>>,
}

/// Fresh, unaligned RGB frame payload (`(height, width, 3)`, row-major).
pub type Array3U8 = ndarray::Array3<u8>;

/// Converts an `Array3<u8>` frame to an `RgbImage`, copying the pixel data.
pub fn arr3_to_rgb(frame: &Array3U8) -> Option<RgbImage> {
    let height = frame.shape()[0];
    let width = frame.shape()[1];
    RgbImage::from_raw(width as u32, height as u32, frame.as_slice()?.to_vec())
}

/// Converts an `RgbImage` to an `Array3<u8>` frame.
pub fn rgb_to_arr3(img: &RgbImage) -> Array3U8 {
    let (width, height) = img.dimensions();
    Array3U8::from_shape_vec((height as usize, width as usize, 3), img.as_raw().to_vec())
        .expect("RGB image buffer is always exactly width*height*3 bytes")
}

/// A color used when drawing overlays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb([u8; 3]);

impl Rgb {
    pub const RED: Rgb = Rgb([255, 64, 64]);
    pub const GREEN: Rgb = Rgb([64, 255, 64]);
    pub const WHITE: Rgb = Rgb([255, 255, 255]);
    pub const BLACK: Rgb = Rgb([0, 0, 0]);

    pub fn array(self) -> [u8; 3] {
        self.0
    }
}

/// Copies an RGB frame into the top-left of a zero-initialized `NCHW` float
/// blob, normalizing pixels as `(pixel - 127.5) / denom`. `out` must be
/// `(1, 3, H, W)`; the frame is written at the top-left and the rest stays
/// zero, which is what both ONNX models expect as their input.
pub fn fill_nchw(frame: &Array3U8, denom: f32, out: &mut Array4<f32>) {
    let (height, width) = (frame.shape()[0], frame.shape()[1]);
    for y in 0..height {
        for x in 0..width {
            out[[0, 0, y, x]] = (frame[[y, x, 0]] as f32 - 127.5) / denom;
            out[[0, 1, y, x]] = (frame[[y, x, 1]] as f32 - 127.5) / denom;
            out[[0, 2, y, x]] = (frame[[y, x, 2]] as f32 - 127.5) / denom;
        }
    }
}

/// Resizes a frame to `(new_width, new_height)` using bilinear filtering and
/// returns a new frame.
pub fn resize_frame(frame: &Array3U8, new_width: usize, new_height: usize) -> Array3U8 {
    let img = arr3_to_rgb(frame).expect("frame has a valid RGB layout");
    let resized = image::imageops::resize(
        &img,
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::Triangle,
    );
    rgb_to_arr3(&resized)
}