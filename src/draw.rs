//! Minimal raster drawing helpers: rectangles, landmarks, and a 5x7 bitmap
//! font (public domain glyphs) for labels on the overlay.

use image::{Rgb, RgbImage};

use crate::face::{Kps, Rgb as FaceRgb};

/// One 5x7 glyph as 5 column bytes; bit 0 is the top row of the column.
/// Indexed by `char_code - 0x20` for ASCII 0x20..=0x7e.
pub const FONT_5X7: [u8; 475] = [
    0x00, 0x00, 0x00, 0x00, 0x00, // 0x20 space
    0x00, 0x00, 0x5F, 0x00, 0x00, // 0x21 !
    0x00, 0x07, 0x00, 0x07, 0x00, // 0x22 "
    0x14, 0x7F, 0x14, 0x7F, 0x14, // 0x23 #
    0x24, 0x2A, 0x7F, 0x2A, 0x12, // 0x24 $
    0x23, 0x13, 0x08, 0x64, 0x62, // 0x25 %
    0x36, 0x49, 0x55, 0x22, 0x50, // 0x26 &
    0x00, 0x05, 0x03, 0x00, 0x00, // 0x27 '
    0x00, 0x1C, 0x22, 0x41, 0x00, // 0x28 (
    0x00, 0x41, 0x22, 0x1C, 0x00, // 0x29 )
    0x08, 0x2A, 0x1C, 0x2A, 0x08, // 0x2A *
    0x08, 0x08, 0x3E, 0x08, 0x08, // 0x2B +
    0x00, 0x50, 0x30, 0x00, 0x00, // 0x2C ,
    0x08, 0x08, 0x08, 0x08, 0x08, // 0x2D -
    0x00, 0x60, 0x60, 0x00, 0x00, // 0x2E .
    0x20, 0x10, 0x08, 0x04, 0x02, // 0x2F /
    0x3E, 0x51, 0x49, 0x45, 0x3E, // 0x30 0
    0x00, 0x42, 0x7F, 0x40, 0x00, // 0x31 1
    0x42, 0x61, 0x51, 0x49, 0x46, // 0x32 2
    0x21, 0x41, 0x45, 0x4B, 0x31, // 0x33 3
    0x18, 0x14, 0x12, 0x7F, 0x10, // 0x34 4
    0x27, 0x45, 0x45, 0x45, 0x39, // 0x35 5
    0x3C, 0x4A, 0x49, 0x49, 0x30, // 0x36 6
    0x01, 0x71, 0x09, 0x05, 0x03, // 0x37 7
    0x36, 0x49, 0x49, 0x49, 0x36, // 0x38 8
    0x06, 0x49, 0x49, 0x29, 0x1E, // 0x39 9
    0x00, 0x36, 0x36, 0x00, 0x00, // 0x3A :
    0x00, 0x56, 0x36, 0x00, 0x00, // 0x3B ;
    0x00, 0x08, 0x14, 0x22, 0x41, // 0x3C <
    0x14, 0x14, 0x14, 0x14, 0x14, // 0x3D =
    0x41, 0x22, 0x14, 0x08, 0x00, // 0x3E >
    0x02, 0x01, 0x51, 0x09, 0x06, // 0x3F ?
    0x32, 0x49, 0x79, 0x41, 0x3E, // 0x40 @
    0x7E, 0x11, 0x11, 0x11, 0x7E, // 0x41 A
    0x7F, 0x49, 0x49, 0x49, 0x36, // 0x42 B
    0x3E, 0x41, 0x41, 0x41, 0x22, // 0x43 C
    0x7F, 0x41, 0x41, 0x22, 0x1C, // 0x44 D
    0x7F, 0x49, 0x49, 0x49, 0x41, // 0x45 E
    0x7F, 0x09, 0x09, 0x01, 0x01, // 0x46 F
    0x3E, 0x41, 0x41, 0x51, 0x32, // 0x47 G
    0x7F, 0x08, 0x08, 0x08, 0x7F, // 0x48 H
    0x00, 0x41, 0x7F, 0x41, 0x00, // 0x49 I
    0x20, 0x40, 0x41, 0x3F, 0x01, // 0x4A J
    0x7F, 0x08, 0x14, 0x22, 0x41, // 0x4B K
    0x7F, 0x40, 0x40, 0x40, 0x40, // 0x4C L
    0x7F, 0x02, 0x04, 0x02, 0x7F, // 0x4D M
    0x7F, 0x04, 0x08, 0x10, 0x7F, // 0x4E N
    0x3E, 0x41, 0x41, 0x41, 0x3E, // 0x4F O
    0x7F, 0x09, 0x09, 0x09, 0x06, // 0x50 P
    0x3E, 0x41, 0x51, 0x21, 0x5E, // 0x51 Q
    0x7F, 0x09, 0x19, 0x29, 0x46, // 0x52 R
    0x46, 0x49, 0x49, 0x49, 0x31, // 0x53 S
    0x01, 0x01, 0x7F, 0x01, 0x01, // 0x54 T
    0x3F, 0x40, 0x40, 0x40, 0x3F, // 0x55 U
    0x1F, 0x20, 0x40, 0x20, 0x1F, // 0x56 V
    0x7F, 0x20, 0x18, 0x20, 0x7F, // 0x57 W
    0x63, 0x14, 0x08, 0x14, 0x63, // 0x58 X
    0x03, 0x04, 0x78, 0x04, 0x03, // 0x59 Y
    0x61, 0x51, 0x49, 0x45, 0x43, // 0x5A Z
    0x00, 0x00, 0x7F, 0x41, 0x41, // 0x5B [
    0x02, 0x04, 0x08, 0x10, 0x20, // 0x5C backslash
    0x41, 0x41, 0x7F, 0x00, 0x00, // 0x5D ]
    0x04, 0x02, 0x01, 0x02, 0x04, // 0x5E ^
    0x40, 0x40, 0x40, 0x40, 0x40, // 0x5F _
    0x00, 0x01, 0x02, 0x04, 0x00, // 0x60 `
    0x20, 0x54, 0x54, 0x54, 0x78, // 0x61 a
    0x7F, 0x48, 0x44, 0x44, 0x38, // 0x62 b
    0x38, 0x44, 0x44, 0x44, 0x20, // 0x63 c
    0x38, 0x44, 0x44, 0x48, 0x7F, // 0x64 d
    0x38, 0x54, 0x54, 0x54, 0x18, // 0x65 e
    0x08, 0x7E, 0x09, 0x01, 0x02, // 0x66 f
    0x0C, 0x52, 0x52, 0x52, 0x3E, // 0x67 g
    0x7F, 0x08, 0x04, 0x04, 0x78, // 0x68 h
    0x00, 0x44, 0x7D, 0x40, 0x00, // 0x69 i
    0x20, 0x40, 0x44, 0x3D, 0x00, // 0x6A j
    0x00, 0x7F, 0x10, 0x28, 0x44, // 0x6B k
    0x00, 0x41, 0x7F, 0x40, 0x00, // 0x6C l
    0x7C, 0x04, 0x18, 0x04, 0x78, // 0x6D m
    0x7C, 0x08, 0x04, 0x04, 0x78, // 0x6E n
    0x38, 0x44, 0x44, 0x44, 0x38, // 0x6F o
    0x7C, 0x14, 0x14, 0x14, 0x08, // 0x70 p
    0x08, 0x14, 0x14, 0x18, 0x7C, // 0x71 q
    0x7C, 0x08, 0x04, 0x04, 0x08, // 0x72 r
    0x48, 0x54, 0x54, 0x54, 0x20, // 0x73 s
    0x04, 0x3F, 0x44, 0x40, 0x20, // 0x74 t
    0x3C, 0x40, 0x40, 0x20, 0x7C, // 0x75 u
    0x1C, 0x20, 0x40, 0x20, 0x1C, // 0x76 v
    0x3C, 0x40, 0x30, 0x40, 0x3C, // 0x77 w
    0x44, 0x28, 0x10, 0x28, 0x44, // 0x78 x
    0x0C, 0x50, 0x50, 0x50, 0x3C, // 0x79 y
    0x44, 0x64, 0x54, 0x4C, 0x44, // 0x7A z
    0x00, 0x08, 0x36, 0x41, 0x00, // 0x7B {
    0x00, 0x00, 0x7F, 0x00, 0x00, // 0x7C |
    0x00, 0x41, 0x36, 0x08, 0x00, // 0x7D }
    0x08, 0x04, 0x08, 0x10, 0x08, // 0x7E ~
];

/// Character cell width (glyph width + 1 spacer column).
pub const CHAR_W: u32 = 6;

fn color(c: FaceRgb) -> Rgb<u8> {
    Rgb(c.array())
}

/// Sets a single pixel, silently ignoring out-of-bounds writes.
pub fn set_pixel(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

/// Draws an axis-aligned rectangle outline with the given thickness.
pub fn draw_rect(img: &mut RgbImage, x1: i32, y1: i32, x2: i32, y2: i32, c: FaceRgb, thickness: u32) {
    let t = thickness.max(1) as i32;
    for i in 0..t {
        draw_line(img, x1 + i, y1 + i, x2 - i, y1 + i, c); // top
        draw_line(img, x1 + i, y2 - i, x2 - i, y2 - i, c); // bottom
        draw_line(img, x1 + i, y1 + i, x1 + i, y2 - i, c); // left
        draw_line(img, x2 - i, y1 + i, x2 - i, y2 - i, c); // right
    }
}

/// Fills an axis-aligned rectangle.
pub fn fill_rect(img: &mut RgbImage, x1: i32, y1: i32, x2: i32, y2: i32, c: FaceRgb) {
    let (x0, x1) = (x1.min(x2), x1.max(x2));
    let (y0, y1) = (y1.min(y2), y1.max(y2));
    for x in x0..=x1 {
        for y in y0..=y1 {
            set_pixel(img, x, y, color(c));
        }
    }
}

/// Bresenham line.
pub fn draw_line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, c: FaceRgb) {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        set_pixel(img, x, y, color(c));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draws the five facial landmarks as small cross markers.
pub fn draw_kps(img: &mut RgbImage, kps: &Kps, c: FaceRgb, r: i32) {
    for pt in kps {
        let (px, py) = (pt[0].round() as i32, pt[1].round() as i32);
        draw_line(img, px - r, py, px + r, py, c);
        draw_line(img, px, py - r, px, py + r, c);
    }
}

/// Returns the pixel width of `text` at the 5x7 font.
pub fn text_width(text: &str) -> u32 {
    text.chars().count() as u32 * CHAR_W
}

/// Draws `text` starting at `(x, y)` (top-left of the first glyph).
pub fn draw_text(img: &mut RgbImage, x: i32, y: i32, text: &str, c: FaceRgb) {
    for (i, ch) in text.chars().enumerate() {
        let code = ch as usize;
        let glyph = if (0x20..=0x7e).contains(&code) {
            &FONT_5X7[(code - 0x20) * 5..(code - 0x20) * 5 + 5]
        } else {
            &FONT_5X7[0..5]
        };
        let ox = x + i as i32 * CHAR_W as i32;
        for col in 0..5 {
            let column = glyph[col];
            for row in 0..7 {
                // Bit 0 is the top row of the glyph.
                if (column >> row) & 1 == 1 {
                    set_pixel(img, ox + col as i32, y + row as i32, color(c));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white() -> Rgb<u8> {
        Rgb([255, 255, 255])
    }
    fn red() -> Rgb<u8> {
        Rgb([255, 0, 0])
    }

    #[test]
    fn renders_a_known_glyph() {
        // 'A' = 0x7E 0x11 0x11 0x11 0x7E; bit 6 is the top row.
        let glyph = FONT_5X7[('A' as usize - 0x20) * 5];
        assert_eq!(glyph, 0x7E);
        let mut img = RgbImage::new(5, 7);
        for row in 0..7 {
            if (glyph >> (6 - row)) & 1 == 1 {
                img.put_pixel(0, row, Rgb([255, 255, 255]));
            }
        }
        // 0x7E = bits 1..=6 -> rows 0..=5 in column 0.
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(0, 5).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(0, 6).0, [0, 0, 0]);
    }

    #[test]
    fn set_pixel_ignores_out_of_bounds() {
        let mut img = RgbImage::new(10, 10);
        set_pixel(&mut img, -5, -5, red());
        set_pixel(&mut img, 10, 10, red());
        set_pixel(&mut img, 9, 9, white());
        assert_eq!(img.get_pixel(9, 9).0, [255, 255, 255]);
        // Image untouched elsewhere stays black.
        assert_eq!(img.get_pixel(5, 5).0, [0, 0, 0]);
    }

    #[test]
    fn draw_rect_paints_outline_only() {
        let mut img = RgbImage::new(20, 20);
        draw_rect(&mut img, 2, 2, 7, 7, FaceRgb::RED, 1);
        assert_eq!(img.get_pixel(2, 2).0, [255, 64, 64]); // corner
        assert_eq!(img.get_pixel(7, 7).0, [255, 64, 64]);
        assert_eq!(img.get_pixel(2, 5).0, [255, 64, 64]); // left edge
        assert_eq!(img.get_pixel(5, 2).0, [255, 64, 64]); // top edge
        assert_eq!(img.get_pixel(4, 4).0, [0, 0, 0]); // interior untouched
    }

    #[test]
    fn draw_rect_thickness_widens_the_band() {
        let mut img = RgbImage::new(20, 20);
        draw_rect(&mut img, 2, 2, 8, 8, FaceRgb::RED, 2);
        // Inner stroke ring painted (thickness >= 2).
        assert_eq!(img.get_pixel(3, 3).0, [255, 64, 64]);
        // Deep interior (5,5) of the 2..8 box stays clean.
        assert_eq!(img.get_pixel(5, 5).0, [0, 0, 0]);
    }

    #[test]
    fn fill_rect_paints_including_inverted_coords() {
        let mut img = RgbImage::new(20, 20);
        fill_rect(&mut img, 3, 3, 6, 6, FaceRgb::GREEN);
        fill_rect(&mut img, 12, 12, 10, 10, FaceRgb::WHITE); // inverted
        assert_eq!(img.get_pixel(3, 3).0, [64, 255, 64]);
        assert_eq!(img.get_pixel(6, 6).0, [64, 255, 64]);
        assert_eq!(img.get_pixel(2, 2).0, [0, 0, 0]);
        assert_eq!(img.get_pixel(10, 10).0, [255, 255, 255]);
        assert_eq!(img.get_pixel(12, 12).0, [255, 255, 255]);
    }

    #[test]
    fn draw_line_renders_diagonal_horizontal_and_point() {
        let mut img = RgbImage::new(10, 10);
        draw_line(&mut img, 0, 0, 4, 4, FaceRgb::RED);
        for i in 0..=4 {
            assert_eq!(img.get_pixel(i, i).0, [255, 64, 64]);
        }
        draw_line(&mut img, 0, 8, 6, 8, FaceRgb::GREEN);
        for x in 0..=6 {
            assert_eq!(img.get_pixel(x, 8).0, [64, 255, 64]);
        }
        draw_line(&mut img, 9, 0, 9, 0, FaceRgb::WHITE);
        assert_eq!(img.get_pixel(9, 0).0, [255, 255, 255]);
    }

    #[test]
    fn draw_kps_marks_crosses_at_each_landmark() {
        let mut img = RgbImage::new(30, 30);
        let kps: Kps = [[10.0, 10.0], [20.0, 10.0], [15.0, 18.0], [10.0, 24.0], [20.0, 24.0]];
        draw_kps(&mut img, &kps, FaceRgb::RED, 2);
        assert_eq!(img.get_pixel(8, 10).0, [255, 64, 64]);
        assert_eq!(img.get_pixel(10, 8).0, [255, 64, 64]);
        assert_eq!(img.get_pixel(10, 12).0, [255, 64, 64]);
        assert_eq!(img.get_pixel(20, 10).0, [255, 64, 64]); // second landmark
        assert_eq!(img.get_pixel(5, 5).0, [0, 0, 0]);
    }

    #[test]
    fn text_width_counts_chars_not_bytes() {
        assert_eq!(text_width("abc"), 18);
        assert_eq!(text_width(""), 0);
        assert_eq!(text_width("éx"), 12); // é is one char, three UTF-8 bytes
        assert_eq!(text_width("🚀"), 6);
    }

    #[test]
    fn draw_text_non_ascii_uses_space_glyph_without_panic() {
        let mut img = RgbImage::new(40, 20);
        draw_text(&mut img, 0, 0, "aé🚀z", FaceRgb::WHITE);
        // Chars are counted, not bytes: 'a' at cell 0, é/🚀 skipped at cells
        // 1/2, 'z' at cell 3 (x = 3 * 6 = 18). 'a' col0 = 0x20 has bit 5 set
        // (row 5); 'z' col0 = 0x44 has bits 2 and 6 (rows 2 and 6).
        assert_eq!(img.get_pixel(0, 5).0, [255, 255, 255], "'a' paints bit 5");
        assert_eq!(img.get_pixel(18, 2).0, [255, 255, 255], "'z' paints bit 2");
        assert_eq!(img.get_pixel(18, 6).0, [255, 255, 255], "'z' paints bit 6");
        // The é/🚀 cells fall back to the space glyph: nothing painted.
        assert_eq!(img.get_pixel(6, 3).0, [0, 0, 0], "é cell is empty");
        assert_eq!(img.get_pixel(12, 3).0, [0, 0, 0], "🚀 cell is empty");
    }
}
