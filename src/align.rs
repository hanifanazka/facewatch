//! Face alignment: a least-squares similarity transform (`estimate_norm` /
//! `norm_crop`), mirrored from insightface's `utils/face_align.py`.

use ndarray::Array3;

use crate::face::{Array3U8, Kps};

/// Canonical ArcFace template points (width/height 1.0 scale), as used by
/// insightface `face_align.arcface_dst`.
pub const ARCFACE_DST: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Computes the similarity-transform matrix `M` (2x3) that best maps the five
/// landmarks `lmk` onto the canonical ArcFace template for the given output
/// size, following insightface's `estimate_norm`.
pub fn estimate_norm(lmk: &Kps, image_size: usize) -> [[f64; 3]; 2] {
    let (ratio, diff_x) = if image_size % 112 == 0 {
        (image_size as f64 / 112.0, 0.0)
    } else if image_size % 128 == 0 {
        (image_size as f64 / 128.0, 8.0 * image_size as f64 / 128.0)
    } else {
        (1.0, 0.0)
    };

    let mut dst = [[0.0f64; 2]; 5];
    for (i, pt) in ARCFACE_DST.iter().enumerate() {
        dst[i] = [pt[0] * ratio + diff_x, pt[1] * ratio];
    }

    // SCRFD landmarks arrive as f32; Umeyama works in f64 (matches the
    // NumPy reference which uses float64).
    let src64: [[f64; 2]; 5] = std::array::from_fn(|i| [lmk[i][0] as f64, lmk[i][1] as f64]);
    umeyama_similarity(&src64, &dst)
}

/// Estimates a similarity transform mapping `src` onto `dst` using the Umeyama
/// least-squares method (as in `skimage.transform.SimilarityTransform`).
pub fn umeyama_similarity(src: &[[f64; 2]; 5], dst: &[[f64; 2]; 5]) -> [[f64; 3]; 2] {
    let num = 5.0f64;

    let src_mean = mean(src);
    let dst_mean = mean(dst);

    // Demeaned coordinates.
    let mut src_dm = [[0.0f64; 2]; 5];
    let mut dst_dm = [[0.0f64; 2]; 5];
    for i in 0..5 {
        src_dm[i][0] = src[i][0] - src_mean[0];
        src_dm[i][1] = src[i][1] - src_mean[1];
        dst_dm[i][0] = dst[i][0] - dst_mean[0];
        dst_dm[i][1] = dst[i][1] - dst_mean[1];
    }

    // A = dst_dm^T @ src_dm / num   (2x2)
    let mut a = [[0.0f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..5 {
                sum += dst_dm[k][i] * src_dm[k][j];
            }
            a[i][j] = sum / num;
        }
    }

    let det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    // d = [1, 1], if det(A) < 0 then d[-1] = -1
    let mut d = [1.0f64, 1.0];
    if det_a < 0.0 {
        d[1] = -1.0;
    }

    // SVD of the 2x2 matrix A: A = U diag(S) V^T
    let (u, s, v) = svd2x2(a);

    // R = U @ diag(d) @ V^T
    let r = {
        let mut m = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[i][k] * v[j][k] * d[k]; // V^T: v[j][k]
                }
                m[i][j] = sum;
            }
        }
        m
    };

    // scale = 1.0 / (src_demean.var(axis=0).sum()) * (S @ d)
    let mut var_sum = 0.0;
    for i in 0..5 {
        var_sum += src_dm[i][0] * src_dm[i][0] + src_dm[i][1] * src_dm[i][1];
    }
    var_sum /= num; // biased variance sum across both axes
    let scale = if var_sum.abs() < 1e-12 {
        1.0
    } else {
        (s[0] * d[0] + s[1] * d[1]) / var_sum
    };

    // t = dst_mean - scale * (R @ src_mean)
    let r_src = [
        r[0][0] * src_mean[0] + r[0][1] * src_mean[1],
        r[1][0] * src_mean[0] + r[1][1] * src_mean[1],
    ];
    let t = [dst_mean[0] - scale * r_src[0], dst_mean[1] - scale * r_src[1]];

    // M = scale * R with translation.
    [
        [scale * r[0][0], scale * r[0][1], t[0]],
        [scale * r[1][0], scale * r[1][1], t[1]],
    ]
}

fn mean(pts: &[[f64; 2]; 5]) -> [f64; 2] {
    let mut m = [0.0f64; 2];
    for p in pts {
        m[0] += p[0];
        m[1] += p[1];
    }
    [m[0] / 5.0, m[1] / 5.0]
}

/// Analytic SVD of a 2x2 matrix: returns `(U, S, V)` with `A = U diag(S) V^T`,
/// `S` sorted descending, and orthogonal `U`, `V` (conjugate sign flips are
/// irrelevant because the Umeyama transform only uses `U V^T` combinations).
fn svd2x2(a: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
    // H = A^T A (symmetric 2x2)
    let h00 = a[0][0] * a[0][0] + a[1][0] * a[1][0];
    let h01 = a[0][0] * a[0][1] + a[1][0] * a[1][1];
    let h11 = a[0][1] * a[0][1] + a[1][1] * a[1][1];

    let trace = h00 + h11;
    let det = h00 * h11 - h01 * h01;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let l1 = (trace + disc) / 2.0;
    let l2 = (trace - disc) / 2.0;
    let s1 = l1.sqrt();
    let s2 = l2.sqrt();

    // Right singular vectors: eigenvectors of H.
    let (v1, v2) = if h01.abs() > 1e-12 {
        let v1 = normalize(h01, l1 - h00);
        let v2 = normalize(h01, l2 - h00);
        (v1, v2)
    } else if h00 >= h11 {
        ([1.0, 0.0], [0.0, 1.0])
    } else {
        ([0.0, 1.0], [1.0, 0.0])
    };

    // Left singular vectors: u_i = A v_i / s_i.
    let mut u1 = if s1 > 1e-12 {
        [a[0][0] * v1[0] + a[0][1] * v1[1], a[1][0] * v1[0] + a[1][1] * v1[1]]
    } else {
        [1.0, 0.0]
    };
    u1 = normalize(u1[0], u1[1]);

    let mut u2 = if s2 > 1e-12 {
        [a[0][0] * v2[0] + a[0][1] * v2[1], a[1][0] * v2[0] + a[1][1] * v2[1]]
    } else {
        [-u1[1], u1[0]]
    };
    u2 = normalize(u2[0], u2[1]);

    (
        [[u1[0], u2[0]], [u1[1], u2[1]]],
        [s1, s2],
        [[v1[0], v2[0]], [v1[1], v2[1]]],
    )
}

fn normalize(x: f64, y: f64) -> [f64; 2] {
    let n = (x * x + y * y).sqrt();
    if n < 1e-12 {
        [1.0, 0.0]
    } else {
        [x / n, y / n]
    }
}

/// Aligns an RGB frame by warping the five detected landmarks onto the
/// canonical ArcFace template, producing an `image_size x image_size` crop.
///
/// Mirrors `cv2.warpAffine(img, M, (size, size), borderValue=0.0)` with
/// bilinear interpolation and constant-zero border.
pub fn norm_crop(src: &Array3U8, landmark: &Kps, image_size: usize) -> Array3U8 {
    let m = estimate_norm(landmark, image_size);
    warp_affine(src, m, image_size)
}

/// Warps `src` with the 2x3 affine matrix `m`, bilinear interpolation,
/// constant zero border.
///
/// `m` maps source points onto destination points (the face-alignment
/// transform `landmarks -> ArcFace template`). `cv2.warpAffine` treats `m`
/// as the content transform and therefore samples `src` at `m^-1 * (x, y, 1)`
/// for each destination pixel; we mirror that by inverting `m` first.
pub fn warp_affine(src: &Array3U8, m: [[f64; 3]; 2], size: usize) -> Array3U8 {
    let (sh, sw) = (src.shape()[0] as i64, src.shape()[1] as i64);

    // Analytical inverse of the padded 3x3 matrix
    //   [[a, b, c], [d, e, f], [0, 0, 1]].
    let [a, b, c] = m[0];
    let [d, e, f] = m[1];
    let det = a * e - b * d;
    // Rows of the inverse (scaled by 1/det):
    //   [ e, -b, b*f - c*e ]
    //   [-d,  a, c*d - a*f ]
    let (inv00, inv01, inv02) = (e, -b, b * f - c * e);
    let (inv10, inv11, inv12) = (-d, a, c * d - a * f);

    let mut out = Array3::<u8>::zeros((size, size, 3));

    for y in 0..size {
        for x in 0..size {
            let xf = x as f64;
            let yf = y as f64;
            let sx = (inv00 * xf + inv01 * yf + inv02) / det;
            let sy = (inv10 * xf + inv11 * yf + inv12) / det;

            let x0 = sx.floor() as i64;
            let y0 = sy.floor() as i64;
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;

            let mut pix = [0.0f64; 3];
            for (ox, oy, w) in [
                (x0, y0, (1.0 - fx) * (1.0 - fy)),
                (x0 + 1, y0, fx * (1.0 - fy)),
                (x0, y0 + 1, (1.0 - fx) * fy),
                (x0 + 1, y0 + 1, fx * fy),
            ] {
                if ox >= 0 && ox < sw && oy >= 0 && oy < sh {
                    let px = src[[oy as usize, ox as usize, 0]] as f64;
                    let py = src[[oy as usize, ox as usize, 1]] as f64;
                    let pz = src[[oy as usize, ox as usize, 2]] as f64;
                    pix[0] += w * px;
                    pix[1] += w * py;
                    pix[2] += w * pz;
                }
            }

            for c in 0..3 {
                out[[y, x, c]] = pix[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_template_maps_to_identity() {
        // When landmarks equal (a scaled) canonical template, the estimate
        // should be close to a pure scaling + translation.
        let kps: Kps = [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ];
        let m = estimate_norm(&kps, 112);
        // Mapping canonical point (38.29, 51.69) through M should return itself.
        assert!((m[0][0] * 38.2946 + m[0][1] * 51.6963 + m[0][2] - 38.2946).abs() < 1e-3);
        assert!((m[1][0] * 38.2946 + m[1][1] * 51.6963 + m[1][2] - 51.6963).abs() < 1e-3);
    }

    #[test]
    fn warp_produces_sized_output() {
        let src = Array3::<u8>::zeros((64, 48, 3));
        let kps: Kps = [
            [20.0, 20.0],
            [30.0, 20.0],
            [25.0, 26.0],
            [22.0, 32.0],
            [28.0, 32.0],
        ];
        let out = norm_crop(&src, &kps, 112);
        assert_eq!(out.shape(), &[112, 112, 3]);
    }

    #[test]
    fn warp_uses_inverse_mapping_like_cv2() {
        // cv2.warpAffine treats the matrix as the content transform and
        // samples src at M^-1 * (x, y, 1). With a pure translation matrix,
        // content at src (10,10) must appear at dst (15,13).
        let mut src = Array3::<u8>::zeros((40, 40, 3));
        src[[10, 10, 0]] = 255;
        let m = [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]];
        let out = warp_affine(&src, m, 40);
        assert_eq!(out[[13, 15, 0]], 255, "content should move to dst(15,13)");
        assert_eq!(out[[10, 10, 0]], 0, "src position should no longer be lit");
    }
}