//! Exact in-circle sign for exact coordinate differences (Shewchuk 1997 stage B).
//! `robust::incircle` zero-fills about 18 KiB per adaptive call, and grids reach that stage on most tests.

// Keeps every expansion product clear of underflow and overflow, so each transformation is exact.
const MIN_DIFF: f64 = 1.0 / (1u128 << 100) as f64 / (1u128 << 100) as f64;
const MAX_DIFF: f64 = (1u128 << 100) as f64 * (1u128 << 100) as f64;

#[inline]
fn in_range(d: f64) -> bool {
    let a = d.abs();
    a == 0.0 || (MIN_DIFF..=MAX_DIFF).contains(&a)
}

#[inline]
fn diff_is_exact(a: f64, b: f64, x: f64) -> bool {
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    around + bround == 0.0
}

#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        (x, a.mul_add(b, -x))
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        const SPLITTER: f64 = 134_217_729.0;
        let split = |v: f64| {
            let c = SPLITTER * v;
            let big = c - v;
            let hi = c - big;
            (hi, v - hi)
        };
        let (ahi, alo) = split(a);
        let (bhi, blo) = split(b);
        let err1 = x - ahi * bhi;
        let err2 = err1 - alo * bhi;
        let err3 = err2 - ahi * blo;
        (x, alo * blo - err3)
    }
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    let avirt = x - bvirt;
    let bround = b - bvirt;
    let around = a - avirt;
    (x, around + bround)
}

#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    (x, around + bround)
}

#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    (x, b - bvirt)
}

#[inline]
fn cross(a: f64, b: f64, c: f64, d: f64) -> [f64; 4] {
    let (a1, a0) = two_product(a, b);
    let (b1, b0) = two_product(c, d);
    let (i, x0) = two_diff(a0, b0);
    let (j, r0) = two_sum(a1, i);
    let (i, x1) = two_diff(r0, b1);
    let (x3, x2) = two_sum(j, i);
    [x0, x1, x2, x3]
}

fn scale_expansion_zeroelim(e: &[f64], b: f64, h: &mut [f64]) -> usize {
    let (mut q, hh) = two_product(e[0], b);
    let mut n = 0;
    if hh != 0.0 {
        h[n] = hh;
        n += 1;
    }
    for &enow in &e[1..] {
        let (product1, product0) = two_product(enow, b);
        let (sum, hh) = two_sum(q, product0);
        if hh != 0.0 {
            h[n] = hh;
            n += 1;
        }
        let (nq, hh) = fast_two_sum(product1, sum);
        q = nq;
        if hh != 0.0 {
            h[n] = hh;
            n += 1;
        }
    }
    if q != 0.0 || n == 0 {
        h[n] = q;
        n += 1;
    }
    n
}

#[inline]
fn smaller_first(enow: f64, fnow: f64) -> bool {
    (fnow > enow) == (fnow > -enow)
}

fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64], h: &mut [f64]) -> usize {
    let (mut ei, mut fi) = (0, 0);
    let mut q;
    if smaller_first(e[0], f[0]) {
        q = e[0];
        ei += 1;
    } else {
        q = f[0];
        fi += 1;
    }
    let mut n = 0;
    if ei < e.len() && fi < f.len() {
        let hh;
        if smaller_first(e[ei], f[fi]) {
            (q, hh) = fast_two_sum(e[ei], q);
            ei += 1;
        } else {
            (q, hh) = fast_two_sum(f[fi], q);
            fi += 1;
        }
        if hh != 0.0 {
            h[n] = hh;
            n += 1;
        }
        while ei < e.len() && fi < f.len() {
            let hh;
            if smaller_first(e[ei], f[fi]) {
                (q, hh) = two_sum(q, e[ei]);
                ei += 1;
            } else {
                (q, hh) = two_sum(q, f[fi]);
                fi += 1;
            }
            if hh != 0.0 {
                h[n] = hh;
                n += 1;
            }
        }
    }
    for &x in e[ei..].iter().chain(&f[fi..]) {
        let hh;
        (q, hh) = two_sum(q, x);
        if hh != 0.0 {
            h[n] = hh;
            n += 1;
        }
    }
    if q != 0.0 || n == 0 {
        h[n] = q;
        n += 1;
    }
    n
}

fn lift(e: &[f64; 4], x: f64, y: f64, out: &mut [f64; 32]) -> usize {
    let mut t8 = [0.0; 8];
    let mut xx = [0.0; 16];
    let mut yy = [0.0; 16];
    let n = scale_expansion_zeroelim(e, x, &mut t8);
    let nx = scale_expansion_zeroelim(&t8[..n], x, &mut xx);
    let n = scale_expansion_zeroelim(e, y, &mut t8);
    let ny = scale_expansion_zeroelim(&t8[..n], y, &mut yy);
    fast_expansion_sum_zeroelim(&xx[..nx], &yy[..ny], out)
}

// Scales to integers below 2^62 so each lifted term stays below 2^250 and the sum fits in 256 bits.
fn scaled_ints(d: [f64; 6]) -> Option<[i64; 6]> {
    let m = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if m == 0.0 {
        return Some([0; 6]);
    }
    let e = ((m.to_bits() >> 52) & 0x7ff) as i64 - 1023;
    let scale = f64::from_bits(((61 - e + 1023) as u64) << 52);
    let mut out = [0i64; 6];
    for (o, &v) in out.iter_mut().zip(&d) {
        let s = v * scale;
        if s.fract() != 0.0 {
            return None;
        }
        *o = s as i64;
    }
    Some(out)
}

fn mul_u128(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a & u128::from(u64::MAX), a >> 64);
    let (b0, b1) = (b & u128::from(u64::MAX), b >> 64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & u128::from(u64::MAX)) + (p10 & u128::from(u64::MAX));
    let lo = (p00 & u128::from(u64::MAX)) | (mid << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

fn lifted_term(x: i64, y: i64, cross: i128) -> (u128, u128) {
    let lift = (i128::from(x) * i128::from(x) + i128::from(y) * i128::from(y)).unsigned_abs();
    let (hi, lo) = mul_u128(lift, cross.unsigned_abs());
    if cross < 0 {
        let lo = (!lo).wrapping_add(1);
        ((!hi).wrapping_add(u128::from(lo == 0)), lo)
    } else {
        (hi, lo)
    }
}

fn add256(a: (u128, u128), b: (u128, u128)) -> (u128, u128) {
    let (lo, carry) = a.1.overflowing_add(b.1);
    (a.0.wrapping_add(b.0).wrapping_add(u128::from(carry)), lo)
}

fn incircle_int(v: [i64; 6]) -> f64 {
    let [adx, bdx, cdx, ady, bdy, cdy] = v.map(i128::from);
    let bc = bdx * cdy - cdx * bdy;
    let ca = cdx * ady - adx * cdy;
    let ab = adx * bdy - bdx * ady;
    let t = add256(
        add256(lifted_term(v[0], v[3], bc), lifted_term(v[1], v[4], ca)),
        lifted_term(v[2], v[5], ab),
    );
    if (t.0 as i128) < 0 {
        -1.0
    } else if t == (0, 0) {
        0.0
    } else {
        1.0
    }
}

/// Returns `None` when a difference is inexact or out of range; the caller then uses `robust`.
#[inline(never)]
pub fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Option<f64> {
    let adx = a[0] - d[0];
    let bdx = b[0] - d[0];
    let cdx = c[0] - d[0];
    let ady = a[1] - d[1];
    let bdy = b[1] - d[1];
    let cdy = c[1] - d[1];
    let exact = diff_is_exact(a[0], d[0], adx)
        & diff_is_exact(b[0], d[0], bdx)
        & diff_is_exact(c[0], d[0], cdx)
        & diff_is_exact(a[1], d[1], ady)
        & diff_is_exact(b[1], d[1], bdy)
        & diff_is_exact(c[1], d[1], cdy);
    let ranged = in_range(adx)
        & in_range(bdx)
        & in_range(cdx)
        & in_range(ady)
        & in_range(bdy)
        & in_range(cdy);
    if exact && ranged {
        let d = [adx, bdx, cdx, ady, bdy, cdy];
        return Some(scaled_ints(d).map_or_else(|| incircle_expansion(d), incircle_int));
    }
    coordinate_ints([a, b, c, d]).map(incircle_int)
}

// Handles inexact float differences: in units of the finest coordinate bit, the differences are exact integers.
fn coordinate_ints(p: [[f64; 2]; 4]) -> Option<[i64; 6]> {
    let mut parts = [(0i64, i32::MAX); 8];
    for (part, &v) in parts.iter_mut().zip(p.as_flattened()) {
        if !in_range(v) {
            return None;
        }
        if v != 0.0 {
            let bits = v.to_bits();
            let m = (bits & ((1 << 52) - 1)) | (1 << 52);
            let tz = m.trailing_zeros();
            let mag = (m >> tz) as i64;
            let lsb = ((bits >> 52) & 0x7ff) as i32 - 1075 + tz as i32;
            *part = (if v < 0.0 { -mag } else { mag }, lsb);
        }
    }
    let emin = parts.iter().map(|&(_, e)| e).min().unwrap_or(i32::MAX);
    let mut ints = [0i128; 8];
    for (int, &(m, e)) in ints.iter_mut().zip(&parts) {
        if m != 0 {
            let shift = e - emin;
            if shift > 72 {
                return None;
            }
            *int = i128::from(m) << shift;
        }
    }
    let mut out = [0i64; 6];
    for (k, o) in out.iter_mut().enumerate() {
        let (i, j) = if k < 3 {
            (2 * k, 6)
        } else {
            (2 * (k - 3) + 1, 7)
        };
        *o = i64::try_from(ints[i] - ints[j])
            .ok()
            .filter(|v| v.unsigned_abs() < 1 << 62)?;
    }
    Some(out)
}

fn incircle_expansion([adx, bdx, cdx, ady, bdy, cdy]: [f64; 6]) -> f64 {
    let bc = cross(bdx, cdy, cdx, bdy);
    let ca = cross(cdx, ady, adx, cdy);
    let ab = cross(adx, bdy, bdx, ady);

    let mut adet = [0.0; 32];
    let mut bdet = [0.0; 32];
    let na = lift(&bc, adx, ady, &mut adet);
    let nb = lift(&ca, bdx, bdy, &mut bdet);
    let mut abdet = [0.0; 64];
    let nab = fast_expansion_sum_zeroelim(&adet[..na], &bdet[..nb], &mut abdet);
    let nc = lift(&ab, cdx, cdy, &mut adet);
    let mut fin = [0.0; 96];
    let n = fast_expansion_sum_zeroelim(&abdet[..nab], &adet[..nc], &mut fin);
    fin[n - 1]
}

#[cfg(test)]
mod tests {
    use super::{
        coordinate_ints, diff_is_exact, incircle, incircle_expansion, incircle_int, scaled_ints,
    };
    use robust::Coord;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn sign(x: f64) -> i8 {
        i8::from(x > 0.0) - i8::from(x < 0.0)
    }

    fn reference(p: [[f64; 2]; 4]) -> i8 {
        let c = |q: [f64; 2]| Coord { x: q[0], y: q[1] };
        sign(robust::incircle(c(p[0]), c(p[1]), c(p[2]), c(p[3])))
    }

    fn check(p: [[f64; 2]; 4], hits: &mut [usize; 4]) {
        let Some(got) = incircle(p[0], p[1], p[2], p[3]) else {
            return;
        };
        let want = reference(p);
        assert_eq!(sign(got), want, "{p:?}");
        let d = [
            p[0][0] - p[3][0],
            p[1][0] - p[3][0],
            p[2][0] - p[3][0],
            p[0][1] - p[3][1],
            p[1][1] - p[3][1],
            p[2][1] - p[3][1],
        ];
        let exact = (0..6).all(|k| {
            let (i, j) = if k < 3 { (k, 0) } else { (k - 3, 1) };
            diff_is_exact(p[i][j], p[3][j], d[k])
        });
        if exact {
            assert_eq!(sign(incircle_expansion(d)), want, "{p:?}");
            if let Some(v) = scaled_ints(d) {
                assert_eq!(sign(incircle_int(v)), want, "{p:?}");
            }
        } else {
            hits[3] += 1;
        }
        if let Some(v) = coordinate_ints(p) {
            assert_eq!(sign(incircle_int(v)), want, "{p:?}");
        }
        hits[(want + 1) as usize] += 1;
    }

    #[test]
    fn matches_robust_on_grids_and_near_degenerate_points() {
        let mut r = rng(7);
        let mut hits = [0usize; 4];
        for round in 0..200_000 {
            let scale = [0.25, 1.0, 1e-3, 360.0 / 1279.0, 1e9, 3e-90][round % 6];
            let off = [0.0, 180.0, -90.0, 1e6, 0.5][(round / 6) % 5];
            let mut q = || {
                [
                    (r() * 8.0).floor() * scale + off,
                    (r() * 8.0).floor() * scale - off,
                ]
            };
            check([q(), q(), q(), q()], &mut hits);

            let cx = r() * 100.0;
            let cy = r() * 100.0;
            let rad = 1e-3 + r();
            let mut on = || {
                let t = r() * std::f64::consts::TAU;
                [cx + rad * t.cos(), cy + rad * t.sin()]
            };
            check([on(), on(), on(), on()], &mut hits);

            let mut across = || {
                let k = [1e-3, 0.3, 7.0][round % 3];
                [(r() - 0.5) * k + [0.0, 360.0][round % 2], (r() - 0.5) * k]
            };
            check([across(), across(), across(), across()], &mut hits);

            let base = [r() * 400.0 - 200.0, r() * 400.0 - 200.0];
            let mut near = || {
                let k = [1e-15, 1e-12, 1e-6][round % 3];
                [base[0] + (r() - 0.5) * k, base[1] + (r() - 0.5) * k]
            };
            check([near(), near(), near(), near()], &mut hits);
        }
        assert!(hits.iter().all(|&h| h > 1000), "{hits:?}");
    }
}
