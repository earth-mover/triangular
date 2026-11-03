#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

#[cfg(feature = "python")]
use numpy::ndarray::Array2;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::{PyModule, PyModuleMethods};
#[cfg(feature = "python")]
use pyo3::{pyfunction, pymodule, wrap_pyfunction, Bound, PyResult, Python};
use rayon::prelude::*;
use robust::{incircle, orient2d, Coord};
use std::borrow::Cow;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

mod exact;

const EMPTY: u32 = u32::MAX;

const PARALLEL_MIN_N: usize = 50_000;

const STRIP_MIN_PTS: usize = 1024;
const PAR_SORT_MIN_N: usize = 1 << 14;
const EXTRACT_CHUNK_QUADS: usize = 1 << 14;
const HULL_FILTER_MIN_N: usize = 64;
const HULL_SCAN_CHUNK: usize = 16_384;
const EXTREME_BLOCK: usize = 1024;
const BOUNDS_LANES: usize = 4;
const EXTREME_SAMPLE: usize = 16_384;
const UNDERFLOW_SLACK: f64 = 1e-300;
const SAFE_MIN_ABS: f64 = 1e-100;
const SAFE_MAX_ABS: f64 = 1e100;
const GRID_MIN_ABS: f64 = 1e-30;
const GRID_MAX_ABS: f64 = 1e30;
const SCAN_CHUNK: usize = 8192;
const SAMPLE_WINDOWS: usize = 64;
const SAMPLE_STEPS: usize = 64;
const SAMPLE_MIN_WINDOWS: usize = 8;
const SAMPLE_MIN_STEPS: usize = 16;

const PRED_EPSILON: f64 = f64::EPSILON * 0.5;
const CCW_ERRBOUND_A: f64 = (3.0 + 16.0 * PRED_EPSILON) * PRED_EPSILON;
const ICC_ERRBOUND_A: f64 = (10.0 + 96.0 * PRED_EPSILON) * PRED_EPSILON;

type StripResult = (Vec<u32>, Vec<u32>, u32, u32);

pub type Pt = [f64; 2];

#[inline]
fn co(p: Pt) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

#[inline]
fn ccw(a: Pt, b: Pt, p: Pt) -> f64 {
    let detleft = (a[0] - p[0]) * (b[1] - p[1]);
    let detright = (a[1] - p[1]) * (b[0] - p[0]);
    let det = detleft - detright;
    let errbound = CCW_ERRBOUND_A * (detleft + detright).abs();
    if det >= errbound || -det >= errbound {
        return det;
    }
    orient2d(co(a), co(b), co(p))
}

#[inline]
fn in_circle(a: Pt, b: Pt, c: Pt, p: Pt) -> bool {
    let adx = a[0] - p[0];
    let bdx = b[0] - p[0];
    let cdx = c[0] - p[0];
    let ady = a[1] - p[1];
    let bdy = b[1] - p[1];
    let cdy = c[1] - p[1];

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let alift = adx * adx + ady * ady;

    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let blift = bdx * bdx + bdy * bdy;

    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;
    let clift = cdx * cdx + cdy * cdy;

    let det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * alift
        + (cdxady.abs() + adxcdy.abs()) * blift
        + (adxbdy.abs() + bdxady.abs()) * clift;
    let errbound = ICC_ERRBOUND_A * permanent;
    if det > errbound {
        return true;
    }
    if -det > errbound {
        return false;
    }
    in_circle_slow(a, b, c, p)
}

#[cold]
#[inline(never)]
fn in_circle_slow(a: Pt, b: Pt, c: Pt, p: Pt) -> bool {
    exact::incircle(a, b, c, p).unwrap_or_else(|| incircle(co(a), co(b), co(c), co(p))) > 0.0
}

#[inline]
fn dist2(a: Pt, b: Pt) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn circumradius2(a: Pt, b: Pt, c: Pt) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let ex = c[0] - a[0];
    let ey = c[1] - a[1];
    let det = dx * ey - dy * ex;
    if det == 0.0 {
        return f64::INFINITY;
    }
    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = 0.5 / det;
    let x = (ey * bl - dy * cl) * d;
    let y = (dx * cl - ex * bl) * d;
    x * x + y * y
}

fn circumcenter(a: Pt, b: Pt, c: Pt) -> Pt {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let ex = c[0] - a[0];
    let ey = c[1] - a[1];
    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = 0.5 / (dx * ey - dy * ex);
    [
        a[0] + (ey * bl - dy * cl) * d,
        a[1] + (dx * cl - ex * bl) * d,
    ]
}

#[inline]
fn pseudo_angle(dx: f64, dy: f64) -> f64 {
    let denom = dx.abs() + dy.abs();
    if denom == 0.0 {
        return 0.0;
    }
    let p = dx / denom;
    if dy > 0.0 {
        (3.0 - p) / 4.0
    } else {
        (1.0 + p) / 4.0
    }
}

#[inline]
fn next_he(e: usize) -> usize {
    if e % 3 == 2 {
        e - 2
    } else {
        e + 1
    }
}

#[inline]
fn prev_he(e: usize) -> usize {
    if e.is_multiple_of(3) {
        e + 2
    } else {
        e - 1
    }
}

// Maps an `f64` to an `i64` whose integer order matches `f64::total_cmp`, except that -0.0 and 0.0
// get the same key. Equal coordinates must sort next to each other, or dedup and the hull chain fail.
#[inline]
fn total_order_key(v: f64) -> i64 {
    let b = (v + 0.0).to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

#[inline]
fn from_total_order_key(k: i64) -> f64 {
    f64::from_bits((k ^ (((k >> 63) as u64) >> 1) as i64) as u64)
}

#[inline]
fn argmin(idx: impl Iterator<Item = usize>, key: impl Fn(usize) -> f64) -> Option<usize> {
    let mut best: Option<(i64, usize)> = None;
    for i in idx {
        let k = total_order_key(key(i));
        if best.is_none_or(|(bk, _)| k < bk) {
            best = Some((k, i));
        }
    }
    best.map(|(_, i)| i)
}

fn find_seed(pts: &[Pt]) -> Option<(usize, usize, usize)> {
    let n = pts.len();

    let (mut minx, mut miny, mut maxx, mut maxy) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &[x, y] in pts {
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    let bbox_center = [f64::midpoint(minx, maxx), f64::midpoint(miny, maxy)];

    let i0 = argmin(0..n, |i| dist2(bbox_center, pts[i]))?;
    let p0 = pts[i0];

    let i1 = argmin(
        (0..n).filter(|&i| i != i0 && !same_point(p0, pts[i])),
        |i| dist2(p0, pts[i]),
    )?;
    let p1 = pts[i1];

    let i2 = argmin(
        (0..n)
            .filter(|&i| i != i0 && i != i1 && !same_point(p0, pts[i]) && !same_point(p1, pts[i])),
        |i| circumradius2(p0, p1, pts[i]),
    )?;
    let p2 = pts[i2];

    if circumradius2(p0, p1, p2).is_infinite() {
        return None;
    }

    if ccw(p0, p1, p2) < 0.0 {
        Some((i0, i2, i1))
    } else {
        Some((i0, i1, i2))
    }
}

struct Sweep<'a> {
    pts: &'a [Pt],
    triangles: Vec<u32>,
    halfedges: Vec<u32>,
    hull_prev: Vec<u32>,
    hull_next: Vec<u32>,
    hull_tri: Vec<u32>,
    hull_hash: Vec<u32>,
    hash_size: usize,
    center: Pt,
    edge_stack: Vec<u32>,
}

impl Sweep<'_> {
    #[inline]
    fn link(&mut self, a: usize, b: u32) {
        self.halfedges[a] = b;
        if b != EMPTY {
            self.halfedges[b as usize] = a as u32;
        }
    }

    fn add_triangle(&mut self, i0: u32, i1: u32, i2: u32, a: u32, b: u32, c: u32) -> usize {
        let t = self.triangles.len();
        self.triangles.extend_from_slice(&[i0, i1, i2]);
        self.halfedges.extend_from_slice(&[EMPTY, EMPTY, EMPTY]);
        self.link(t, a);
        self.link(t + 1, b);
        self.link(t + 2, c);
        t
    }

    #[inline]
    fn hash_key(&self, p: Pt) -> usize {
        let angle = pseudo_angle(p[0] - self.center[0], p[1] - self.center[1]);
        ((angle * self.hash_size as f64) as usize) & (self.hash_size - 1)
    }

    #[inline]
    fn hash_set(&mut self, p: Pt, i: u32) {
        let k = self.hash_key(p);
        self.hull_hash[k] = i;
    }

    fn legalize(&mut self, mut a: usize) -> u32 {
        loop {
            let b = self.halfedges[a];
            let ar = prev_he(a);
            if b == EMPTY {
                match self.edge_stack.pop() {
                    Some(next) => {
                        a = next as usize;
                        continue;
                    }
                    None => return ar as u32,
                }
            }
            let b = b as usize;
            let al = next_he(a);
            let bl = prev_he(b);
            let p0 = self.triangles[ar];
            let pr = self.triangles[a];
            let pl = self.triangles[al];
            let p1 = self.triangles[bl];

            let pts = &self.pts;
            if in_circle(
                pts[p0 as usize],
                pts[pr as usize],
                pts[pl as usize],
                pts[p1 as usize],
            ) {
                self.triangles[a] = p1;
                self.triangles[b] = p0;
                let hbl = self.halfedges[bl];
                if hbl == EMPTY {
                    self.hull_tri[p1 as usize] = a as u32;
                }
                self.link(a, hbl);
                let har = self.halfedges[ar];
                self.link(b, har);
                self.link(ar, bl as u32);
                self.edge_stack.push(next_he(b) as u32);
            } else {
                match self.edge_stack.pop() {
                    Some(next) => a = next as usize,
                    None => return ar as u32,
                }
            }
        }
    }

    #[inline]
    fn visible(&self, a: u32, b: u32, p: Pt) -> bool {
        ccw(self.pts[a as usize], self.pts[b as usize], p) < 0.0
    }

    fn run(&mut self, center: Pt, seed: (usize, usize, usize)) {
        let (i0, i1, i2) = seed;
        self.center = center;

        self.hull_next[i0] = i1 as u32;
        self.hull_next[i1] = i2 as u32;
        self.hull_next[i2] = i0 as u32;
        self.hull_prev[i0] = i2 as u32;
        self.hull_prev[i1] = i0 as u32;
        self.hull_prev[i2] = i1 as u32;
        self.hull_tri[i0] = 0;
        self.hull_tri[i1] = 1;
        self.hull_tri[i2] = 2;
        self.hash_set(self.pts[i0], i0 as u32);
        self.hash_set(self.pts[i1], i1 as u32);
        self.hash_set(self.pts[i2], i2 as u32);

        self.add_triangle(i0 as u32, i1 as u32, i2 as u32, EMPTY, EMPTY, EMPTY);

        for i in 0..self.pts.len() {
            if i == i0 || i == i1 || i == i2 {
                continue;
            }
            if i > 0 && same_point(self.pts[i - 1], self.pts[i]) {
                continue;
            }
            self.insert(i as u32);
        }
    }

    fn insert(&mut self, i: u32) {
        let p = self.pts[i as usize];

        let key = self.hash_key(p);
        let mut start = EMPTY;
        for j in 0..self.hash_size {
            start = self.hull_hash[(key + j) & (self.hash_size - 1)];
            if start != EMPTY && start != self.hull_next[start as usize] {
                break;
            }
        }

        start = self.hull_prev[start as usize];
        let mut e = start;
        let mut en = self.hull_next[e as usize];
        while !self.visible(e, en, p) {
            e = en;
            if e == start {
                return;
            }
            en = self.hull_next[e as usize];
        }

        let t = self.add_triangle(e, i, en, EMPTY, EMPTY, self.hull_tri[e as usize]);
        self.hull_tri[i as usize] = self.legalize(t + 2);
        self.hull_tri[e as usize] = t as u32;

        let mut nxt = self.hull_next[e as usize];
        loop {
            let q = self.hull_next[nxt as usize];
            if !self.visible(nxt, q, p) {
                break;
            }
            let t = self.add_triangle(
                nxt,
                i,
                q,
                self.hull_tri[i as usize],
                EMPTY,
                self.hull_tri[nxt as usize],
            );
            self.hull_tri[i as usize] = self.legalize(t + 2);
            self.hull_next[nxt as usize] = nxt;
            nxt = q;
        }

        if e == start {
            loop {
                let q = self.hull_prev[e as usize];
                if !self.visible(q, e, p) {
                    break;
                }
                let t = self.add_triangle(
                    q,
                    i,
                    e,
                    EMPTY,
                    self.hull_tri[e as usize],
                    self.hull_tri[q as usize],
                );
                self.legalize(t + 2);
                self.hull_tri[q as usize] = t as u32;
                self.hull_next[e as usize] = e;
                e = q;
            }
        }

        self.hull_prev[i as usize] = e;
        self.hull_next[e as usize] = i;
        self.hull_prev[nxt as usize] = i;
        self.hull_next[i as usize] = nxt;

        self.hull_hash[key] = i;
        let pe = self.pts[e as usize];
        self.hash_set(pe, e);
    }
}

struct RowGrid<'a> {
    q: Cow<'a, [Pt]>,
    rows: Vec<usize>,
    flip: bool,
}

// Waking the thread pool costs more than the whole grid on small inputs, so they run on this thread.
fn map_items<I: Send, T: Send>(
    items: Vec<I>,
    par: bool,
    f: impl Fn(usize, I) -> T + Sync + Send,
) -> Vec<T> {
    if par {
        items
            .into_par_iter()
            .enumerate()
            .map(|(k, x)| f(k, x))
            .collect()
    } else {
        items
            .into_iter()
            .enumerate()
            .map(|(k, x)| f(k, x))
            .collect()
    }
}

// Nonzero differences of coordinates in this range stay within about 2e-46..2e30. So in-circle
// fourth powers and their expansion tails stay normal, and the exact.rs range guard accepts them.
#[inline]
fn grid_safe(v: f64) -> bool {
    v == 0.0 || (GRID_MIN_ABS..=GRID_MAX_ABS).contains(&v.abs())
}

// Rows have one exact row coordinate and a strictly monotone along-row coordinate.
// The canonical frame (along-row = x up, rows = y up) is an isometry of the input, so it keeps Delaunay.
#[allow(clippy::float_cmp)]
fn row_grid(pts: &[Pt]) -> Option<RowGrid<'_>> {
    let n = pts.len();
    if n < 4 || n > (u32::MAX / 8) as usize {
        return None;
    }
    let a = if pts[0][1] == pts[1][1] {
        0
    } else if pts[0][0] == pts[1][0] {
        1
    } else {
        return None;
    };
    let b = 1 - a;
    let du = pts[1][a] - pts[0][a];
    if !(du.is_finite() && du != 0.0) {
        return None;
    }
    let u_up = du > 0.0;

    let mut starts = vec![0usize];
    let mut prev_v = f64::NAN;
    let mut v_up = None;
    let mut s = 0usize;
    while s < n {
        let v = pts[s][b];
        let mut last_u = pts[s][a];
        if !(grid_safe(v) && grid_safe(last_u)) {
            return None;
        }
        if !prev_v.is_nan() {
            let dir = v > prev_v;
            if *v_up.get_or_insert(dir) != dir {
                return None;
            }
        }
        let mut e = s + 1;
        while e < n && pts[e][b] == v {
            let u = pts[e][a];
            if !grid_safe(u) || (u_up && u <= last_u) || (!u_up && u >= last_u) {
                return None;
            }
            last_u = u;
            e += 1;
        }
        if e - s < 2 {
            return None;
        }
        starts.push(e);
        prev_v = v;
        s = e;
    }
    let nr = starts.len() - 1;
    if nr < 2 {
        return None;
    }
    let v_up = v_up.unwrap_or(true);
    // A reflected frame turns counter-clockwise triangles clockwise in the input frame.
    let flip = (a == 1) != (u_up != v_up);
    if a == 0 && u_up && v_up {
        return Some(RowGrid {
            q: Cow::Borrowed(pts),
            rows: starts,
            flip,
        });
    }

    let su = if u_up { 1.0 } else { -1.0 };
    let sv = if v_up { 1.0 } else { -1.0 };
    let frame = |p: &Pt| [su * p[a], sv * p[b]];
    let q = if n < PARALLEL_MIN_N {
        pts.iter().map(frame).collect()
    } else {
        pts.par_iter().with_min_len(4096).map(frame).collect()
    };
    Some(RowGrid {
        q: Cow::Owned(q),
        rows: starts,
        flip,
    })
}

// Every grid point is outside or on the circumcircle of each cell of a tensor grid, so
// splitting each cell along one diagonal is Delaunay without any predicate.
fn tensor_triangles(rows: &[usize], flip: bool) -> Vec<u32> {
    let m = rows[1] - rows[0];
    let nr = rows.len() - 1;
    let mut out = vec![0u32; 6 * (nr - 1) * (m - 1)];
    let row_chunks: Vec<&mut [u32]> = out.chunks_mut(6 * (m - 1)).collect();
    map_items(row_chunks, nr * m >= PARALLEL_MIN_N, |r, chunk| {
        for (j, cell) in chunk.as_chunks_mut::<6>().0.iter_mut().enumerate() {
            let k = (r * m + j) as u32;
            let up = k + m as u32;
            *cell = if flip {
                [k, up, k + 1, k + 1, up, up + 1]
            } else {
                [k, k + 1, up, k + 1, up + 1, up]
            };
        }
    });
    out
}

#[inline]
fn link_he(he: &mut [u32], a: usize, b: u32) {
    he[a] = b;
    if b != EMPTY {
        he[b as usize] = a as u32;
    }
}

#[inline]
fn edge_illegal(q: &[Pt], tris: &[u32], he: &[u32], e: usize) -> bool {
    let tw = he[e];
    if tw == EMPTY {
        return false;
    }
    let p0 = tris[prev_he(e)];
    let pr = tris[e];
    let pl = tris[next_he(e)];
    let p1 = tris[prev_he(tw as usize)];
    in_circle(
        q[p0 as usize],
        q[pr as usize],
        q[pl as usize],
        q[p1 as usize],
    )
}

fn lawson(q: &[Pt], tris: &mut [u32], he: &mut [u32], stack: &mut Vec<u32>, budget: usize) -> bool {
    let mut flips = 0usize;
    while let Some(e) = stack.pop() {
        let a = e as usize;
        if !edge_illegal(q, tris, he, a) {
            continue;
        }
        flips += 1;
        if flips > budget {
            return false;
        }
        let b = he[a] as usize;
        let ar = prev_he(a);
        let al = next_he(a);
        let bl = prev_he(b);
        let br = next_he(b);
        let p0 = tris[ar];
        let p1 = tris[bl];
        tris[a] = p1;
        tris[b] = p0;
        let hbl = he[bl];
        let har = he[ar];
        link_he(he, a, hbl);
        link_he(he, b, har);
        link_he(he, ar, bl as u32);
        stack.extend_from_slice(&[a as u32, al as u32, b as u32, br as u32]);
    }
    true
}

fn push_tri(tris: &mut Vec<u32>, he: &mut Vec<u32>, v: [u32; 3], k: usize, twin: u32) -> usize {
    let t = tris.len();
    tris.extend_from_slice(&v);
    he.extend_from_slice(&[EMPTY, EMPTY, EMPTY]);
    link_he(he, t + k, twin);
    t
}

// Sorted random rows zip to about one illegal diagonal in six, and a fan has illegal row edges over most points.
// Both need about n sequential flips, so a sample of short zipped windows stops them before the full zip.
fn sample_needs_many_flips(q: &[Pt], rows: &[usize]) -> bool {
    let n = q.len();
    // Small inputs also get a sample, because a failed full attempt costs more than the general path.
    let windows = (n / (64 * SAMPLE_STEPS)).clamp(SAMPLE_MIN_WINDOWS, SAMPLE_WINDOWS);
    let steps = (n / (16 * windows)).clamp(SAMPLE_MIN_STEPS, SAMPLE_STEPS);
    let span = rows[rows.len() - 2];
    let (mut tested, mut illegal) = (0usize, 0usize);
    let (mut row_tested, mut row_illegal) = (0usize, 0usize);
    for k in 0..windows {
        let p = (2 * k + 1) * span / (2 * windows);
        let b = rows.partition_point(|&r| r <= p) - 1;
        let (l0, l1, u1) = (rows[b], rows[b + 1], rows[b + 2]);
        let (nl, nu) = (l1 - l0, u1 - l1);
        let above = q[l1..u1].partition_point(|u| u[0] < q[p][0]);
        let mut i = (p - l0).min(nl - 2);
        let mut j = above.saturating_sub(1).min(nu - 2);
        let next = if b + 3 < rows.len() {
            &q[rows[b + 2]..rows[b + 3]]
        } else {
            &[]
        };
        let mut kn = next.partition_point(|u| u[0] < q[l1 + j + 1][0]);
        // A row edge counts once per lower point under it, so one long edge over a fan weighs as much as the fan.
        let mut under = 0usize;
        let mut test_row = |i: usize, j: usize, under: usize| {
            if next.is_empty() || under == 0 {
                return;
            }
            let xe = q[l1 + j + 1][0];
            while kn < next.len() && next[kn][0] < xe {
                kn += 1;
            }
            let twin = next[kn.saturating_sub(1).min(next.len() - 1)];
            row_tested += under;
            if in_circle(q[l0 + i], q[l1 + j + 1], q[l1 + j], twin) {
                row_illegal += under;
            }
        };
        let mut prev: Option<[Pt; 3]> = None;
        for _ in 0..steps {
            if i + 1 == nl && j + 1 == nu {
                break;
            }
            let lower = j + 1 == nu || (i + 1 < nl && q[l0 + i + 1][0] <= q[l1 + j + 1][0]);
            let (apex, tri) = if lower {
                (q[l0 + i + 1], [q[l0 + i], q[l0 + i + 1], q[l1 + j]])
            } else {
                (q[l1 + j + 1], [q[l0 + i], q[l1 + j + 1], q[l1 + j]])
            };
            if let Some([a, b, c]) = prev {
                tested += 1;
                illegal += usize::from(in_circle(a, b, c, apex));
            }
            prev = Some(tri);
            if lower {
                under += 1;
                i += 1;
            } else {
                test_row(i, j, under.max(1));
                under = 0;
                j += 1;
            }
        }
        if j + 1 < nu {
            test_row(i, j, under);
        }
        // Near-degenerate rows make each exact in-circle test slow, so stop once the verdict is certain.
        let rest = (windows - k - 1) * (steps + 1);
        if verdict(illegal, tested + rest, row_illegal, row_tested + rest) {
            return true;
        }
    }
    verdict(illegal, tested, row_illegal, row_tested)
}

// Each point owns about one diagonal and one row edge, and the full scan stops at n/8 illegal edges.
fn verdict(illegal: usize, tested: usize, row_illegal: usize, row_tested: usize) -> bool {
    let diag = illegal as f64 / tested.max(1) as f64;
    let row = row_illegal as f64 / row_tested.max(1) as f64;
    10.0 * diag > 1.0 || 8.0 * (diag + row) > 1.0
}

// Zip consecutive rows, fill the hull pockets, then flip to Delaunay with Lawson's algorithm.
// A triangulation of the hull with every edge locally Delaunay is Delaunay.
#[allow(clippy::too_many_lines)]
fn zipper_lawson(q: &[Pt], rows: &[usize], rect: &[bool]) -> Option<Vec<u32>> {
    let n = q.len();
    if sample_needs_many_flips(q, rows) {
        return None;
    }
    let nr = rows.len() - 1;
    let nb = nr - 1;
    let mut toff = Vec::with_capacity(nb + 1);
    toff.push(0usize);
    for b in 0..nb {
        toff.push(toff[b] + rows[b + 2] - rows[b] - 2);
    }
    let nt_band = toff[nb];
    let mut tris = vec![0u32; 3 * nt_band];
    let mut he = vec![EMPTY; 3 * nt_band];
    let mut seg_above = vec![EMPTY; n - nr];
    let mut seg_below = vec![EMPTY; n - nr];

    let mut tri_s = Vec::with_capacity(nb);
    let mut he_s = Vec::with_capacity(nb);
    let mut sa_s = Vec::with_capacity(nb);
    let mut sb_s = Vec::with_capacity(nb);
    {
        let (mut tr, mut hr) = (&mut tris[..], &mut he[..]);
        let mut sa = &mut seg_above[..];
        let (_, mut sb) = seg_below.split_at_mut(rows[1] - rows[0] - 1);
        for b in 0..nb {
            let len = 3 * (toff[b + 1] - toff[b]);
            let (x, rest) = tr.split_at_mut(len);
            tri_s.push(x);
            tr = rest;
            let (x, rest) = hr.split_at_mut(len);
            he_s.push(x);
            hr = rest;
            let (x, rest) = sa.split_at_mut(rows[b + 1] - rows[b] - 1);
            sa_s.push(x);
            sa = rest;
            let (x, rest) = sb.split_at_mut(rows[b + 2] - rows[b + 1] - 1);
            sb_s.push(x);
            sb = rest;
        }
    }

    let par = n >= PARALLEL_MIN_N;
    let bands: Vec<_> = tri_s.into_iter().zip(he_s).zip(sa_s).zip(sb_s).collect();
    let bounds: Vec<(u32, u32)> = map_items(bands, par, |b, (((tri, he), sa), sb)| {
        let base = 3 * toff[b];
        let (l0, l1, u1) = (rows[b], rows[b + 1], rows[b + 2]);
        let (nl, nu) = (l1 - l0, u1 - l1);
        let (mut i, mut j) = (0usize, 0usize);
        let mut prev = EMPTY;
        let mut lt = 0usize;
        while i + 1 < nl || j + 1 < nu {
            let lower = j + 1 == nu || (i + 1 < nl && q[l0 + i + 1][0] <= q[l1 + j + 1][0]);
            let g = (base + lt) as u32;
            if lower {
                tri[lt..lt + 3].copy_from_slice(&[
                    (l0 + i) as u32,
                    (l0 + i + 1) as u32,
                    (l1 + j) as u32,
                ]);
                sa[i] = g;
                i += 1;
            } else {
                tri[lt..lt + 3].copy_from_slice(&[
                    (l0 + i) as u32,
                    (l1 + j + 1) as u32,
                    (l1 + j) as u32,
                ]);
                sb[j] = g + 1;
                j += 1;
            }
            he[lt + 2] = prev;
            if prev != EMPTY {
                he[prev as usize - base] = g + 2;
            }
            prev = if lower { g + 1 } else { g };
            lt += 3;
        }
        (base as u32 + 2, prev)
    });

    {
        let mut hr = &mut he[..];
        let mut he_s = Vec::with_capacity(nb);
        for b in 0..nb {
            let (x, rest) = hr.split_at_mut(3 * (toff[b + 1] - toff[b]));
            he_s.push(x);
            hr = rest;
        }
        let (seg_above, seg_below) = (&seg_above, &seg_below);
        map_items(he_s, par, |b, hb| {
            let base = 3 * toff[b];
            if b > 0 {
                for s in rows[b] - b..rows[b + 1] - b - 1 {
                    hb[seg_above[s] as usize - base] = seg_below[s];
                }
            }
            if b + 1 < nb {
                for s in rows[b + 1] - b - 1..rows[b + 2] - b - 2 {
                    hb[seg_below[s] as usize - base] = seg_above[s];
                }
            }
        });
    }

    let mut stack: Vec<(u32, u32)> = vec![(rows[0] as u32, EMPTY)];
    for b in 0..nb {
        let c = rows[b + 1] as u32;
        let mut e_cb = bounds[b].0;
        while stack.len() >= 2 {
            let (bv, e_ba) = stack[stack.len() - 1];
            let (av, _) = stack[stack.len() - 2];
            if ccw(q[c as usize], q[bv as usize], q[av as usize]) >= 0.0 {
                break;
            }
            let t = push_tri(&mut tris, &mut he, [av, bv, c], 0, e_ba);
            link_he(&mut he, t + 1, e_cb);
            e_cb = (t + 2) as u32;
            stack.pop();
        }
        stack.push((c, e_cb));
    }
    stack.clear();
    stack.push(((rows[1] - 1) as u32, EMPTY));
    for b in 0..nb {
        let c = (rows[b + 2] - 1) as u32;
        let mut e_bc = bounds[b].1;
        while stack.len() >= 2 {
            let (bv, e_ab) = stack[stack.len() - 1];
            let (av, _) = stack[stack.len() - 2];
            if ccw(q[av as usize], q[bv as usize], q[c as usize]) >= 0.0 {
                break;
            }
            let t = push_tri(&mut tris, &mut he, [av, c, bv], 1, e_bc);
            link_he(&mut he, t + 2, e_ab);
            e_bc = t as u32;
            stack.pop();
        }
        stack.push((c, e_bc));
    }

    let pocket = 3 * nt_band as u32;
    let ne = 3 * nt_band;
    // Lawson runs in sequence, so a grid that needs about n flips costs more than the general path.
    // The grid goes to the general path early when it has many illegal edges or a long flip cascade.
    let limit = n / 8 + 1024;
    let found = AtomicUsize::new(0);
    // Chunks split by edge, not by band, so a grid with few long bands still scans in parallel.
    let chunks: Option<Vec<Vec<u32>>> =
        map_items((0..ne.div_ceil(SCAN_CHUNK)).collect(), par, |_, c| {
            if found.load(Relaxed) > limit {
                return None;
            }
            let lo = c * SCAN_CHUNK;
            let mut b = toff.partition_point(|&t| 3 * t <= lo) - 1;
            let illegal: Vec<u32> = (lo..ne.min(lo + SCAN_CHUNK))
                .filter(|&e| {
                    while 3 * toff[b + 1] <= e {
                        b += 1;
                    }
                    let tw = he[e];
                    if tw == EMPTY || (tw < pocket && (tw as usize) < e) {
                        return false;
                    }
                    if rect[b] && tw < pocket {
                        let t = e - e % 3;
                        let row_edge = if (tris[t + 1] as usize) < rows[b + 1] {
                            t
                        } else {
                            t + 1
                        };
                        if e != row_edge {
                            return false;
                        }
                    }
                    edge_illegal(q, &tris, &he, e)
                })
                .map(|e| e as u32)
                .collect();
            (found.fetch_add(illegal.len(), Relaxed) + illegal.len() <= limit).then_some(illegal)
        })
        .into_iter()
        .collect();
    let mut todo = chunks?.concat();
    let budget = 4 * todo.len() + n / 32 + 1024;
    todo.extend((pocket..tris.len() as u32).filter(|&e| {
        let tw = he[e as usize];
        tw != EMPTY && (tw < pocket || tw > e)
    }));

    if !lawson(q, &mut tris, &mut he, &mut todo, budget) {
        return None;
    }
    Some(tris)
}

#[allow(clippy::float_cmp)]
fn triangulate_row_grid(pts: &[Pt]) -> Option<Vec<u32>> {
    let g = row_grid(pts)?;
    let q: &[Pt] = &g.q;
    let rows = &g.rows;
    let band_rect = |b: usize| {
        let lower = &q[rows[b]..rows[b + 1]];
        let upper = &q[rows[b + 1]..rows[b + 2]];
        lower.len() == upper.len() && lower.iter().zip(upper).all(|(l, u)| l[0] == u[0])
    };
    let par = q.len() >= PARALLEL_MIN_N;
    let rect: Vec<bool> = map_items((0..rows.len() - 2).collect(), par, |_, b| band_rect(b));
    if rect.iter().all(|&r| r) {
        return Some(tensor_triangles(rows, g.flip));
    }
    let mut tris = zipper_lawson(q, rows, &rect)?;
    if g.flip && par {
        tris.par_chunks_mut(3)
            .with_min_len(4096)
            .for_each(|t| t.swap(1, 2));
    } else if g.flip {
        tris.chunks_mut(3).for_each(|t| t.swap(1, 2));
    }
    Some(tris)
}

#[must_use]
pub fn triangulate(pts: &[Pt]) -> Vec<u32> {
    if let Some(tris) = triangulate_row_grid(pts) {
        return tris;
    }
    if pts.len() >= PARALLEL_MIN_N {
        if let Some(tris) = triangulate_parallel(pts) {
            return tris;
        }
    }
    triangulate_seq(pts)
}

type SortKey = (i64, i64, u32);

#[inline]
fn sort_key(p: Pt, i: u32) -> SortKey {
    (total_order_key(p[0]), total_order_key(p[1]), i)
}

#[inline]
fn key_point(k: &SortKey) -> Pt {
    [from_total_order_key(k.0), from_total_order_key(k.1)]
}

// Keeps the lowest input index of each set of equal points, as the base dedup does.
fn sorted_unique(mut keyed: Vec<SortKey>) -> Vec<SortKey> {
    if keyed.len() >= PAR_SORT_MIN_N {
        keyed.par_sort_unstable();
    } else {
        keyed.sort_unstable();
    }
    keyed.dedup_by(|cur, last| same_point(key_point(last), key_point(cur)));
    keyed
}

fn triangulate_parallel(pts: &[Pt]) -> Option<Vec<u32>> {
    let keyed = sorted_unique(
        pts.par_iter()
            .enumerate()
            .map(|(i, &p)| sort_key(p, i as u32))
            .collect(),
    );
    let m = keyed.len();
    if m < 3 {
        return Some(Vec::new());
    }
    let sites: Vec<u32> = keyed.par_iter().map(|k| k.2).collect();
    let spts: Vec<Pt> = keyed.par_iter().map(key_point).collect();
    drop(keyed);
    let (a, b) = (spts[0], spts[m - 1]);
    if spts.iter().all(|&p| ccw(a, b, p) == 0.0) {
        return Some(Vec::new());
    }
    // Vertex ids are sorted ranks for memory locality. `extract_triangles` maps them back to input ids.

    let nstrips = strip_count(m);
    if nstrips < 2 {
        return None;
    }

    let bounds: Vec<usize> = (0..=nstrips).map(|s| s * m / nstrips).collect();
    let ranges: Vec<(usize, usize)> = (0..nstrips)
        .map(|s| (bounds[s], bounds[s + 1]))
        .filter(|&(lo, hi)| hi > lo)
        .collect();

    let parts: Vec<StripResult> = ranges
        .par_iter()
        .map(|&(lo, hi)| {
            let mut qe = QuadEdge::with_capacity(&spts, 3 * (hi - lo));
            let (ldo, rdo) = qe.delaunay(lo as u32, hi as u32)?;
            Some((qe.onext, qe.data, ldo, rdo))
        })
        .collect::<Option<Vec<StripResult>>>()?;

    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let merge_headroom = (total / 4).max(1024);
    let mut handles: Vec<(u32, u32)> = Vec::with_capacity(parts.len());
    let mut combined = QuadEdge {
        pts: &spts,
        onext: Vec::with_capacity(total + merge_headroom),
        data: Vec::with_capacity(total + merge_headroom),
        free: Vec::new(),
    };
    for (onext, data, ldo, rdo) in parts {
        let off = combined.onext.len() as u32;
        combined.onext.extend(onext.into_iter().map(|x| x + off));
        combined.data.extend(data);
        handles.push((ldo + off, rdo + off));
    }

    let (mut acc_ldo, mut acc_rdi) = handles[0];
    for &(ldo, rdo) in &handles[1..] {
        let (nldo, nrdo) = combined.merge(acc_ldo, acc_rdi, ldo, rdo)?;
        acc_ldo = nldo;
        acc_rdi = nrdo;
    }
    let _ = acc_ldo;

    Some(extract_triangles(&combined, &sites))
}

fn strip_count(m: usize) -> usize {
    let cores = rayon::current_num_threads().max(1);
    let target = cores.saturating_sub(2).clamp(1, 64);
    let by_size = m / STRIP_MIN_PTS;
    target.min(by_size.max(1))
}

fn triangulate_seq(pts: &[Pt]) -> Vec<u32> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }

    let Some((i0, i1, i2)) = find_seed(pts) else {
        return Vec::new();
    };
    let center = circumcenter(pts[i0], pts[i1], pts[i2]);

    let mut keyed: Vec<(i64, u32)> = pts
        .iter()
        .enumerate()
        .map(|(i, &p)| (total_order_key(dist2(center, p)), i as u32))
        .collect();
    keyed.sort_unstable_by(|&(ka, a), &(kb, b)| {
        ka.cmp(&kb)
            .then_with(|| sort_key(pts[a as usize], a).cmp(&sort_key(pts[b as usize], b)))
    });

    let perm: Vec<u32> = keyed.iter().map(|&(_, i)| i).collect();
    drop(keyed);
    let permuted: Vec<Pt> = perm.iter().map(|&i| pts[i as usize]).collect();
    let pos = |old: usize| perm.iter().position(|&i| i as usize == old).unwrap_or(0);
    let seed = (pos(i0), pos(i1), pos(i2));

    let hash_size = ((n as f64).sqrt().ceil() as usize).next_power_of_two();
    let mut sweep = Sweep {
        pts: &permuted,
        triangles: Vec::with_capacity(3 * (2 * n - 5)),
        halfedges: Vec::with_capacity(3 * (2 * n - 5)),
        hull_prev: vec![EMPTY; n],
        hull_next: vec![EMPTY; n],
        hull_tri: vec![EMPTY; n],
        hull_hash: vec![EMPTY; hash_size],
        hash_size,
        center: [0.0, 0.0],
        edge_stack: Vec::new(),
    };
    sweep.run(center, seed);
    let mut tris = sweep.triangles;
    for t in &mut tris {
        *t = perm[*t as usize];
    }
    tris
}

struct QuadEdge<'a> {
    pts: &'a [Pt],
    onext: Vec<u32>,
    data: Vec<u32>,
    free: Vec<u32>,
}

impl<'a> QuadEdge<'a> {
    fn with_capacity(pts: &'a [Pt], cap: usize) -> Self {
        QuadEdge {
            pts,
            onext: Vec::with_capacity(4 * cap),
            data: Vec::with_capacity(4 * cap),
            free: Vec::new(),
        }
    }

    #[inline]
    fn rot(e: u32) -> u32 {
        (e & !3) | ((e + 1) & 3)
    }

    #[inline]
    fn inv_rot(e: u32) -> u32 {
        (e & !3) | ((e + 3) & 3)
    }

    #[inline]
    fn sym(e: u32) -> u32 {
        (e & !3) | ((e + 2) & 3)
    }

    #[inline]
    fn onext(&self, e: u32) -> u32 {
        self.onext[e as usize]
    }

    #[inline]
    fn oprev(&self, e: u32) -> u32 {
        Self::rot(self.onext(Self::rot(e)))
    }

    #[inline]
    fn lnext(&self, e: u32) -> u32 {
        Self::rot(self.onext(Self::inv_rot(e)))
    }

    #[inline]
    fn org(&self, e: u32) -> u32 {
        self.data[e as usize]
    }

    #[inline]
    fn dest(&self, e: u32) -> u32 {
        self.data[Self::sym(e) as usize]
    }

    #[inline]
    fn p(&self, v: u32) -> Pt {
        self.pts[v as usize]
    }

    #[inline]
    fn make_edge(&mut self, org: u32, dest: u32) -> u32 {
        let base = if let Some(slot) = self.free.pop() {
            slot
        } else {
            let b = self.onext.len() as u32;
            self.onext.extend_from_slice(&[0, 0, 0, 0]);
            self.data.extend_from_slice(&[EMPTY, EMPTY, EMPTY, EMPTY]);
            b
        };
        self.onext[base as usize] = base;
        self.onext[(base + 1) as usize] = base + 3;
        self.onext[(base + 2) as usize] = base + 2;
        self.onext[(base + 3) as usize] = base + 1;
        self.data[base as usize] = org;
        self.data[(base + 2) as usize] = dest;
        self.data[(base + 1) as usize] = EMPTY;
        self.data[(base + 3) as usize] = EMPTY;
        base
    }

    #[inline]
    fn splice(&mut self, a: u32, b: u32) {
        let a_on = self.onext(a);
        let b_on = self.onext(b);
        let alpha = Self::rot(a_on);
        let beta = Self::rot(b_on);
        let al_on = self.onext(alpha);
        let be_on = self.onext(beta);
        self.onext[a as usize] = b_on;
        self.onext[b as usize] = a_on;
        self.onext[alpha as usize] = be_on;
        self.onext[beta as usize] = al_on;
    }

    #[inline]
    fn connect(&mut self, a: u32, b: u32) -> u32 {
        let da = self.dest(a);
        let ob = self.org(b);
        let e = self.make_edge(da, ob);
        let la = self.lnext(a);
        self.splice(e, la);
        let se = Self::sym(e);
        self.splice(se, b);
        e
    }

    #[inline]
    fn delete_edge(&mut self, e: u32) {
        let op = self.oprev(e);
        self.splice(e, op);
        let se = Self::sym(e);
        let sop = self.oprev(se);
        self.splice(se, sop);
        self.free.push(e & !3);
    }

    #[inline]
    fn right_of(&self, v: u32, e: u32) -> bool {
        ccw(self.p(v), self.p(self.dest(e)), self.p(self.org(e))) > 0.0
    }

    #[inline]
    fn left_of(&self, v: u32, e: u32) -> bool {
        ccw(self.p(v), self.p(self.org(e)), self.p(self.dest(e))) > 0.0
    }
}

impl QuadEdge<'_> {
    fn merge(&mut self, ldo: u32, ldi: u32, rdi: u32, rdo: u32) -> Option<(u32, u32)> {
        let (mut ldo, mut ldi, mut rdi, rdo) = (ldo, ldi, rdi, rdo);

        let cap = 4 * self.onext.len() + 64;
        let mut guard = 0usize;
        loop {
            if self.left_of(self.org(rdi), ldi) {
                ldi = self.lnext(ldi);
            } else if self.right_of(self.org(ldi), rdi) {
                rdi = self.onext(Self::sym(rdi));
            } else {
                break;
            }
            guard += 1;
            if guard > cap {
                return None;
            }
        }

        let mut basel = self.connect(Self::sym(rdi), ldi);
        if self.org(ldi) == self.org(ldo) {
            ldo = Self::sym(basel);
        }
        let mut rdo = rdo;
        if self.org(rdi) == self.org(rdo) {
            rdo = basel;
        }

        guard = 0;
        loop {
            guard += 1;
            if guard > cap {
                return None;
            }
            let bd = self.p(self.dest(basel));
            let bo = self.p(self.org(basel));
            let mut lcand = self.onext(Self::sym(basel));
            let mut lp = self.p(self.dest(lcand));
            let mut l_valid = ccw(lp, bd, bo) > 0.0;
            if l_valid {
                let mut g = 0usize;
                loop {
                    let t = self.onext(lcand);
                    let tp = self.p(self.dest(t));
                    if !in_circle(bd, bo, lp, tp) {
                        break;
                    }
                    self.delete_edge(lcand);
                    lcand = t;
                    lp = tp;
                    g += 1;
                    if g > cap {
                        return None;
                    }
                }
                if g > 0 {
                    l_valid = ccw(lp, bd, bo) > 0.0;
                }
            }

            let mut rcand = self.oprev(basel);
            let mut rp = self.p(self.dest(rcand));
            let mut r_valid = ccw(rp, bd, bo) > 0.0;
            if r_valid {
                let mut g = 0usize;
                loop {
                    let t = self.oprev(rcand);
                    let tp = self.p(self.dest(t));
                    if !in_circle(bd, bo, rp, tp) {
                        break;
                    }
                    self.delete_edge(rcand);
                    rcand = t;
                    rp = tp;
                    g += 1;
                    if g > cap {
                        return None;
                    }
                }
                if g > 0 {
                    r_valid = ccw(rp, bd, bo) > 0.0;
                }
            }

            if !l_valid && !r_valid {
                break;
            }

            if !l_valid || (r_valid && in_circle(lp, bd, bo, rp)) {
                basel = self.connect(rcand, Self::sym(basel));
            } else {
                basel = self.connect(Self::sym(basel), Self::sym(lcand));
            }
        }

        Some((ldo, rdo))
    }

    fn base2(&mut self, s0: u32, s1: u32) -> (u32, u32) {
        let a = self.make_edge(s0, s1);
        (a, Self::sym(a))
    }

    fn base3(&mut self, s0: u32, s1: u32, s2: u32) -> (u32, u32) {
        let a = self.make_edge(s0, s1);
        let b = self.make_edge(s1, s2);
        self.splice(Self::sym(a), b);
        let turn = ccw(self.p(s0), self.p(s1), self.p(s2));
        if turn > 0.0 {
            let _c = self.connect(b, a);
            (a, Self::sym(b))
        } else if turn < 0.0 {
            let c = self.connect(b, a);
            (Self::sym(c), c)
        } else {
            (a, Self::sym(b))
        }
    }

    fn delaunay(&mut self, lo: u32, hi: u32) -> Option<(u32, u32)> {
        let n = hi - lo;
        if n == 2 {
            return Some(self.base2(lo, lo + 1));
        }
        if n == 3 {
            return Some(self.base3(lo, lo + 1, lo + 2));
        }
        let mid = lo + n / 2;
        let (ldo, ldi) = self.delaunay(lo, mid)?;
        let (rdi, rdo) = self.delaunay(mid, hi)?;
        self.merge(ldo, ldi, rdi, rdo)
    }
}

// Emits each face from its lowest-numbered edge, which is the edge a serial scan meets first.
// That keeps the serial output order with no shared `seen` state, so the scan runs in parallel.
fn extract_triangles(qe: &QuadEdge<'_>, sites: &[u32]) -> Vec<u32> {
    let mut dead = vec![false; qe.onext.len() / 4];
    for &slot in &qe.free {
        dead[slot as usize / 4] = true;
    }
    let parts: Vec<Vec<u32>> = dead
        .par_chunks(EXTRACT_CHUNK_QUADS)
        .enumerate()
        .map(|(c, chunk)| {
            let mut out = Vec::with_capacity(2 * chunk.len() + 16);
            for (k, &is_dead) in chunk.iter().enumerate() {
                let start = ((c * EXTRACT_CHUNK_QUADS + k) * 4) as u32;
                if is_dead || qe.org(start) == EMPTY {
                    continue;
                }
                for e0 in [start, QuadEdge::sym(start)] {
                    let e1 = qe.lnext(e0);
                    let e2 = qe.lnext(e1);
                    if e1 < e0 || e2 < e0 || qe.lnext(e2) != e0 {
                        continue;
                    }
                    let a = qe.org(e0);
                    let b = qe.org(e1);
                    let c = qe.org(e2);
                    if ccw(qe.p(a), qe.p(b), qe.p(c)) <= 0.0 {
                        continue;
                    }
                    out.extend_from_slice(&[
                        sites[a as usize],
                        sites[b as usize],
                        sites[c as usize],
                    ]);
                }
            }
            out
        })
        .collect();

    let mut out = vec![0u32; parts.iter().map(Vec::len).sum()];
    let mut rest = out.as_mut_slice();
    let mut dests = Vec::with_capacity(parts.len());
    for part in &parts {
        let (head, tail) = rest.split_at_mut(part.len());
        dests.push(head);
        rest = tail;
    }
    dests
        .into_par_iter()
        .zip(parts.par_iter())
        .for_each(|(dst, src)| dst.copy_from_slice(src));
    out
}

#[allow(clippy::float_cmp)]
fn same_point(p: Pt, q: Pt) -> bool {
    p[0] == q[0] && p[1] == q[1]
}

#[allow(clippy::float_cmp)]
fn is_collinear(uniq: &[(Pt, u32)]) -> bool {
    let a = uniq[0].0;
    let b = uniq[uniq.len() - 1].0;
    uniq.iter().all(|&(p, _)| ccw(a, b, p) == 0.0)
}

fn half_chain<'a>(seq: impl Iterator<Item = &'a (Pt, u32)>) -> Vec<(Pt, u32)> {
    let mut chain: Vec<(Pt, u32)> = Vec::new();
    for &(p, i) in seq {
        while chain.len() >= 2 && ccw(chain[chain.len() - 2].0, chain[chain.len() - 1].0, p) < 0.0 {
            chain.pop();
        }
        chain.push((p, i));
    }
    chain
}

#[derive(Clone, Copy)]
struct Extremes {
    lo: [(f64, usize); 4],
    hi: [(f64, usize); 4],
}

#[inline]
fn extreme_keys(p: Pt) -> [f64; 4] {
    [p[0], p[1], p[0] + p[1], p[0] - p[1]]
}

type Bounds = ([f64; 4], [f64; 4]);

#[inline]
fn widen(lo: &mut [f64; 4], hi: &mut [f64; 4], p: Pt) {
    let keys = extreme_keys(p);
    for ((l, h), k) in lo.iter_mut().zip(hi.iter_mut()).zip(keys) {
        *l = l.min(k);
        *h = h.max(k);
    }
}

fn block_bounds(block: &[Pt]) -> Bounds {
    let mut acc: [Bounds; BOUNDS_LANES] =
        [([f64::INFINITY; 4], [f64::NEG_INFINITY; 4]); BOUNDS_LANES];
    let (groups, rest) = block.as_chunks::<BOUNDS_LANES>();
    for group in groups {
        for ((lo, hi), &p) in acc.iter_mut().zip(group) {
            widen(lo, hi, p);
        }
    }
    let (mut lo, mut hi) = acc[0];
    for &p in rest {
        widen(&mut lo, &mut hi, p);
    }
    for (l, h) in &acc[1..] {
        lo = std::array::from_fn(|d| lo[d].min(l[d]));
        hi = std::array::from_fn(|d| hi[d].max(h[d]));
    }
    (lo, hi)
}

#[allow(clippy::float_cmp)]
fn locate(block: &[Pt], d: usize, key: f64) -> usize {
    block
        .iter()
        .position(|&p| extreme_keys(p)[d] == key)
        .unwrap_or(0)
}

fn scan_extremes(pts: &[Pt], offset: usize) -> Extremes {
    let mut ex = Extremes {
        lo: [(f64::INFINITY, offset); 4],
        hi: [(f64::NEG_INFINITY, offset); 4],
    };
    for (c, block) in pts.chunks(EXTREME_BLOCK).enumerate() {
        let start = offset + c * EXTREME_BLOCK;
        let (lo, hi) = block_bounds(block);
        for d in 0..4 {
            if lo[d] < ex.lo[d].0 {
                ex.lo[d] = (lo[d], start + locate(block, d, lo[d]));
            }
            if hi[d] > ex.hi[d].0 {
                ex.hi[d] = (hi[d], start + locate(block, d, hi[d]));
            }
        }
    }
    ex
}

fn merge_extremes(a: Extremes, b: Extremes) -> Extremes {
    Extremes {
        lo: std::array::from_fn(|d| {
            if b.lo[d].0 < a.lo[d].0 {
                b.lo[d]
            } else {
                a.lo[d]
            }
        }),
        hi: std::array::from_fn(|d| {
            if b.hi[d].0 > a.hi[d].0 {
                b.hi[d]
            } else {
                a.hi[d]
            }
        }),
    }
}

fn find_extremes(pts: &[Pt]) -> Extremes {
    if pts.len() < PARALLEL_MIN_N {
        return scan_extremes(pts, 0);
    }
    pts.par_chunks(HULL_SCAN_CHUNK)
        .enumerate()
        .map(|(c, chunk)| scan_extremes(chunk, c * HULL_SCAN_CHUNK))
        .reduce_with(merge_extremes)
        .unwrap_or_else(|| scan_extremes(pts, 0))
}

#[inline]
fn certainly_left(a: Pt, b: Pt, p: Pt) -> bool {
    let detleft = (a[0] - p[0]) * (b[1] - p[1]);
    let detright = (a[1] - p[1]) * (b[0] - p[0]);
    // Add an absolute slack: the relative bound does not hold when the products underflow.
    detleft - detright > CCW_ERRBOUND_A * (detleft.abs() + detright.abs()) + UNDERFLOW_SLACK
}

fn predicate_safe(p: Pt) -> bool {
    p.iter()
        .all(|v| *v == 0.0 || (SAFE_MIN_ABS..=SAFE_MAX_ABS).contains(&v.abs()))
}

#[inline]
fn in_rect([x_lo, x_hi, y_lo, y_hi]: [f64; 4], p: Pt) -> bool {
    p[0] > x_lo && p[0] < x_hi && p[1] > y_lo && p[1] < y_hi
}

struct InteriorTest {
    ring: Vec<Pt>,
    rect: Option<[f64; 4]>,
}

impl InteriorTest {
    fn new(pts: &[Pt], ex: &Extremes) -> Option<Self> {
        let [min_x, min_y, min_s, min_t] = ex.lo.map(|(_, i)| pts[i]);
        let [max_x, max_y, max_s, max_t] = ex.hi.map(|(_, i)| pts[i]);
        let octagon = [min_y, max_t, max_x, max_s, max_y, min_t, min_x, min_s];
        let mut ring: Vec<Pt> = Vec::with_capacity(8);
        for p in octagon {
            if ring.last().is_none_or(|&q| !same_point(q, p)) {
                ring.push(p);
            }
        }
        while ring.len() > 1 && same_point(ring[0], ring[ring.len() - 1]) {
            ring.pop();
        }
        if ring.len() < 3 {
            return None;
        }

        let x_lo = min_x[0].max(min_s[0]).max(min_t[0]);
        let x_hi = max_x[0].min(max_s[0]).min(max_t[0]);
        let y_lo = min_y[1].max(min_s[1]).max(max_t[1]);
        let y_hi = max_y[1].min(max_s[1]).min(min_t[1]);
        let corners = [[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]];
        let k = ring.len();
        // Use the box only when all corners are in the ring, so its interior is strictly inside the hull.
        let rect = (x_lo < x_hi
            && y_lo < y_hi
            && octagon.iter().chain(&corners).all(|&p| predicate_safe(p))
            && corners
                .iter()
                .all(|&c| (0..k).all(|e| ccw(ring[e], ring[(e + 1) % k], c) >= 0.0)))
        .then_some([x_lo, x_hi, y_lo, y_hi]);
        Some(Self { ring, rect })
    }

    #[inline]
    fn contains(&self, p: Pt) -> bool {
        if self.rect.is_some_and(|rect| in_rect(rect, p)) {
            return true;
        }
        let k = self.ring.len();
        (0..k).all(|e| certainly_left(self.ring[e], self.ring[(e + 1) % k], p))
    }
}

fn outside_rect(pts: &[Pt], offset: usize, rect: [f64; 4]) -> Vec<usize> {
    let mut out = Vec::new();
    for (c, block) in pts.chunks(64).enumerate() {
        let mut mask = block
            .iter()
            .enumerate()
            .fold(0u64, |m, (j, &p)| m | (u64::from(!in_rect(rect, p)) << j));
        while mask != 0 {
            out.push(offset + c * 64 + mask.trailing_zeros() as usize);
            mask &= mask - 1;
        }
    }
    out
}

// Any input points give a valid filter; the sample only sets how many points survive it.
fn sampled_extremes(pts: &[Pt]) -> Extremes {
    let stride = (pts.len() / EXTREME_SAMPLE).max(1);
    let sample: Vec<Pt> = pts.iter().step_by(stride).copied().collect();
    let ex = scan_extremes(&sample, 0);
    Extremes {
        lo: ex.lo.map(|(k, i)| (k, i * stride)),
        hi: ex.hi.map(|(k, i)| (k, i * stride)),
    }
}

fn extremes_of(pts: &[Pt], idx: &[usize]) -> Extremes {
    if idx.len() == pts.len() {
        return find_extremes(pts);
    }
    let sub: Vec<Pt> = idx.iter().map(|&i| pts[i]).collect();
    let ex = scan_extremes(&sub, 0);
    Extremes {
        lo: ex.lo.map(|(k, i)| (k, idx[i])),
        hi: ex.hi.map(|(k, i)| (k, idx[i])),
    }
}

fn hull_candidates(pts: &[Pt]) -> Vec<usize> {
    let n = pts.len();
    if n < HULL_FILTER_MIN_N {
        return (0..n).collect();
    }
    let coarse: Vec<usize> =
        match InteriorTest::new(pts, &sampled_extremes(pts)).and_then(|t| t.rect) {
            Some(rect) if n >= PARALLEL_MIN_N => pts
                .par_chunks(HULL_SCAN_CHUNK)
                .enumerate()
                .flat_map_iter(|(c, chunk)| outside_rect(chunk, c * HULL_SCAN_CHUNK, rect))
                .collect(),
            Some(rect) => outside_rect(pts, 0, rect),
            None => (0..n).collect(),
        };
    match InteriorTest::new(pts, &extremes_of(pts, &coarse)) {
        None => coarse,
        Some(t) if coarse.len() >= PARALLEL_MIN_N => coarse
            .into_par_iter()
            .with_min_len(HULL_SCAN_CHUNK)
            .filter(|&i| !t.contains(pts[i]))
            .collect(),
        Some(t) => coarse
            .into_iter()
            .filter(|&i| !t.contains(pts[i]))
            .collect(),
    }
}

#[must_use]
pub fn hull_edges(pts: &[Pt]) -> Vec<[i32; 2]> {
    let keyed: Vec<SortKey> = hull_candidates(pts)
        .into_iter()
        .map(|i| sort_key(pts[i], i as u32))
        .collect();
    let uniq: Vec<(Pt, u32)> = sorted_unique(keyed)
        .iter()
        .map(|k| (key_point(k), k.2))
        .collect();

    if uniq.len() < 3 || is_collinear(&uniq) {
        return Vec::new();
    }

    let mut hull = half_chain(uniq.iter());
    hull.pop();
    let mut upper = half_chain(uniq.iter().rev());
    upper.pop();
    hull.append(&mut upper);

    let k = hull.len();
    (0..k)
        .map(|i| [hull[i].1 as i32, hull[(i + 1) % k].1 as i32])
        .collect()
}

#[cfg(feature = "python")]
fn collect_xy<'a>(points: &'a PyReadonlyArray2<'_, f64>) -> Cow<'a, [Pt]> {
    points.as_slice().map_or_else(
        |_| {
            Cow::Owned(
                points
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|r| [r[0], r[1]])
                    .collect(),
            )
        },
        |flat| {
            Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(flat.as_ptr().cast::<Pt>(), flat.len() / 2)
            })
        },
    )
}

#[cfg(feature = "python")]
fn into_index_array(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    data: Vec<i32>,
) -> PyResult<Bound<'_, PyArray2<i32>>> {
    Array2::from_shape_vec((rows, cols), data)
        .map(|arr| arr.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn delaunay<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let coords = collect_xy(&points);
    let tris = triangulate(coords.as_ref());
    let rows = tris.len() / 3;
    let data: Vec<i32> = tris.into_iter().map(|t| t as i32).collect();
    into_index_array(py, rows, 3, data)
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn convex_hull<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let coords = collect_xy(&points);
    let edges = hull_edges(coords.as_ref());
    let rows = edges.len();
    let data: Vec<i32> = edges.into_iter().flatten().collect();
    into_index_array(py, rows, 2, data)
}

#[cfg(feature = "python")]
#[pymodule]
fn _triangular(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delaunay, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ccw, hull_edges, in_circle, sample_needs_many_flips, triangulate, triangulate_seq, Pt,
    };

    const PTS: [[f64; 2]; 5] = [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0], [1.0, 0.0]];

    fn assert_delaunay(pts: &[Pt], tris: &[u32]) {
        assert_eq!(tris.len(), triangulate_seq(pts).len());
        for t in tris.chunks(3) {
            let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
            assert!(ccw(a, b, c) > 0.0);
            assert!(pts.iter().all(|&p| !in_circle(a, b, c, p)));
        }
    }

    fn grid(nu: u32, nv: u32, f: impl Fn(f64, f64) -> Pt) -> Vec<Pt> {
        (0..nv)
            .flat_map(|j| (0..nu).map(move |i| (f64::from(i), f64::from(j * j))))
            .map(|(u, v)| f(u, v))
            .collect()
    }

    #[test]
    fn triangulates_tensor_grids_counter_clockwise_in_every_frame() {
        let frames: [fn(f64, f64) -> Pt; 6] = [
            |u, v| [u * 0.5, v],
            |u, v| [-u * 0.5, v],
            |u, v| [u * 0.5, 2.0 - v],
            |u, v| [v, u * 0.5],
            |u, v| [2.0 - v, u * 0.5],
            |u, v| [-v, -u * 0.5],
        ];
        for f in frames {
            let pts = grid(7, 5, f);
            let tris = triangulate(&pts);
            assert_eq!(tris.len(), 3 * 2 * 6 * 4);
            assert_delaunay(&pts, &tris);
        }
    }

    #[test]
    fn triangulates_reduced_row_grids_counter_clockwise_in_every_frame() {
        let counts = [3u32, 7, 12, 12, 16, 12, 5, 4];
        let rows: Vec<Pt> = counts
            .iter()
            .enumerate()
            .flat_map(|(r, &c)| {
                let start = if r == 0 || r == 6 { -1.5 } else { 0.0 };
                (0..c).map(move |k| [start + f64::from(k) * 30.0 / f64::from(c), -0.4 * r as f64])
            })
            .collect();
        let frames: [fn(Pt) -> Pt; 4] = [
            |p| p,
            |p| [-p[0], p[1]],
            |p| [p[1], p[0]],
            |p| [-p[1], -p[0]],
        ];
        for f in frames {
            let pts: Vec<Pt> = rows.iter().map(|&p| f(p)).collect();
            assert_delaunay(&pts, &triangulate(&pts));
        }
    }

    #[test]
    fn subnormal_row_grids_use_the_general_path() {
        let cases: [&[Pt]; 3] = [
            &[
                [5e-324, 5e-324],
                [1.0, 5e-324],
                [0.0, 0.5],
                [1.0, 0.5],
                [2.0, 0.5],
            ],
            &[
                [1.5e-323, 0.0],
                [0.3, 0.0],
                [0.0, 0.5],
                [0.3, 0.5],
                [0.0, 1.0],
                [0.5, 1.0],
            ],
            &[
                [1e-323, 5e-324],
                [0.5, 5e-324],
                [1.0, 5e-324],
                [0.0, 0.25],
                [0.3, 0.25],
                [0.6, 0.25],
                [0.899_999_999_999_999_9, 0.25],
                [1e-323, 0.5],
                [1.0, 0.5],
                [1.5e-323, 0.75],
                [0.5, 0.75],
                [1.0, 0.75],
            ],
        ];
        for pts in cases {
            assert_eq!(triangulate(pts), triangulate_seq(pts));
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn sample_flags_scan_lines_and_fans_but_not_reduced_grids() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut sorted_row = |m: usize| {
            let mut xs: Vec<f64> = (0..m)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (seed >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect();
            xs.sort_by(f64::total_cmp);
            xs
        };
        let scan_rows = |xs: Vec<Vec<f64>>, dy: f64| -> Vec<Pt> {
            xs.into_iter()
                .zip(0u32..)
                .flat_map(|(row, r)| row.into_iter().map(move |x| [x, f64::from(r) * dy]))
                .collect()
        };
        let scan = scan_rows((0..10).map(|_| sorted_row(5000)).collect(), 1e-6);
        let small_scan = scan_rows((0..100).map(|_| sorted_row(50)).collect(), 1e-6);
        let shared = sorted_row(20);
        let mixed = scan_rows(
            (0..100)
                .map(|r| {
                    if r < 35 {
                        sorted_row(20)
                    } else {
                        shared.clone()
                    }
                })
                .collect(),
            1e-6 / 20.0,
        );
        let dense = (1..20_000).map(|k| f64::from(k) / 20_000.0);
        let fan: Vec<Pt> = dense
            .clone()
            .map(|x| [x, 0.0])
            .chain([[0.0, 1e-9], [1.0, 1e-9]])
            .chain(dense.map(|x| [x, 1.0]))
            .collect();
        let reduced: Vec<Pt> = (0..100u32)
            .flat_map(|r| {
                let c = 300 + 4 * r.min(99 - r);
                (0..c).map(move |k| [f64::from(k) / f64::from(c), f64::from(r) * 2e-3])
            })
            .collect();
        let small_reduced: Vec<Pt> = (0..10u32)
            .flat_map(|r| {
                let c = 6 + r.min(9 - r);
                (0..c).map(move |k| [f64::from(k) / f64::from(c), f64::from(r) * 0.1])
            })
            .collect();
        for (pts, flagged) in [
            (&scan, true),
            (&fan, true),
            (&reduced, false),
            (&small_scan, true),
            (&mixed, true),
            (&small_reduced, false),
        ] {
            let mut rows = vec![0usize];
            rows.extend((1..pts.len()).filter(|&i| pts[i][1] != pts[i - 1][1]));
            rows.push(pts.len());
            assert_eq!(sample_needs_many_flips(pts, &rows), flagged);
            assert_eq!(triangulate(pts).len(), triangulate_seq(pts).len());
        }
    }

    #[test]
    fn triangulates_readme_example() {
        let tris = triangulate(&PTS);
        assert_eq!(tris.len(), 4 * 3);
        assert!(tris.chunks(3).all(|t| t.contains(&2)));
    }

    #[test]
    fn hulls_readme_example() {
        let mut edges: Vec<[i32; 2]> = hull_edges(&PTS)
            .into_iter()
            .map(|[a, b]| [a.min(b), a.max(b)])
            .collect();
        edges.sort_unstable();
        assert_eq!(edges, vec![[0, 1], [0, 4], [1, 3], [3, 4]]);
    }
}
