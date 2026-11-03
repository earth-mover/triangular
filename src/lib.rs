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
#[cfg(feature = "python")]
use std::borrow::Cow;

mod exact;

const EMPTY: u32 = u32::MAX;

const PARALLEL_MIN_N: usize = 50_000;
const DELAUNAY_PARALLEL_MIN_N: usize = 10_000;

const STRIP_MIN_PTS: usize = 1024;
const STRIP_TARGET_PTS: usize = 4096;
// A planar graph on k sites has at most 3k edges, so 3 quad edges per site never run out.
const QUAD_SLOTS: usize = 12;
const BUCKETS_PER_STRIP: usize = 16;
const AXIS_SAMPLE: usize = 4096;
const GRID_COLUMN_MIN_PTS: usize = 128;
// Uniform input flips about 3 times per point. Rows of collinear points flip 30 or more times,
// and divide and conquer is faster on them.
const STRIP_MAX_FLIPS_PER_PT: usize = 12;
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

const PRED_EPSILON: f64 = f64::EPSILON * 0.5;
const CCW_ERRBOUND_A: f64 = (3.0 + 16.0 * PRED_EPSILON) * PRED_EPSILON;
const ICC_ERRBOUND_A: f64 = (10.0 + 96.0 * PRED_EPSILON) * PRED_EPSILON;

pub type Pt = [f64; 2];

type Sites = (Vec<Pt>, Vec<u32>, Vec<(usize, usize)>);
type Grouped = (Vec<SortKey>, Vec<usize>);

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
    in_circle_slow(a, b, c, p, det)
}

#[cold]
#[inline(never)]
fn in_circle_slow(a: Pt, b: Pt, c: Pt, p: Pt, det: f64) -> bool {
    if small_dyadic(a, b, c, p) {
        return det > 0.0;
    }
    exact::incircle(a, b, c, p).unwrap_or_else(|| incircle(co(a), co(b), co(c), co(p))) > 0.0
}

#[inline]
fn exact_diff(a: f64, b: f64) -> bool {
    let x = a - b;
    let bvirt = a - x;
    let avirt = x + bvirt;
    (bvirt - b) + (a - avirt) == 0.0
}

// True when the differences to p are exact integer multiples of 2^e below 2^(e+12): then
// every in_circle term stays below 2^53 and the float determinant is exact.
#[inline]
#[allow(clippy::float_cmp)]
fn small_dyadic(a: Pt, b: Pt, c: Pt, p: Pt) -> bool {
    let mut m = 0.0f64;
    let mut d = [0.0f64; 6];
    for (k, q) in [a, b, c].into_iter().enumerate() {
        for j in 0..2 {
            if !exact_diff(q[j], p[j]) {
                return false;
            }
            d[2 * k + j] = q[j] - p[j];
            m = m.max(d[2 * k + j].abs());
        }
    }
    let exp = ((m.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if !(-240..=240).contains(&exp) {
        return false;
    }
    let scale = f64::from_bits(((1023 - (exp + 1 - 12)) as u64) << 52);
    d.iter().all(|&x| {
        let s = x * scale;
        // A nonzero difference that underflows to 0 is not a multiple of 2^e.
        s == s.trunc() && (s != 0.0 || x == 0.0)
    })
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

    let third = |i: usize| i != i0 && i != i1 && !same_point(p0, pts[i]) && !same_point(p1, pts[i]);
    let mut i2 = argmin((0..n).filter(|&i| third(i)), |i| {
        circumradius2(p0, p1, pts[i])
    })?;
    // A rounded determinant can give an exactly collinear point a finite radius and a zero-area seed.
    if ccw(p0, p1, pts[i2]) == 0.0 {
        i2 = argmin(
            (0..n).filter(|&i| third(i) && ccw(p0, p1, pts[i]) != 0.0),
            |i| circumradius2(p0, p1, pts[i]),
        )?;
    }
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
    flips: usize,
    max_flips: usize,
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
                self.flips += 1;
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

    fn run(&mut self, center: Pt, seed: (usize, usize, usize)) -> bool {
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
            if self.flips > self.max_flips {
                return false;
            }
        }
        true
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

#[must_use]
pub fn triangulate(pts: &[Pt]) -> Vec<u32> {
    // `triangulate_seq` returns the same empty result as the parallel path for degenerate input.
    if pts.len() >= DELAUNAY_PARALLEL_MIN_N && rayon::current_num_threads() >= 2 {
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

fn dedup_into(sorted: &[SortKey], spts: &mut [Pt], ids: &mut [u32]) -> usize {
    let mut k = 0;
    for key in sorted {
        let p = key_point(key);
        if k == 0 || !same_point(spts[k - 1], p) {
            spts[k] = p;
            ids[k] = key.2;
            k += 1;
        }
    }
    k
}

fn sorted_sites(pts: &[Pt]) -> Sites {
    let keyed = sorted_unique(
        pts.par_iter()
            .enumerate()
            .map(|(i, &p)| sort_key(p, i as u32))
            .collect(),
    );
    let m = keyed.len();
    // Dedup can remove most points. Count strips from m so that each strip keeps STRIP_MIN_PTS sites.
    let nstrips = strip_count(m);
    let spts: Vec<Pt> = keyed.par_iter().map(key_point).collect();
    let ids: Vec<u32> = keyed.par_iter().map(|k| k.2).collect();
    let ranges = (0..nstrips)
        .map(|s| (s * m / nstrips, (s + 1) * m / nstrips))
        .filter(|&(lo, hi)| hi > lo)
        .collect();
    (spts, ids, ranges)
}

#[derive(Clone, Copy)]
struct XBuckets {
    xmin: f64,
    scale: f64,
    n: usize,
}

impl XBuckets {
    // The map is monotone in x, so equal x share a bucket and strips never overlap in x.
    #[inline]
    fn of(self, x: f64) -> usize {
        (((x - self.xmin) * self.scale) as usize).min(self.n - 1)
    }
}

struct BucketPlan {
    xb: XBuckets,
    chunk: usize,
    hists: Vec<Vec<u32>>,
    strip_of: Vec<u32>,
    sizes: Vec<usize>,
}

impl BucketPlan {
    #[inline]
    fn strip(&self, x: f64) -> usize {
        self.strip_of[self.xb.of(x)] as usize
    }
}

fn bucket_plan(pts: &[Pt], nstrips: usize) -> Option<BucketPlan> {
    let n = pts.len();
    let (xmin, xmax) = pts.par_iter().map(|p| (p[0], p[0])).reduce(
        || (f64::INFINITY, f64::NEG_INFINITY),
        |a, b| (a.0.min(b.0), a.1.max(b.1)),
    );
    let nbuckets = nstrips * BUCKETS_PER_STRIP;
    let scale = nbuckets as f64 / (xmax - xmin);
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let xb = XBuckets {
        xmin,
        scale,
        n: nbuckets,
    };
    let chunk = n.div_ceil(4 * rayon::current_num_threads()).max(1 << 12);
    let hists: Vec<Vec<u32>> = pts
        .par_chunks(chunk)
        .map(|c| {
            let mut h = vec![0u32; nbuckets];
            for p in c {
                h[xb.of(p[0])] += 1;
            }
            h
        })
        .collect();
    let mut totals = vec![0usize; nbuckets];
    for h in &hists {
        for (t, &c) in totals.iter_mut().zip(h) {
            *t += c as usize;
        }
    }

    let target = n / nstrips;
    let mut strip_of = vec![0u32; nbuckets];
    let mut sizes: Vec<usize> = vec![0];
    for (b, &t) in totals.iter().enumerate() {
        if sizes[sizes.len() - 1] >= target {
            sizes.push(0);
        }
        let last = sizes.len() - 1;
        sizes[last] += t;
        strip_of[b] = last as u32;
    }
    if sizes.len() > 1 && sizes[sizes.len() - 1] < target / 2 {
        let tail = sizes.pop().unwrap_or(0);
        let last = sizes.len() - 1;
        sizes[last] += tail;
        for s in &mut strip_of {
            *s = (*s).min(last as u32);
        }
    }
    if sizes.len() < 2 || sizes.iter().any(|&k| k > 4 * target) {
        return None;
    }
    Some(BucketPlan {
        xb,
        chunk,
        hists,
        strip_of,
        sizes,
    })
}

// Strips are x ranges, each sorted on its own. The global sort in `sorted_sites` handles skewed x.
fn bucketed_sites(pts: &[Pt], nstrips: usize) -> Option<Sites> {
    let plan = bucket_plan(pts, nstrips)?;
    let nstrips = plan.sizes.len();
    let grouped: Vec<Grouped> = pts
        .par_chunks(plan.chunk)
        .zip(plan.hists.par_iter())
        .enumerate()
        .map(|(c, (part, h))| {
            let mut cursor = vec![0usize; nstrips + 1];
            for (b, &k) in h.iter().enumerate() {
                cursor[plan.strip_of[b] as usize + 1] += k as usize;
            }
            for s in 0..nstrips {
                cursor[s + 1] += cursor[s];
            }
            let starts = cursor.clone();
            let mut buf: Vec<SortKey> = vec![(0, 0, 0); part.len()];
            for (j, &p) in part.iter().enumerate() {
                let s = plan.strip(p[0]);
                buf[cursor[s]] = sort_key(p, (c * plan.chunk + j) as u32);
                cursor[s] += 1;
            }
            (buf, starts)
        })
        .collect();

    let n = pts.len();
    let mut spts = vec![[0.0; 2]; n];
    let mut ids = vec![0u32; n];
    let mut jobs = Vec::with_capacity(nstrips);
    let (mut sp_rest, mut id_rest) = (spts.as_mut_slice(), ids.as_mut_slice());
    let mut off = 0;
    for (s, &k) in plan.sizes.iter().enumerate() {
        let (sp, sp_tail) = sp_rest.split_at_mut(k);
        let (id, id_tail) = id_rest.split_at_mut(k);
        jobs.push((s, off, sp, id));
        sp_rest = sp_tail;
        id_rest = id_tail;
        off += k;
    }
    let ranges: Vec<(usize, usize)> = jobs
        .into_par_iter()
        .map(|(s, off, sp, id)| {
            let mut strip: Vec<SortKey> = Vec::with_capacity(sp.len());
            for (buf, starts) in &grouped {
                strip.extend_from_slice(&buf[starts[s]..starts[s + 1]]);
            }
            strip.sort_unstable();
            (off, off + dedup_into(&strip, sp, id))
        })
        .collect();
    if ranges.iter().any(|&(lo, hi)| hi - lo < 2) {
        return None;
    }
    Some((spts, ids, ranges))
}

// Strips are cut across x, so on tall input they become slivers and merge seams cross most sites.
// Mirror such input across y = x first.
fn triangulate_parallel(pts: &[Pt]) -> Option<Vec<u32>> {
    if !taller_than_wide(pts) {
        return triangulate_strips(pts);
    }
    let mirrored: Vec<Pt> = pts.par_iter().map(|&[x, y]| [y, x]).collect();
    let mut tris = triangulate_strips(&mirrored)?;
    // The mirror reverses orientation, and predicates negate exactly under it, so a swap restores CCW.
    tris.par_chunks_mut(3).for_each(|t| t.swap(1, 2));
    Some(tris)
}

// Use interdecile spans of a strided sample, so that a few outliers do not set the cut axis.
fn taller_than_wide(pts: &[Pt]) -> bool {
    let step = (pts.len() / AXIS_SAMPLE).max(1);
    let (mut xs, mut ys): (Vec<f64>, Vec<f64>) =
        pts.iter().step_by(step).map(|p| (p[0], p[1])).unzip();
    interdecile_span(&mut ys) > interdecile_span(&mut xs)
}

fn interdecile_span(v: &mut [f64]) -> f64 {
    let k = v.len() / 10;
    let hi = *v.select_nth_unstable_by(v.len() - 1 - k, f64::total_cmp).1;
    let lo = *v.select_nth_unstable_by(k, f64::total_cmp).1;
    hi - lo
}

fn triangulate_strips(pts: &[Pt]) -> Option<Vec<u32>> {
    let (spts, ids, ranges) =
        bucketed_sites(pts, strip_count(pts.len())).unwrap_or_else(|| sorted_sites(pts));
    let m: usize = ranges.iter().map(|&(lo, hi)| hi - lo).sum();
    if m < 3 {
        return Some(Vec::new());
    }
    let a = spts[ranges[0].0];
    let b = spts[ranges[ranges.len() - 1].1 - 1];
    if ranges
        .par_iter()
        .all(|&(lo, hi)| spts[lo..hi].iter().all(|&p| ccw(a, b, p) == 0.0))
    {
        return Some(Vec::new());
    }
    // Dedup can leave only one strip. Triangulate it here, because the sequential path sorts all n points again.
    // Vertex ids are positions in `spts` for memory locality. `extract_triangles` maps them back to input ids.

    let len = QUAD_SLOTS * spts.len();
    if len >= EMPTY as usize {
        return None;
    }
    let mut onext = vec![0u32; len];
    let mut data = vec![0u32; len];
    let root = build_strips(&spts, &ranges, &mut onext, &mut data, 0)?;
    for &slot in &root.free {
        data[slot as usize] = EMPTY;
    }
    for &(lo, hi) in &root.spare {
        data[lo as usize..hi as usize].fill(EMPTY);
    }
    let qe = QuadEdge::new(&spts, &mut onext, &mut data, 0, Vec::new(), Vec::new());
    Some(extract_triangles(&qe, &ids))
}

struct Built {
    ldo: u32,
    rdo: u32,
    free: Vec<u32>,
    spare: Vec<(u32, u32)>,
}

fn build_strips(
    spts: &[Pt],
    ranges: &[(usize, usize)],
    onext: &mut [u32],
    data: &mut [u32],
    base: u32,
) -> Option<Built> {
    if let [(lo, hi)] = *ranges {
        if !grid_columns(&spts[lo..hi]) {
            if let Some(built) = sweep_strip(spts, lo, hi, onext, data, base) {
                return Some(built);
            }
        }
        let spare = vec![(base, base + onext.len() as u32)];
        let mut qe = QuadEdge::new(spts, onext, data, base, Vec::new(), spare);
        let (ldo, rdo) = qe.delaunay(lo as u32, hi as u32)?;
        return Some(Built {
            ldo,
            rdo,
            free: qe.free,
            spare: qe.spare,
        });
    }
    let mid = ranges.len() / 2;
    let split = QUAD_SLOTS * (ranges[mid].0 - ranges[0].0);
    let (left, right) = {
        let (lo_on, hi_on) = onext.split_at_mut(split);
        let (lo_da, hi_da) = data.split_at_mut(split);
        let right_base = base + split as u32;
        rayon::join(
            || build_strips(spts, &ranges[..mid], lo_on, lo_da, base),
            || build_strips(spts, &ranges[mid..], hi_on, hi_da, right_base),
        )
    };
    let (mut left, mut right) = (left?, right?);
    left.free.append(&mut right.free);
    left.spare.append(&mut right.spare);
    let mut qe = QuadEdge::new(spts, onext, data, base, left.free, left.spare);
    let (ldo, rdo) = qe.merge(left.ldo, left.rdo, right.ldo, right.rdo)?;
    Some(Built {
        ldo,
        rdo,
        free: qe.free,
        spare: qe.spare,
    })
}

// Divide and conquer is faster than the sweep on long grid columns.
#[allow(clippy::float_cmp)]
fn grid_columns(strip: &[Pt]) -> bool {
    let columns = 1 + strip.windows(2).filter(|w| w[0][0] != w[1][0]).count();
    strip.len() >= GRID_COLUMN_MIN_PTS * columns
}

fn strip_count(m: usize) -> usize {
    let threads = rayon::current_num_threads();
    threads
        .max(m / STRIP_TARGET_PTS)
        .min(m / STRIP_MIN_PTS)
        .max(1)
}

fn triangulate_seq(pts: &[Pt]) -> Vec<u32> {
    let Some((mut tris, _, perm)) = sweep(pts, usize::MAX) else {
        return Vec::new();
    };
    for t in &mut tris {
        *t = perm[*t as usize];
    }
    tris
}

fn sweep_strip(
    spts: &[Pt],
    lo: usize,
    hi: usize,
    onext: &mut [u32],
    data: &mut [u32],
    base: u32,
) -> Option<Built> {
    let k = hi - lo;
    let (tri, half, perm) = sweep(&spts[lo..hi], STRIP_MAX_FLIPS_PER_PT * k)?;
    let global = |l: u32| lo as u32 + perm[l as usize];

    let nh = tri.len();
    let mut dir = vec![0u32; nh];
    let mut hull_out = vec![EMPTY; k];
    let mut nquads = 0u32;
    let mut hull_len = 0usize;
    for h in 0..nh {
        let opp = half[h];
        if opp == EMPTY {
            hull_len += 1;
            hull_out[tri[h] as usize] = h as u32;
        }
        if opp == EMPTY || (h as u32) < opp {
            dir[h] = 4 * nquads;
            if opp != EMPTY {
                dir[opp as usize] = 4 * nquads + 2;
            }
            nquads += 1;
        }
    }
    // A rounding fault can make the sweep drop a vertex. Divide and conquer then triangulates the strip.
    if nh / 3 + hull_len + 2 != 2 * k || 4 * nquads as usize > onext.len() {
        return None;
    }

    for h in 0..nh {
        let d = dir[h];
        let nxt = next_he(h);
        data[d as usize] = global(tri[h]);
        onext[d as usize] = base + QuadEdge::sym(dir[prev_he(h)]);
        if half[h] == EMPTY {
            let s = QuadEdge::sym(d);
            data[s as usize] = global(tri[nxt]);
            onext[s as usize] = base + dir[hull_out[tri[nxt] as usize] as usize];
        }
    }
    for q in 0..nquads {
        for d in [4 * q, 4 * q + 2] {
            let on = onext[d as usize] - base;
            onext[QuadEdge::rot(on) as usize] = base + QuadEdge::inv_rot(d);
        }
    }

    let first = lo as u32;
    let last = (hi - 1) as u32;
    let mut ldo = EMPTY;
    let mut rdo = EMPTY;
    for &h in &hull_out {
        if h == EMPTY {
            continue;
        }
        let h = h as usize;
        if global(tri[h]) == first {
            ldo = base + dir[h];
        }
        if global(tri[next_he(h)]) == last {
            rdo = base + QuadEdge::sym(dir[h]);
        }
    }
    if ldo == EMPTY || rdo == EMPTY {
        return None;
    }
    Some(Built {
        ldo,
        rdo,
        free: Vec::new(),
        spare: vec![(base + 4 * nquads, base + onext.len() as u32)],
    })
}

// Bucket order is monotone in distance, so sorting each bucket gives the order of one full sort.
fn sort_by_distance(pts: &[Pt], center: Pt) -> Vec<u32> {
    let cmp = |&(ka, a): &(i64, u32), &(kb, b): &(i64, u32)| {
        ka.cmp(&kb)
            .then_with(|| sort_key(pts[a as usize], a).cmp(&sort_key(pts[b as usize], b)))
    };
    let n = pts.len();
    let d2: Vec<f64> = pts.iter().map(|&p| dist2(center, p)).collect();
    let max = d2.iter().fold(0.0f64, |m, &v| m.max(v));
    let scale = (n - 1) as f64 / max;
    let mut keyed: Vec<(i64, u32)>;
    if max.is_finite() && scale.is_finite() {
        let bucket = |v: f64| ((v * scale) as usize).min(n - 1);
        let mut start = vec![0u32; n + 1];
        for &v in &d2 {
            start[bucket(v) + 1] += 1;
        }
        for b in 0..n {
            start[b + 1] += start[b];
        }
        keyed = vec![(0, 0); n];
        let mut fill = start.clone();
        for (i, &v) in d2.iter().enumerate() {
            let b = bucket(v);
            keyed[fill[b] as usize] = (total_order_key(v), i as u32);
            fill[b] += 1;
        }
        for w in start.windows(2) {
            if w[1] - w[0] > 1 {
                keyed[w[0] as usize..w[1] as usize].sort_unstable_by(cmp);
            }
        }
    } else {
        keyed = d2
            .iter()
            .enumerate()
            .map(|(i, &v)| (total_order_key(v), i as u32))
            .collect();
        keyed.sort_unstable_by(cmp);
    }
    keyed.into_iter().map(|(_, i)| i).collect()
}

// Returns the triangles and halfedges over positions in `perm`, which maps them to indices of `pts`.
fn sweep(pts: &[Pt], max_flips: usize) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let n = pts.len();
    if n < 3 {
        return None;
    }

    let (i0, i1, i2) = find_seed(pts)?;
    let center = circumcenter(pts[i0], pts[i1], pts[i2]);

    let perm = sort_by_distance(pts, center);
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
        flips: 0,
        max_flips,
    };
    if !sweep.run(center, seed) {
        return None;
    }
    Some((sweep.triangles, sweep.halfedges, perm))
}

struct QuadEdge<'a, 's> {
    pts: &'a [Pt],
    onext: &'s mut [u32],
    data: &'s mut [u32],
    base: u32,
    free: Vec<u32>,
    spare: Vec<(u32, u32)>,
}

impl<'a, 's> QuadEdge<'a, 's> {
    fn new(
        pts: &'a [Pt],
        onext: &'s mut [u32],
        data: &'s mut [u32],
        base: u32,
        free: Vec<u32>,
        spare: Vec<(u32, u32)>,
    ) -> Self {
        QuadEdge {
            pts,
            onext,
            data,
            base,
            free,
            spare,
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
        self.onext[(e - self.base) as usize]
    }

    #[inline]
    fn set_onext(&mut self, e: u32, v: u32) {
        self.onext[(e - self.base) as usize] = v;
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
        self.data[(e - self.base) as usize]
    }

    #[inline]
    fn dest(&self, e: u32) -> u32 {
        self.org(Self::sym(e))
    }

    #[inline]
    fn p(&self, v: u32) -> Pt {
        self.pts[v as usize]
    }

    #[inline]
    fn make_edge(&mut self, org: u32, dest: u32) -> Option<u32> {
        let e = if let Some(slot) = self.free.pop() {
            slot
        } else {
            let range = self.spare.last_mut()?;
            let e = range.0;
            range.0 += 4;
            if range.0 >= range.1 {
                self.spare.pop();
            }
            e
        };
        let i = (e - self.base) as usize;
        self.onext[i..i + 4].copy_from_slice(&[e, e + 3, e + 2, e + 1]);
        self.data[i..i + 4].copy_from_slice(&[org, EMPTY, dest, EMPTY]);
        Some(e)
    }

    #[inline]
    fn splice(&mut self, a: u32, b: u32) {
        let a_on = self.onext(a);
        let b_on = self.onext(b);
        let alpha = Self::rot(a_on);
        let beta = Self::rot(b_on);
        let al_on = self.onext(alpha);
        let be_on = self.onext(beta);
        self.set_onext(a, b_on);
        self.set_onext(b, a_on);
        self.set_onext(alpha, be_on);
        self.set_onext(beta, al_on);
    }

    #[inline]
    fn connect(&mut self, a: u32, b: u32) -> Option<u32> {
        let da = self.dest(a);
        let ob = self.org(b);
        let e = self.make_edge(da, ob)?;
        let la = self.lnext(a);
        self.splice(e, la);
        let se = Self::sym(e);
        self.splice(se, b);
        Some(e)
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

impl QuadEdge<'_, '_> {
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

        let mut basel = self.connect(Self::sym(rdi), ldi)?;
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
                basel = self.connect(rcand, Self::sym(basel))?;
            } else {
                basel = self.connect(Self::sym(basel), Self::sym(lcand))?;
            }
        }

        Some((ldo, rdo))
    }

    fn base2(&mut self, s0: u32, s1: u32) -> Option<(u32, u32)> {
        let a = self.make_edge(s0, s1)?;
        Some((a, Self::sym(a)))
    }

    fn base3(&mut self, s0: u32, s1: u32, s2: u32) -> Option<(u32, u32)> {
        let a = self.make_edge(s0, s1)?;
        let b = self.make_edge(s1, s2)?;
        self.splice(Self::sym(a), b);
        let turn = ccw(self.p(s0), self.p(s1), self.p(s2));
        if turn > 0.0 {
            self.connect(b, a)?;
            Some((a, Self::sym(b)))
        } else if turn < 0.0 {
            let c = self.connect(b, a)?;
            Some((Self::sym(c), c))
        } else {
            Some((a, Self::sym(b)))
        }
    }

    fn delaunay(&mut self, lo: u32, hi: u32) -> Option<(u32, u32)> {
        let n = hi - lo;
        // A 1-site range has no edge, and n == 1 gives mid == lo, which recurses with no end.
        if n < 2 {
            return None;
        }
        if n == 2 {
            return self.base2(lo, lo + 1);
        }
        if n == 3 {
            return self.base3(lo, lo + 1, lo + 2);
        }
        let mid = lo + n / 2;
        let (ldo, ldi) = self.delaunay(lo, mid)?;
        let (rdi, rdo) = self.delaunay(mid, hi)?;
        self.merge(ldo, ldi, rdi, rdo)
    }
}

// Emits each face from its lowest-numbered edge, which is the edge a serial scan meets first.
// That keeps the serial output order with no shared `seen` state, so the scan runs in parallel.
fn extract_triangles(qe: &QuadEdge<'_, '_>, sites: &[u32]) -> Vec<u32> {
    let nquads = qe.onext.len() / 4;
    let parts: Vec<Vec<u32>> = (0..nquads.div_ceil(EXTRACT_CHUNK_QUADS))
        .into_par_iter()
        .map(|c| {
            let quads = c * EXTRACT_CHUNK_QUADS..nquads.min((c + 1) * EXTRACT_CHUNK_QUADS);
            let mut out = Vec::with_capacity(2 * quads.len() + 16);
            for q in quads {
                let start = (q * 4) as u32;
                if qe.org(start) == EMPTY {
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
    use super::{hull_edges, triangulate};

    const PTS: [[f64; 2]; 5] = [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0], [1.0, 0.0]];

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
