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

#[must_use]
pub fn triangulate(pts: &[Pt]) -> Vec<u32> {
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
