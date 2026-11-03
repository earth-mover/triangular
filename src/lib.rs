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

const EMPTY: u32 = u32::MAX;

const PARALLEL_MIN_N: usize = 50_000;

const STRIP_MIN_PTS: usize = 1024;
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
    incircle(co(a), co(b), co(c), co(p)) > 0.0
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

    let i0 = (0..n)
        .min_by(|&a, &b| dist2(bbox_center, pts[a]).total_cmp(&dist2(bbox_center, pts[b])))
        .unwrap();
    let p0 = pts[i0];

    let i1 = (0..n)
        .filter(|&i| i != i0 && !same_point(p0, pts[i]))
        .min_by(|&a, &b| dist2(p0, pts[a]).total_cmp(&dist2(p0, pts[b])))?;
    let p1 = pts[i1];

    let i2 = (0..n)
        .filter(|&i| i != i0 && i != i1 && !same_point(p0, pts[i]) && !same_point(p1, pts[i]))
        .min_by(|&a, &b| circumradius2(p0, p1, pts[a]).total_cmp(&circumradius2(p0, p1, pts[b])))?;
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

        self.hash_set(p, i);
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

fn dedup_sorted_sites(pts: &[Pt]) -> Vec<u32> {
    let n = pts.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by(|&a, &b| {
        pts[a as usize][0]
            .total_cmp(&pts[b as usize][0])
            .then(pts[a as usize][1].total_cmp(&pts[b as usize][1]))
            .then(a.cmp(&b))
    });
    let mut sites: Vec<u32> = Vec::with_capacity(n);
    for &i in &order {
        if sites
            .last()
            .is_none_or(|&last| !same_point(pts[last as usize], pts[i as usize]))
        {
            sites.push(i);
        }
    }
    sites
}

fn triangulate_parallel(pts: &[Pt]) -> Option<Vec<u32>> {
    let sites = dedup_sorted_sites(pts);
    let m = sites.len();
    if m < 3 {
        return Some(Vec::new());
    }
    let a = pts[sites[0] as usize];
    let b = pts[sites[m - 1] as usize];
    if sites.iter().all(|&i| ccw(a, b, pts[i as usize]) == 0.0) {
        return Some(Vec::new());
    }

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
            let strip = &sites[lo..hi];
            let mut qe = QuadEdge::with_capacity(pts, 3 * strip.len());
            let (ldo, rdo) = qe.delaunay(strip)?;
            Some((qe.onext, qe.data, ldo, rdo))
        })
        .collect::<Option<Vec<StripResult>>>()?;

    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let merge_headroom = (total / 4).max(1024);
    let mut handles: Vec<(u32, u32)> = Vec::with_capacity(parts.len());
    let mut combined = QuadEdge {
        pts,
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

    let mut out = Vec::with_capacity(6 * m);
    extract_triangles(&combined, &mut out);
    Some(out)
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

    let d2: Vec<f64> = pts.iter().map(|&p| dist2(center, p)).collect();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_unstable_by(|&a, &b| {
        d2[a].total_cmp(&d2[b]).then_with(|| {
            pts[a][0]
                .total_cmp(&pts[b][0])
                .then_with(|| pts[a][1].total_cmp(&pts[b][1]))
                .then(a.cmp(&b))
        })
    });

    let mut inv = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        inv[old_i] = new_i;
    }
    let seed = (inv[i0], inv[i1], inv[i2]);

    let permuted: Vec<Pt> = perm.iter().map(|&i| pts[i]).collect();

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
    sweep
        .triangles
        .iter()
        .map(|&t| perm[t as usize] as u32)
        .collect()
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

    #[inline]
    fn in_circle_e(&self, a: u32, b: u32, c: u32, d: u32) -> bool {
        in_circle(self.p(a), self.p(b), self.p(c), self.p(d))
    }
}

impl QuadEdge<'_> {
    #[inline]
    fn valid(&self, e: u32, basel: u32) -> bool {
        self.right_of(self.dest(e), basel)
    }

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
            let mut lcand = self.onext(Self::sym(basel));
            if self.valid(lcand, basel) {
                let mut g = 0usize;
                while self.in_circle_e(
                    self.dest(basel),
                    self.org(basel),
                    self.dest(lcand),
                    self.dest(self.onext(lcand)),
                ) {
                    let t = self.onext(lcand);
                    self.delete_edge(lcand);
                    lcand = t;
                    g += 1;
                    if g > cap {
                        return None;
                    }
                }
            }

            let mut rcand = self.oprev(basel);
            if self.valid(rcand, basel) {
                let mut g = 0usize;
                while self.in_circle_e(
                    self.dest(basel),
                    self.org(basel),
                    self.dest(rcand),
                    self.dest(self.oprev(rcand)),
                ) {
                    let t = self.oprev(rcand);
                    self.delete_edge(rcand);
                    rcand = t;
                    g += 1;
                    if g > cap {
                        return None;
                    }
                }
            }

            let l_valid = self.valid(lcand, basel);
            let r_valid = self.valid(rcand, basel);
            if !l_valid && !r_valid {
                break;
            }

            if !l_valid
                || (r_valid
                    && self.in_circle_e(
                        self.dest(lcand),
                        self.org(lcand),
                        self.org(rcand),
                        self.dest(rcand),
                    ))
            {
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

    fn delaunay(&mut self, sites: &[u32]) -> Option<(u32, u32)> {
        let n = sites.len();
        if n == 2 {
            return Some(self.base2(sites[0], sites[1]));
        }
        if n == 3 {
            return Some(self.base3(sites[0], sites[1], sites[2]));
        }
        let half = n / 2;
        let (ldo, ldi) = self.delaunay(&sites[..half])?;
        let (rdi, rdo) = self.delaunay(&sites[half..])?;
        self.merge(ldo, ldi, rdi, rdo)
    }
}

fn extract_triangles(qe: &QuadEdge<'_>, out: &mut Vec<u32>) {
    let mut seen = vec![false; qe.onext.len()];
    for &slot in &qe.free {
        seen[slot as usize] = true;
        seen[QuadEdge::sym(slot) as usize] = true;
    }
    for start in (0..qe.onext.len() as u32).step_by(4) {
        if qe.org(start) == EMPTY {
            continue;
        }
        for e0 in [start, QuadEdge::sym(start)] {
            if seen[e0 as usize] {
                continue;
            }
            let e1 = qe.lnext(e0);
            let e2 = qe.lnext(e1);
            if qe.lnext(e2) != e0 {
                continue;
            }
            let a = qe.org(e0);
            let b = qe.org(e1);
            let c = qe.org(e2);
            if ccw(qe.p(a), qe.p(b), qe.p(c)) <= 0.0 {
                continue;
            }
            seen[e0 as usize] = true;
            seen[e1 as usize] = true;
            seen[e2 as usize] = true;
            out.push(a);
            out.push(b);
            out.push(c);
        }
    }
}

#[allow(clippy::float_cmp)]
fn same_point(p: Pt, q: Pt) -> bool {
    p[0] == q[0] && p[1] == q[1]
}

#[allow(clippy::float_cmp)]
fn is_collinear(pts: &[Pt], uniq: &[usize]) -> bool {
    let a = pts[uniq[0]];
    let b = pts[uniq[uniq.len() - 1]];
    uniq.iter().all(|&i| ccw(a, b, pts[i]) == 0.0)
}

fn half_chain(pts: &[Pt], seq: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut chain: Vec<usize> = Vec::new();
    for i in seq {
        while chain.len() >= 2
            && ccw(
                pts[chain[chain.len() - 2]],
                pts[chain[chain.len() - 1]],
                pts[i],
            ) < 0.0
        {
            chain.pop();
        }
        chain.push(i);
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

#[inline]
fn total_order_key(v: f64) -> i64 {
    // Map -0.0 to 0.0: the chain needs equal coordinates to sort as equal.
    let bits = (v + 0.0).to_bits() as i64;
    bits ^ ((((bits >> 63) as u64) >> 1) as i64)
}

#[must_use]
pub fn hull_edges(pts: &[Pt]) -> Vec<[i32; 2]> {
    let mut keyed: Vec<(i64, i64, usize)> = hull_candidates(pts)
        .into_iter()
        .map(|i| (total_order_key(pts[i][0]), total_order_key(pts[i][1]), i))
        .collect();
    keyed.sort_unstable();

    let mut uniq: Vec<usize> = Vec::with_capacity(keyed.len());
    for &(_, _, i) in &keyed {
        if uniq
            .last()
            .is_none_or(|&last| !same_point(pts[last], pts[i]))
        {
            uniq.push(i);
        }
    }

    if uniq.len() < 3 || is_collinear(pts, &uniq) {
        return Vec::new();
    }

    let mut hull = half_chain(pts, uniq.iter().copied());
    hull.pop();
    let mut upper = half_chain(pts, uniq.iter().rev().copied());
    upper.pop();
    hull.append(&mut upper);

    let k = hull.len();
    (0..k)
        .map(|i| [hull[i] as i32, hull[(i + 1) % k] as i32])
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
