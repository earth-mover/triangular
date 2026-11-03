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

const PARALLEL_MIN_N: usize = 10_000;

const STRIP_MIN_PTS: usize = 1024;
const STRIP_TARGET_PTS: usize = 4096;
const QUAD_SLOTS: usize = 12;
const BUCKETS_PER_STRIP: usize = 16;

type Sites = (Vec<Pt>, Vec<u32>, Vec<(usize, usize)>);
type Grouped = (Vec<(Pt, u32)>, Vec<usize>);
const GRID_HULL_MIN_N: usize = 100_000;
const GRID_HULL_MIN_ROWS: usize = 16;
const GRID_CLOSE_REL: f64 = 1e-10;
const GRID_CLOSE_ABS: f64 = 1e-4;

const PRED_EPSILON: f64 = f64::EPSILON * 0.5;
const CCW_ERRBOUND_A: f64 = (3.0 + 16.0 * PRED_EPSILON) * PRED_EPSILON;
const ICC_ERRBOUND_A: f64 = (10.0 + 96.0 * PRED_EPSILON) * PRED_EPSILON;

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
    in_circle_exact(a, b, c, p, det)
}

#[cold]
#[inline(never)]
fn in_circle_exact(a: Pt, b: Pt, c: Pt, p: Pt, det: f64) -> bool {
    if small_dyadic(a, b, c, p) {
        return det > 0.0;
    }
    incircle(co(a), co(b), co(c), co(p)) > 0.0
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
        s == s.trunc()
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
fn close_grid_value(left: f64, right: f64) -> bool {
    (left - right).abs()
        <= GRID_CLOSE_ABS.max(GRID_CLOSE_REL * left.abs().max(right.abs()).max(1.0))
}

fn ordered_longitude_grid_rows(pts: &[Pt]) -> Option<Vec<(usize, usize)>> {
    if pts.len() < GRID_HULL_MIN_N {
        return None;
    }

    let mut rows = Vec::new();
    let mut row_start = 0usize;
    let mut previous_y = None;
    let mut lat_direction = 0i8;
    while row_start < pts.len() {
        let row_y = pts[row_start][1];
        if !row_y.is_finite() {
            return None;
        }
        if let Some(last_y) = previous_y {
            if close_grid_value(row_y, last_y) {
                return None;
            }
            let row_direction = if row_y > last_y { 1 } else { -1 };
            if lat_direction == 0 {
                lat_direction = row_direction;
            } else if lat_direction != row_direction {
                return None;
            }
        }

        let mut row_end = row_start + 1;
        while row_end < pts.len() && close_grid_value(pts[row_end][1], row_y) {
            row_end += 1;
        }
        if row_end - row_start < 2 {
            return None;
        }

        let first_x = pts[row_start][0];
        let expected_dx = pts[row_start + 1][0] - first_x;
        if !(first_x.is_finite() && expected_dx.is_finite() && expected_dx > 0.0) {
            return None;
        }

        let mut previous_x = first_x;
        for point in &pts[row_start + 1..row_end] {
            let current_x = point[0];
            if !current_x.is_finite()
                || current_x <= previous_x
                || !close_grid_value(current_x - previous_x, expected_dx)
            {
                return None;
            }
            previous_x = current_x;
        }

        if !close_grid_value(previous_x + expected_dx, first_x + 360.0) {
            return None;
        }

        rows.push((row_start, row_end));
        previous_y = Some(row_y);
        row_start = row_end;
    }

    if rows.len() < GRID_HULL_MIN_ROWS {
        return None;
    }
    Some(rows)
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

fn cmp_site(a: &(Pt, u32), b: &(Pt, u32)) -> std::cmp::Ordering {
    a.0[0]
        .total_cmp(&b.0[0])
        .then(a.0[1].total_cmp(&b.0[1]))
        .then(a.1.cmp(&b.1))
}

fn dedup_into(sorted: &[(Pt, u32)], spts: &mut [Pt], ids: &mut [u32]) -> usize {
    let mut k = 0;
    for &(p, i) in sorted {
        if k == 0 || !same_point(spts[k - 1], p) {
            spts[k] = p;
            ids[k] = i;
            k += 1;
        }
    }
    k
}

fn sorted_sites(pts: &[Pt]) -> Sites {
    let mut order: Vec<(Pt, u32)> = pts
        .par_iter()
        .zip(0..pts.len() as u32)
        .map(|(&p, i)| (p, i))
        .collect();
    order.par_sort_unstable_by(cmp_site);
    let mut spts = vec![[0.0; 2]; order.len()];
    let mut ids = vec![0u32; order.len()];
    let m = dedup_into(&order, &mut spts, &mut ids);
    let nstrips = strip_count(m).max(1);
    let ranges = (0..nstrips)
        .map(|s| (s * m / nstrips, (s + 1) * m / nstrips))
        .filter(|&(lo, hi)| hi > lo)
        .collect();
    (spts, ids, ranges)
}

struct BucketPlan {
    xmin: f64,
    scale: f64,
    nbuckets: usize,
    chunk: usize,
    hists: Vec<Vec<u32>>,
    strip_of: Vec<u32>,
    sizes: Vec<usize>,
}

impl BucketPlan {
    #[inline]
    fn strip(&self, x: f64) -> usize {
        let b = (((x - self.xmin) * self.scale) as usize).min(self.nbuckets - 1);
        self.strip_of[b] as usize
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
    let chunk = n.div_ceil(4 * rayon::current_num_threads()).max(1 << 12);
    let hists: Vec<Vec<u32>> = pts
        .par_chunks(chunk)
        .map(|c| {
            let mut h = vec![0u32; nbuckets];
            for p in c {
                h[(((p[0] - xmin) * scale) as usize).min(nbuckets - 1)] += 1;
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
        xmin,
        scale,
        nbuckets,
        chunk,
        hists,
        strip_of,
        sizes,
    })
}

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
            let mut buf = vec![([0.0; 2], 0u32); part.len()];
            for (j, &p) in part.iter().enumerate() {
                let s = plan.strip(p[0]);
                buf[cursor[s]] = (p, (c * plan.chunk + j) as u32);
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
            let mut strip = Vec::with_capacity(sp.len());
            for (buf, starts) in &grouped {
                strip.extend_from_slice(&buf[starts[s]..starts[s + 1]]);
            }
            strip.sort_unstable_by(cmp_site);
            (off, off + dedup_into(&strip, sp, id))
        })
        .collect();
    if ranges.iter().any(|&(lo, hi)| hi - lo < 2) {
        return None;
    }
    Some((spts, ids, ranges))
}

fn triangulate_parallel(pts: &[Pt]) -> Option<Vec<u32>> {
    let nstrips = strip_count(pts.len());
    if nstrips < 2 {
        return None;
    }
    let (spts, ids, ranges) = bucketed_sites(pts, nstrips).unwrap_or_else(|| sorted_sites(pts));
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
    if ranges.len() < 2 {
        return None;
    }

    let len = QUAD_SLOTS * spts.len();
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
        if let Some(built) = sweep_strip(spts, lo, hi, onext, data, base) {
            return Some(built);
        }
        let sites: Vec<u32> = (lo as u32..hi as u32).collect();
        let spare = vec![(base, base + onext.len() as u32)];
        let mut qe = QuadEdge::new(spts, onext, data, base, Vec::new(), spare);
        let (ldo, rdo) = qe.delaunay(&sites)?;
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

fn strip_count(m: usize) -> usize {
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return 1;
    }
    threads.max(m / STRIP_TARGET_PTS).min(m / STRIP_MIN_PTS)
}

fn triangulate_seq(pts: &[Pt]) -> Vec<u32> {
    if pts.len() < 3 {
        return Vec::new();
    }
    let Some((triangles, _, perm)) = sweep(pts) else {
        return Vec::new();
    };
    triangles.iter().map(|&t| perm[t as usize] as u32).collect()
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
    let (tri, half, perm) = sweep(&spts[lo..hi])?;
    let global = |l: u32| (lo + perm[l as usize]) as u32;

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

fn sweep(pts: &[Pt]) -> Option<(Vec<u32>, Vec<u32>, Vec<usize>)> {
    let n = pts.len();
    let (i0, i1, i2) = find_seed(pts)?;
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

    #[inline]
    fn in_circle_e(&self, a: u32, b: u32, c: u32, d: u32) -> bool {
        in_circle(self.p(a), self.p(b), self.p(c), self.p(d))
    }
}

impl QuadEdge<'_, '_> {
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

    fn delaunay(&mut self, sites: &[u32]) -> Option<(u32, u32)> {
        let n = sites.len();
        if n == 2 {
            return self.base2(sites[0], sites[1]);
        }
        if n == 3 {
            return self.base3(sites[0], sites[1], sites[2]);
        }
        let half = n / 2;
        let (ldo, ldi) = self.delaunay(&sites[..half])?;
        let (rdi, rdo) = self.delaunay(&sites[half..])?;
        self.merge(ldo, ldi, rdi, rdo)
    }
}

fn extract_triangles(qe: &QuadEdge<'_, '_>, ids: &[u32]) -> Vec<u32> {
    const CHUNK: usize = 1 << 13;
    let nquads = qe.onext.len() / 4;
    let pieces: Vec<Vec<u32>> = (0..nquads.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let mut out = Vec::with_capacity(3 * CHUNK);
            for q in c * CHUNK..nquads.min((c + 1) * CHUNK) {
                let start = 4 * q as u32;
                if qe.org(start) == EMPTY {
                    continue;
                }
                for e0 in [start, QuadEdge::sym(start)] {
                    let e1 = qe.lnext(e0);
                    if e1 < e0 {
                        continue;
                    }
                    let e2 = qe.lnext(e1);
                    if e2 < e0 || qe.lnext(e2) != e0 {
                        continue;
                    }
                    let a = qe.org(e0);
                    let b = qe.org(e1);
                    let c = qe.org(e2);
                    if ccw(qe.p(a), qe.p(b), qe.p(c)) > 0.0 {
                        out.extend_from_slice(&[ids[a as usize], ids[b as usize], ids[c as usize]]);
                    }
                }
            }
            out
        })
        .collect();
    par_concat(&pieces)
}

fn par_concat(pieces: &[Vec<u32>]) -> Vec<u32> {
    let total = pieces.iter().map(Vec::len).sum();
    let mut out = vec![0u32; total];
    let mut rest = out.as_mut_slice();
    let mut slots = Vec::with_capacity(pieces.len());
    for p in pieces {
        let (head, tail) = rest.split_at_mut(p.len());
        slots.push(head);
        rest = tail;
    }
    slots
        .into_par_iter()
        .zip(pieces.par_iter())
        .for_each(|(dst, src)| dst.copy_from_slice(src));
    out
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

fn hull_edges_full(pts: &[Pt]) -> Vec<[i32; 2]> {
    let n = pts.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        pts[a][0]
            .total_cmp(&pts[b][0])
            .then(pts[a][1].total_cmp(&pts[b][1]))
            .then(a.cmp(&b))
    });

    let mut uniq: Vec<usize> = Vec::with_capacity(n);
    for &i in &order {
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

fn longitude_grid_hull_edges(pts: &[Pt]) -> Option<Vec<[i32; 2]>> {
    let rows = ordered_longitude_grid_rows(pts)?;
    let mut candidates = Vec::with_capacity(
        2 * rows.len() + rows[0].1 - rows[0].0 + rows[rows.len() - 1].1 - rows[rows.len() - 1].0,
    );
    for (row_index, &(row_start, row_end)) in rows.iter().enumerate() {
        candidates.push(row_start);
        candidates.push(row_end - 1);
        if row_index == 0 || row_index + 1 == rows.len() {
            candidates.extend(row_start..row_end);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();

    let candidate_pts: Vec<Pt> = candidates.iter().map(|&index| pts[index]).collect();
    let edges = hull_edges_full(&candidate_pts);
    Some(
        edges
            .into_iter()
            .map(|[left, right]| {
                [
                    candidates[left as usize] as i32,
                    candidates[right as usize] as i32,
                ]
            })
            .collect(),
    )
}

#[must_use]
pub fn hull_edges(pts: &[Pt]) -> Vec<[i32; 2]> {
    longitude_grid_hull_edges(pts).unwrap_or_else(|| hull_edges_full(pts))
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
