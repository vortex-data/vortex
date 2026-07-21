// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the envelope row loop's output-writing shape.
//!
//! `push` is the original loop: `BufferMut::with_capacity` plus four `push` calls per row —
//! each an out-of-line call, since `push`'s grow path defeats inlining. `indexed` is the
//! shipped loop: `BufferMut::zeroed` up front, plain indexed stores, and adjacent offset
//! pairs instead of a bounds-checked `row_offsets[r + 1]`. The reduction is identical in
//! both, so any difference is the write path.

use divan::Bencher;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

fn main() {
    divan::main();
}

/// `(rows, vertices per row)`: per-row write overhead matters most when rows are short.
const SHAPES: &[(usize, usize)] = &[(65536, 1), (65536, 4), (65536, 16), (16384, 64)];

fn setup(rows: usize, verts: usize) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let n = rows * verts;
    let val = |i: usize, salt: usize| ((i * 2654435761 + salt) % 1_000_003) as f64 * 0.001;
    (
        (0..=rows).map(|r| r * verts).collect(),
        (0..n).map(|i| val(i, 17)).collect(),
        (0..n).map(|i| val(i, 89)).collect(),
    )
}

fn box_corners(xs: &[f64], ys: &[f64]) -> [f64; 4] {
    let (mut xmin, mut ymin) = (f64::INFINITY, f64::INFINITY);
    let (mut xmax, mut ymax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (&x, &y) in xs.iter().zip(ys) {
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }
    [xmin, ymin, xmax, ymax]
}

#[divan::bench(args = SHAPES)]
fn push(bencher: Bencher, (rows, verts): (usize, usize)) {
    let (row_offsets, xs, ys) = setup(rows, verts);
    bencher.bench(|| -> [Buffer<f64>; 4] {
        let len = rows;
        let mut xmins = BufferMut::with_capacity(len);
        let mut ymins = BufferMut::with_capacity(len);
        let mut xmaxs = BufferMut::with_capacity(len);
        let mut ymaxs = BufferMut::with_capacity(len);
        for r in 0..len {
            let (start, end) = (row_offsets[r], row_offsets[r + 1]);
            let [xmin, ymin, xmax, ymax] = box_corners(&xs[start..end], &ys[start..end]);
            xmins.push(xmin);
            ymins.push(ymin);
            xmaxs.push(xmax);
            ymaxs.push(ymax);
        }
        [
            xmins.freeze(),
            ymins.freeze(),
            xmaxs.freeze(),
            ymaxs.freeze(),
        ]
    });
}

#[divan::bench(args = SHAPES)]
fn indexed(bencher: Bencher, (rows, verts): (usize, usize)) {
    let (row_offsets, xs, ys) = setup(rows, verts);
    bencher.bench(|| -> [Buffer<f64>; 4] {
        let len = rows;
        let mut xmins = BufferMut::zeroed(len);
        let mut ymins = BufferMut::zeroed(len);
        let mut xmaxs = BufferMut::zeroed(len);
        let mut ymaxs = BufferMut::zeroed(len);
        for (r, (&start, &end)) in row_offsets.iter().zip(&row_offsets[1..]).enumerate() {
            let [xmin, ymin, xmax, ymax] = box_corners(&xs[start..end], &ys[start..end]);
            xmins[r] = xmin;
            ymins[r] = ymin;
            xmaxs[r] = xmax;
            ymaxs[r] = ymax;
        }
        [
            xmins.freeze(),
            ymins.freeze(),
            xmaxs.freeze(),
            ymaxs.freeze(),
        ]
    });
}
