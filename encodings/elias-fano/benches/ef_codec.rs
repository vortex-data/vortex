// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for the [`vortex_elias_fano::ef`] codec on its own.
//!
//! Low bits come from the encoder's own `Vec`, so only the codec is measured. `encode` and `decode`
//! cover the batch paths and `element_at` the per-element read.
//!
//! Three shapes, because the layout behaves differently in each:
//!
//! * `Sparse`: a wide universe, so `lower_width` is large and every read touches the low bits.
//! * `Dense`: a universe no wider than the row count, so `lower_width` is zero and the low bits are
//!   never read at all.
//! * `Duplicates`: few distinct values over a wide universe, so each occupied bucket is deep.
//!   Random data almost never produces this.

#![expect(clippy::unwrap_used)]

use std::convert::Infallible;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_elias_fano::ef;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Sparse,
    Dense,
    Duplicates,
}

const SHAPES: &[Shape] = &[Shape::Sparse, Shape::Dense, Shape::Duplicates];

/// Elements per sequence: small enough that encoding or decoding all of them stays inside the 1 ms
/// per-iteration budget.
const LEN: usize = 1 << 16;

/// The universe `Sparse` and `Duplicates` draw from.
const UNIVERSE: u64 = 1 << 30;

/// Elements per distinct value in `Duplicates`, so an occupied bucket fills a whole word of the
/// upper array.
const ELEMENTS_PER_VALUE: usize = 64;

/// Lookups per iteration, held fixed across shapes so the figures compare.
const PROBES: usize = 4096;

/// [`LEN`] sorted elements starting at zero, as the encoder sees them once the reference is
/// subtracted.
fn sequence(shape: Shape) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut values: Vec<u64> = match shape {
        Shape::Sparse => (0..LEN).map(|_| rng.random_range(0..UNIVERSE)).collect(),
        // A universe no wider than the row count drives `lower_width` to zero.
        Shape::Dense => (0..LEN).map(|_| rng.random_range(0..LEN as u64)).collect(),
        Shape::Duplicates => {
            let levels = (LEN / ELEMENTS_PER_VALUE) as u64;
            let step = UNIVERSE / levels;
            (0..LEN)
                .map(|_| rng.random_range(0..levels) * step)
                .collect()
        }
    };
    values.sort_unstable();
    let reference = values[0];
    values.iter().map(|value| value - reference).collect()
}

/// [`PROBES`] draws in a fixed random order, so nothing about locality is measured by accident.
fn probes<T>(mut draw: impl FnMut(&mut StdRng) -> T) -> Vec<T> {
    let mut rng = StdRng::seed_from_u64(1);
    (0..PROBES).map(|_| draw(&mut rng)).collect()
}

/// An encoded sequence and everything a reader borrows from it.
struct Fixture {
    upper: Vec<u8>,
    /// Both sample tables as little-endian bytes, zeros first.
    samples: Vec<u8>,
    /// Byte offset of the one-sample table within `samples`.
    seam: usize,
    lower: Vec<u64>,
    lower_width: u8,
    upper_len: usize,
    len: usize,
}

impl Fixture {
    fn new(shape: Shape) -> Self {
        let elements = sequence(shape);
        let span = elements[elements.len() - 1];
        let encoded = ef::encode(elements.iter().copied(), span).unwrap();
        let samples = encoded
            .samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let seam = usize::try_from(ef::num_samples0(span, encoded.lower_width)).unwrap()
            * size_of::<u64>();
        Self {
            upper: encoded.upper,
            samples,
            seam,
            lower: encoded.lower,
            lower_width: encoded.lower_width,
            upper_len: usize::try_from(encoded.upper_len).unwrap(),
            len: elements.len(),
        }
    }

    fn bits(&self) -> ef::Bits<'_> {
        ef::Bits::new(&self.upper, 0, self.upper_len)
    }

    fn samples1(&self) -> &[u8] {
        &self.samples[self.seam..]
    }
}

/// Low bits served straight from a slice.
struct SliceLows<'a>(&'a [u64]);

impl ef::LowBits for SliceLows<'_> {
    type Error = Infallible;

    #[expect(clippy::cast_possible_truncation)]
    fn get(&mut self, rank: u64) -> Result<u64, Infallible> {
        Ok(self.0[rank as usize])
    }
}

/// Building the upper array, both sample tables and the low bits, in one pass over the elements.
#[divan::bench(args = SHAPES)]
fn encode(bencher: Bencher, shape: Shape) {
    let elements = sequence(shape);
    let span = elements[elements.len() - 1];
    bencher
        .with_inputs(|| (&elements, span))
        .bench_values(|(elements, span)| ef::encode(elements.iter().copied(), span).unwrap());
}

/// Point lookups: one sampled `select1` for the high part and one low-bits read apiece.
#[divan::bench(args = SHAPES)]
fn element_at(bencher: Bencher, shape: Shape) {
    let fixture = Fixture::new(shape);
    let layout = ef::Layout::new(
        fixture.bits(),
        fixture.samples1(),
        fixture.lower_width,
        0,
        fixture.len,
    );
    let indices = probes(|rng| rng.random_range(0..fixture.len));
    bencher
        .with_inputs(|| (layout, SliceLows(fixture.lower.as_slice()), &indices))
        .bench_values(|(layout, mut lows, indices)| {
            for &index in indices {
                divan::black_box(ef::element_at(layout, index, &mut lows).unwrap());
            }
        });
}

/// Whole-sequence decode as a host runs it: bound the window with two selects, copy it out as
/// words, then walk it with the `Decoder`.
#[divan::bench(args = SHAPES)]
fn decode(bencher: Bencher, shape: Shape) {
    let fixture = Fixture::new(shape);
    bencher
        .with_inputs(|| (&fixture, vec![0u64; fixture.len]))
        .bench_values(|(fixture, mut out)| {
            let bits = fixture.bits();
            let samples1 = fixture.samples1();
            let start = ef::position_of_rank(bits, samples1, 0).unwrap();
            let end = ef::position_of_rank(bits, samples1, fixture.len as u64 - 1).unwrap() + 1;
            let words = ef::window_words(bits, start, end);

            let mut decoder =
                ef::Decoder::new(&words, start, 0, fixture.len, fixture.lower_width).unwrap();
            // At width zero nothing is stored, and a host passes no low bits at all.
            let lows = (fixture.lower_width > 0).then_some(fixture.lower.as_slice());
            decoder.segment(lows, |index, element| out[index] = element);
            decoder.finish().unwrap();
            out
        });
}
