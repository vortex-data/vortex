// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for the run-end filter inner loop (`filter_run_end_primitive`).
//!
//! This measures the kernel directly rather than going through the lazy
//! `ArrayRef::filter` (which only builds a `FilterArray` node and does not run
//! the kernel). The hot work is a per-run popcount of the predicate mask, which
//! now uses `BitBuffer::count_range` (SIMD) instead of a bit-by-bit walk.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_precision_loss)]
#![expect(clippy::cast_sign_loss)]
#![expect(clippy::expect_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_runend::RunEnd;
use vortex_runend::_benchmarking::filter_run_end_primitive;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

#[derive(Clone, Copy)]
struct FilterBenchArgs {
    /// Total logical length of the decoded array.
    length: usize,
    /// Average run length used when building the run-end array.
    run_length: usize,
    /// Fraction of mask bits that are set to `true`.
    density: f64,
    /// Bit offset of a sliced predicate mask.
    mask_offset: usize,
}

impl fmt::Display for FilterBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "len={}_run={}_density={:.1}_offset={}",
            self.length, self.run_length, self.density, self.mask_offset
        )
    }
}

const FILTER_ARGS: [FilterBenchArgs; 36] = {
    let mut args = [FilterBenchArgs {
        length: 16_384,
        run_length: 1,
        density: 0.1,
        mask_offset: 0,
    }; 36];
    let lengths = [1, 4, 16, 64, 256, 1024];
    let densities = [0.1, 0.5, 0.9];
    let mut i = 0;
    while i < args.len() {
        args[i] = FilterBenchArgs {
            length: 16_384,
            run_length: lengths[i / 6],
            density: densities[(i / 2) % 3],
            mask_offset: (i % 2) * 3,
        };
        i += 1;
    }
    args
};

/// Build the run-end boundaries (cumulative run lengths) for `length` rows.
fn build_run_ends(length: usize, run_length: usize) -> Vec<u32> {
    let n_runs = length.div_ceil(run_length);
    (0..n_runs)
        .map(|r| (((r + 1) * run_length).min(length)) as u32)
        .collect()
}

/// Build a predicate mask of `length` bits with approximately `density` set bits,
/// shuffled so the set bits are spread across runs.
fn build_mask(length: usize, density: f64) -> BitBuffer {
    let n_true = (length as f64 * density).round() as usize;
    let mut bits = vec![false; length];
    for b in bits.iter_mut().take(n_true) {
        *b = true;
    }
    let mut rng = StdRng::seed_from_u64(0x5eed);
    bits.shuffle(&mut rng);
    BitBuffer::from(bits)
}

#[divan::bench(args = FILTER_ARGS)]
fn filter_run_end(bencher: Bencher, args: FilterBenchArgs) {
    let run_ends = build_run_ends(args.length, args.run_length);
    let mask = build_mask(args.length + args.mask_offset, args.density)
        .slice(args.mask_offset..args.length + args.mask_offset);
    let length = args.length as u64;
    bencher
        .with_inputs(|| (run_ends.clone(), mask.clone()))
        .bench_refs(|(run_ends, mask)| {
            filter_run_end_primitive::<u32>(run_ends, 0, length, mask).expect("filter")
        });
}

#[divan::bench(args = FILTER_ARGS)]
fn filter_run_end_array(bencher: Bencher, args: FilterBenchArgs) {
    let ends = build_run_ends(args.length, args.run_length);
    let values = PrimitiveArray::from_iter(0..ends.len() as i64).into_array();
    let array = RunEnd::new(
        PrimitiveArray::from_iter(ends).into_array(),
        values,
        &mut SESSION.create_execution_ctx(),
    )
    .into_array();
    let mask = Mask::from(
        build_mask(args.length + args.mask_offset, args.density)
            .slice(args.mask_offset..args.length + args.mask_offset),
    );
    bencher
        .with_inputs(|| (array.clone(), mask.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mask, mut ctx)| {
            array
                .filter(mask)
                .expect("filter")
                .execute::<PrimitiveArray>(&mut ctx)
                .expect("execute")
        });
}
