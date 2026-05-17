// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OVC over RunEnd: row-by-row baseline vs encoding-aware kernel.
//!
//! Varies `avg_run_length` to show the O(num_runs) curve of the
//! encoding-aware kernel against the row-by-row scalar_at baseline.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]
#![allow(deprecated)]

use std::fmt;
use std::hint::black_box;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArray;
use vortex_runend::prototype_ovc::RUNEND_OVC_KERNELS;
use vortex_runend::prototype_ovc::ovc_runend_materialise;

fn main() {
    divan::main();
}

const N: usize = 100_000;

#[derive(Clone, Copy)]
struct Args {
    avg_run_length: usize,
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rl{:04}", self.avg_run_length)
    }
}

const ARGS: &[Args] = &[
    Args { avg_run_length: 10 },
    Args { avg_run_length: 100 },
    Args { avg_run_length: 1000 },
];

/// Build a RunEnd of `N` rows; each run carries the next u64 so OVC
/// offsets are meaningful at every boundary.
fn make_runend(avg_run_length: usize) -> (RunEndArray, ArrayRef) {
    let num_runs = N.div_ceil(avg_run_length);
    let mut ends = BufferMut::<u32>::with_capacity(num_runs);
    let mut values = Vec::<u64>::with_capacity(num_runs);

    let mut pos = 0usize;
    let mut v: u64 = 0;
    while pos < N {
        let run_len = avg_run_length.min(N - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(v);
        v += 1;
    }

    let ends_arr = PrimitiveArray::new(ends.freeze(), Validity::NonNullable).into_array();
    let values_arr =
        PrimitiveArray::new(Buffer::<u64>::copy_from(&values), Validity::NonNullable)
            .into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let re = RunEnd::try_new(ends_arr, values_arr, &mut ctx).expect("runend");
    let erased = re.clone().into_array();
    (re, erased)
}

#[divan::bench(args = ARGS, sample_count = 30)]
fn runend_naive_scalar_at(bencher: Bencher, args: Args) {
    let (_, erased) = make_runend(args.avg_run_length);
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut last = 0u64;
        for i in 0..erased.len() {
            last = u64::try_from(&erased.scalar_at(i).expect("scalar_at")).expect("u64");
        }
        black_box(last)
    });
}

/// Flat materialisation: walks runs but writes N rows.
#[divan::bench(args = ARGS, sample_count = 30)]
fn runend_materialise(bencher: Bencher, args: Args) {
    let (typed, _) = make_runend(args.avg_run_length);
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| black_box(ovc_runend_materialise(typed.as_view(), 0)));
}

/// Encoding-aware: return the input (O(1) Arc clone) after a O(num_runs) walk.
#[divan::bench(args = ARGS, sample_count = 30)]
fn runend_via_pks(bencher: Bencher, args: Args) {
    let (typed, erased) = make_runend(args.avg_run_length);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        black_box(
            RUNEND_OVC_KERNELS
                .execute(typed.as_view(), black_box(&erased), 0, &mut ctx)
                .expect("execute"),
        )
    });
}
