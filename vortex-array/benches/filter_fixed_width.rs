// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compact fixed-width filter benchmarks.
//!
//! The cases cover the dispatch dimensions without a large Cartesian product so CodSpeed's
//! instruction-count simulation remains inexpensive.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unwrap_used
)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

// Keep each case small: the sweep has 24 cases and the targeted sections add only seven more.
// Sized to keep CodSpeed simulation under 1ms per benchmark.
const LEN: usize = 4_096;
const DENSITIES: &[f64] = &[0.01, 0.5, 0.8, 0.95];
const CACHED_DENSITIES: &[f64] = &[0.01, 0.1];

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Random,
    Runs,
    Contiguous,
}

const PATTERNS: &[Pattern] = &[Pattern::Random, Pattern::Runs, Pattern::Contiguous];

fn random_mask(density: f64) -> Mask {
    let threshold = (density * u64::MAX as f64) as u64;
    let mut state = 0x1234_5678_9abc_def0u64;
    Mask::from_buffer(BitBuffer::from_iter((0..LEN).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state <= threshold
    })))
}

fn pattern_mask(pattern: Pattern) -> Mask {
    match pattern {
        Pattern::Random => random_mask(0.5),
        Pattern::Runs => Mask::from_iter((0..LEN).map(|index| (index / 32).is_multiple_of(2))),
        Pattern::Contiguous => Mask::from_slices(LEN, vec![(LEN / 4, LEN * 3 / 4)]),
    }
}

fn bench_filter(
    bencher: Bencher,
    array: ArrayRef,
    make_mask: impl Fn() -> Mask + Sync,
    cache_indices: bool,
) {
    bencher
        .with_inputs(|| {
            let mask = make_mask();
            if cache_indices {
                let _ = mask.values().unwrap().indices();
            }
            (array.clone(), mask, SESSION.create_execution_ctx())
        })
        .bench_refs(|(array, mask, ctx)| {
            divan::black_box(
                array
                    .clone()
                    .filter(mask.clone())
                    .unwrap()
                    .execute::<Canonical>(ctx)
                    .unwrap(),
            );
        });
}

fn i8_array() -> ArrayRef {
    PrimitiveArray::from_iter((0..LEN).map(|index| index as i8)).into_array()
}

fn i16_array() -> ArrayRef {
    PrimitiveArray::from_iter((0..LEN).map(|index| index as i16)).into_array()
}

fn i32_array() -> ArrayRef {
    PrimitiveArray::from_iter((0..LEN).map(|index| index as i32)).into_array()
}

fn i64_array() -> ArrayRef {
    PrimitiveArray::from_iter((0..LEN).map(|index| index as i64)).into_array()
}

fn i128_array() -> ArrayRef {
    DecimalArray::from_iter(
        (0..LEN).map(|index| index as i128),
        DecimalDType::new(19, 0),
    )
    .into_array()
}

fn i256_array() -> ArrayRef {
    DecimalArray::from_iter(
        (0..LEN).map(|index| i256::from_i128(index as i128)),
        DecimalDType::new(39, 0),
    )
    .into_array()
}

macro_rules! random_density_benchmark {
    ($name:ident, $array:ident) => {
        #[divan::bench(args = DENSITIES)]
        fn $name(bencher: Bencher, density: f64) {
            bench_filter(bencher, $array(), || random_mask(density), false);
        }
    };
}

random_density_benchmark!(random_i8, i8_array);
random_density_benchmark!(random_i16, i16_array);
random_density_benchmark!(random_i32, i32_array);
random_density_benchmark!(random_i64, i64_array);
random_density_benchmark!(random_i128, i128_array);
random_density_benchmark!(random_i256, i256_array);

#[divan::bench(args = PATTERNS)]
fn patterns_i128(bencher: Bencher, pattern: Pattern) {
    bench_filter(bencher, i128_array(), || pattern_mask(pattern), false);
}

#[divan::bench(args = CACHED_DENSITIES)]
fn cached_indices_i32(bencher: Bencher, density: f64) {
    bench_filter(bencher, i32_array(), || random_mask(density), true);
}

#[divan::bench(args = CACHED_DENSITIES)]
fn cached_indices_i128(bencher: Bencher, density: f64) {
    bench_filter(bencher, i128_array(), || random_mask(density), true);
}
