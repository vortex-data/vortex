// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare an already-packed `BitPackedArray` against a constant value.
//!
//! Three families:
//!
//! * `*_out_of_range` — constant outside `[0, 2^bit_width - 1]`, hits the
//!   `ConstantArray<bool>` short-circuit on all six operators.
//! * `*_in_range_swar_w8` — constant inside the packable range at `bit_width = 8` on
//!   `u32` storage, hits the Knuth-broadword SWAR fast path.
//! * `baseline_*` — explicit "execute to PrimitiveArray, then Arrow compare" path that
//!   would run if neither fast path existed.
//!
//! Sized to finish quickly. Run with `cargo bench -p vortex-fastlanes --bench bitpack_compare`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedData;

fn main() {
    divan::main();
}

const LENS: &[usize] = &[1024, 64 * 1024];
const BIT_WIDTHS_OOR: &[u8] = &[4, 16];

/// Build a packed array of varied in-range values, plus a same-typed constant RHS.
fn build_inputs<const BW: u8>(len: usize, constant: u32) -> (ArrayRef, ArrayRef, ExecutionCtx) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let buf: BufferMut<u32> = (0..len).map(|i| (i as u32) % (1 << BW)).collect();
    let array = BitPackedData::encode(
        &PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array(),
        BW,
        &mut ctx,
    )
    .unwrap()
    .into_array();
    let rhs = ConstantArray::new(constant, len).into_array();
    (array, rhs, ctx)
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS_OOR)]
fn fast_eq_out_of_range<const BW: u8>(bencher: Bencher, len: usize) {
    // 1 << BW is just past the packable range, so the out-of-range fast path fires.
    let (array, rhs, mut ctx) = build_inputs::<BW>(len, 1u32 << BW);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS_OOR)]
fn baseline_eq_out_of_range<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len, 1u32 << BW);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS_OOR)]
fn fast_lt_out_of_range<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len, 1u32 << BW);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS_OOR)]
fn baseline_lt_out_of_range<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len, 1u32 << BW);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

// In-range SWAR fast paths, one bench per (op, W).

const IN_RANGE_LENS: &[usize] = &[1024, 64 * 1024];

macro_rules! in_range_benches {
    ($w:expr, $constant:expr, $fast_eq:ident, $baseline_eq:ident, $fast_lt:ident, $baseline_lt:ident) => {
        #[divan::bench(args = IN_RANGE_LENS)]
        fn $fast_eq(bencher: Bencher, len: usize) {
            let (array, rhs, mut ctx) = build_inputs::<$w>(len, $constant);
            bencher.counter(ItemsCount::new(len)).bench_local(|| {
                array
                    .clone()
                    .binary(rhs.clone(), Operator::Eq)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap()
            });
        }

        #[divan::bench(args = IN_RANGE_LENS)]
        fn $baseline_eq(bencher: Bencher, len: usize) {
            let (array, rhs, mut ctx) = build_inputs::<$w>(len, $constant);
            bencher.counter(ItemsCount::new(len)).bench_local(|| {
                let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
                primitive
                    .into_array()
                    .binary(rhs.clone(), Operator::Eq)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap()
            });
        }

        #[divan::bench(args = IN_RANGE_LENS)]
        fn $fast_lt(bencher: Bencher, len: usize) {
            let (array, rhs, mut ctx) = build_inputs::<$w>(len, $constant);
            bencher.counter(ItemsCount::new(len)).bench_local(|| {
                array
                    .clone()
                    .binary(rhs.clone(), Operator::Lt)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap()
            });
        }

        #[divan::bench(args = IN_RANGE_LENS)]
        fn $baseline_lt(bencher: Bencher, len: usize) {
            let (array, rhs, mut ctx) = build_inputs::<$w>(len, $constant);
            bencher.counter(ItemsCount::new(len)).bench_local(|| {
                let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
                primitive
                    .into_array()
                    .binary(rhs.clone(), Operator::Lt)
                    .unwrap()
                    .execute::<BoolArray>(&mut ctx)
                    .unwrap()
            });
        }
    };
}

// W = 1 (boolean-style): constant 1.
in_range_benches!(
    1,
    1,
    fast_eq_in_range_swar_w1,
    baseline_eq_in_range_w1,
    fast_lt_in_range_swar_w1,
    baseline_lt_in_range_w1
);

// W = 2: constant 2.
in_range_benches!(
    2,
    2,
    fast_eq_in_range_swar_w2,
    baseline_eq_in_range_w2,
    fast_lt_in_range_swar_w2,
    baseline_lt_in_range_w2
);

// W = 4: constant 7.
in_range_benches!(
    4,
    7,
    fast_eq_in_range_swar_w4,
    baseline_eq_in_range_w4,
    fast_lt_in_range_swar_w4,
    baseline_lt_in_range_w4
);

// W = 8: constant 127.
in_range_benches!(
    8,
    127,
    fast_eq_in_range_swar_w8,
    baseline_eq_in_range_w8,
    fast_lt_in_range_swar_w8,
    baseline_lt_in_range_w8
);

// W = 16: constant 32_000.
in_range_benches!(
    16,
    32_000,
    fast_eq_in_range_swar_w16,
    baseline_eq_in_range_w16,
    fast_lt_in_range_swar_w16,
    baseline_lt_in_range_w16
);

// Non-power-of-two widths — generic Knuth-rotation-table kernel.

// W = 3: constant 3 (max for 3 bits is 7).
in_range_benches!(
    3,
    3,
    fast_eq_in_range_swar_w3,
    baseline_eq_in_range_w3,
    fast_lt_in_range_swar_w3,
    baseline_lt_in_range_w3
);

// W = 5: constant 13.
in_range_benches!(
    5,
    13,
    fast_eq_in_range_swar_w5,
    baseline_eq_in_range_w5,
    fast_lt_in_range_swar_w5,
    baseline_lt_in_range_w5
);

// W = 7: constant 100.
in_range_benches!(
    7,
    100,
    fast_eq_in_range_swar_w7,
    baseline_eq_in_range_w7,
    fast_lt_in_range_swar_w7,
    baseline_lt_in_range_w7
);

// W = 11: constant 1000.
in_range_benches!(
    11,
    1000,
    fast_eq_in_range_swar_w11,
    baseline_eq_in_range_w11,
    fast_lt_in_range_swar_w11,
    baseline_lt_in_range_w11
);
