// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::all_identical::all_identical;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const NUM_LISTS: usize = 4_096;
const LIST_SIZE: usize = 16;
const FSL_ROWS: usize = 16_384;
const FSL_LIST_SIZE: u32 = 8;
const DECIMAL_LEN: usize = 65_536;

fn scalar_loop_identical(lhs: &ArrayRef, rhs: &ArrayRef) -> bool {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    (0..lhs.len()).all(|idx| {
        lhs.execute_scalar(idx, &mut ctx).unwrap() == rhs.execute_scalar(idx, &mut ctx).unwrap()
    })
}

fn all_identical_fast(lhs: &ArrayRef, rhs: &ArrayRef) -> bool {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    all_identical(lhs, rhs, &mut ctx).unwrap()
}

fn make_zero_copy_list_pair() -> (ArrayRef, ArrayRef) {
    let len = NUM_LISTS * LIST_SIZE;
    let elements = PrimitiveArray::from_iter((0..len).map(|i| i as i32)).into_array();

    let offsets_u16: Buffer<u16> = (0..NUM_LISTS).map(|i| (i * LIST_SIZE) as u16).collect();
    let offsets_i32: Buffer<i32> = (0..NUM_LISTS).map(|i| (i * LIST_SIZE) as i32).collect();
    let sizes_u8: Buffer<u8> = std::iter::repeat_n(LIST_SIZE as u8, NUM_LISTS).collect();
    let sizes_i16: Buffer<i16> = std::iter::repeat_n(LIST_SIZE as i16, NUM_LISTS).collect();

    let lhs_list = ListViewArray::try_new(
        elements.clone(),
        offsets_u16.into_array(),
        sizes_u8.into_array(),
        Validity::NonNullable,
    )
    .unwrap();
    let rhs_list = ListViewArray::try_new(
        elements,
        offsets_i32.into_array(),
        sizes_i16.into_array(),
        Validity::NonNullable,
    )
    .unwrap();
    (lhs_list.into_array(), rhs_list.into_array())
}

fn make_nullable_fsl_pair() -> (ArrayRef, ArrayRef) {
    let values_a =
        PrimitiveArray::from_iter((0..(FSL_ROWS * FSL_LIST_SIZE as usize)).map(|i| i as i32))
            .into_array();

    let values_b = PrimitiveArray::from_iter((0..(FSL_ROWS * FSL_LIST_SIZE as usize)).map(|i| {
        if i / FSL_LIST_SIZE as usize % 2 == 0 {
            i as i32
        } else {
            -(i as i32)
        }
    }))
    .into_array();

    let validity = Validity::from_iter((0..FSL_ROWS).map(|row| row % 2 == 0));

    (
        FixedSizeListArray::try_new(values_a, FSL_LIST_SIZE, validity.clone(), FSL_ROWS)
            .unwrap()
            .into_array(),
        FixedSizeListArray::try_new(values_b, FSL_LIST_SIZE, validity, FSL_ROWS)
            .unwrap()
            .into_array(),
    )
}

fn make_decimal_pair() -> (ArrayRef, ArrayRef) {
    let decimal_dtype = DecimalDType::new(4, 0);
    let lhs = DecimalArray::new(
        Buffer::from_iter((0..DECIMAL_LEN).map(|i| (i % 10_000) as i16)),
        decimal_dtype,
        Validity::NonNullable,
    )
    .into_array();
    let rhs = DecimalArray::new(
        Buffer::from_iter((0..DECIMAL_LEN).map(|i| (i % 10_000) as i32)),
        decimal_dtype,
        Validity::NonNullable,
    )
    .into_array();
    (lhs, rhs)
}

#[divan::bench]
fn list_zero_copy_all_identical(bencher: Bencher) {
    let pair = make_zero_copy_list_pair();
    bencher
        .with_inputs(|| (&pair.0, &pair.1))
        .bench_refs(|(lhs, rhs)| {
            divan::black_box(all_identical_fast(lhs, rhs));
        });
}

#[divan::bench]
fn list_zero_copy_scalar_loop(bencher: Bencher) {
    let pair = make_zero_copy_list_pair();
    bencher
        .with_inputs(|| (&pair.0, &pair.1))
        .bench_refs(|(lhs, rhs)| {
            divan::black_box(scalar_loop_identical(lhs, rhs));
        });
}

#[divan::bench]
fn fsl_nullable_all_identical(bencher: Bencher) {
    let pair = make_nullable_fsl_pair();
    bencher
        .with_inputs(|| (&pair.0, &pair.1))
        .bench_refs(|(lhs, rhs)| {
            divan::black_box(all_identical_fast(lhs, rhs));
        });
}

#[divan::bench]
fn fsl_nullable_scalar_loop(bencher: Bencher) {
    let pair = make_nullable_fsl_pair();
    bencher
        .with_inputs(|| (&pair.0, &pair.1))
        .bench_refs(|(lhs, rhs)| {
            divan::black_box(scalar_loop_identical(lhs, rhs));
        });
}

#[divan::bench]
fn decimal_widen_all_identical(bencher: Bencher) {
    let pair = make_decimal_pair();
    bencher
        .with_inputs(|| (&pair.0, &pair.1))
        .bench_refs(|(lhs, rhs)| {
            divan::black_box(all_identical_fast(lhs, rhs));
        });
}

#[divan::bench]
fn decimal_widen_scalar_loop(bencher: Bencher) {
    let pair = make_decimal_pair();
    bencher
        .with_inputs(|| (&pair.0, &pair.1))
        .bench_refs(|(lhs, rhs)| {
            divan::black_box(scalar_loop_identical(lhs, rhs));
        });
}
