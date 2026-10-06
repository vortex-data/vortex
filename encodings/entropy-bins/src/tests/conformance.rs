// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared compute conformance suites, run on EntropyBins arrays with the encoding's kernels
//! and rules registered.

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
use vortex_array::compute::conformance::cast::test_cast_conformance;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::mask::test_mask_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_error::VortexResult;

use crate::BLOCK_VALUES;
use crate::CHUNK_VALUES;
use crate::EntropyBins;
use crate::EntropyBinsOptions;
use crate::tests::skewed;

fn encode(values: PrimitiveArray, lag: usize, block_values: usize) -> VortexResult<ArrayRef> {
    Ok(EntropyBins::from_primitive(
        values.as_view(),
        8,
        EntropyBinsOptions::new(lag, block_values),
    )?
    .into_array())
}

fn i64s(lag: usize, block_values: usize) -> VortexResult<ArrayRef> {
    encode(
        PrimitiveArray::from_iter(skewed(5000, 1)),
        lag,
        block_values,
    )
}

fn narrow() -> VortexResult<ArrayRef> {
    encode(
        PrimitiveArray::from_iter(skewed(3000, 2).into_iter().map(|v| v as i16)),
        0,
        BLOCK_VALUES,
    )
}

fn unsigned() -> VortexResult<ArrayRef> {
    encode(
        PrimitiveArray::from_iter((0..3000u32).map(|i| i % 97 + (i / 1000) * 1000)),
        1,
        BLOCK_VALUES,
    )
}

fn bytes() -> VortexResult<ArrayRef> {
    encode(
        PrimitiveArray::from_iter((0..2500u32).map(|i| (i * 7 % 13) as u8)),
        0,
        BLOCK_VALUES,
    )
}

fn nullable() -> VortexResult<ArrayRef> {
    encode(
        PrimitiveArray::from_option_iter(
            skewed(4000, 3)
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i % 7 != 3).then_some(v as i32)),
        ),
        0,
        BLOCK_VALUES,
    )
}

fn sliced() -> VortexResult<ArrayRef> {
    i64s(1, 2 * BLOCK_VALUES)?.slice(1500..4200)
}

/// A slice across the boundary between two chunks of bins.
fn across_chunks() -> VortexResult<ArrayRef> {
    let values = skewed(CHUNK_VALUES + 3000, 4);
    encode(PrimitiveArray::from_iter(values), 0, BLOCK_VALUES)?
        .slice(CHUNK_VALUES - 700..CHUNK_VALUES + 900)
}

#[rstest]
#[case::i64_lag0(i64s(0, BLOCK_VALUES))]
#[case::i64_lag3_4k(i64s(3, 4 * BLOCK_VALUES))]
#[case::i16(narrow())]
#[case::u32_lag1(unsigned())]
#[case::u8(bytes())]
#[case::nullable_i32(nullable())]
#[case::sliced(sliced())]
#[case::across_chunks(across_chunks())]
fn test_conformance(#[case] array: VortexResult<ArrayRef>) -> VortexResult<()> {
    let array = array?;
    let session = array_session();
    crate::initialize(&session);
    let ctx: &mut ExecutionCtx = &mut session.create_execution_ctx();
    test_array_consistency(&array, ctx);
    test_filter_conformance(&array, ctx);
    test_take_conformance(&array, ctx);
    test_mask_conformance(&array, ctx);
    test_cast_conformance(&array, ctx);
    test_binary_numeric_array(&array, ctx);
    Ok(())
}
