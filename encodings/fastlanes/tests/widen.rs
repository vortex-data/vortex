// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `Widen` operations over a `BitPacked` child.
//!
//! Selection and comparison keep the stored width of the child, aggregates read the child, and
//! canonical execution widens the values to the logical dtype of the `Widen`.

#![cfg(test)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_mask::Mask;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_fastlanes::initialize(&session);
    session
});

/// Returns a `u64` `Widen` over `values` packed at `bit_width` bits.
fn widen_bitpacked(values: PrimitiveArray, bit_width: u8) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let packed = BitPacked::encode(&values.into_array(), bit_width, &mut ctx)?.into_array();
    let dtype = DType::Primitive(PType::U64, packed.dtype().nullability());

    Ok(WidenArray::try_new(packed, dtype)?.into_array())
}

#[test]
fn test_operations() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = widen_bitpacked(PrimitiveArray::from_iter([0u8, 1, 3, 7]), 3)?;

    let selected = array.filter(Mask::from_iter([true, false, true, true]))?;
    assert!(selected.is::<Widen>());
    assert_eq!(selected.as_::<Widen>().values().dtype(), &PType::U8.into());
    assert_arrays_eq!(selected, buffer![0u64, 3, 7].into_array(), &mut ctx);

    assert_arrays_eq!(
        array.take(buffer![3u32, 1].into_array())?,
        buffer![7u64, 1].into_array(),
        &mut ctx
    );
    assert_arrays_eq!(
        array.binary(
            ConstantArray::new(Scalar::from(3u64), 4).into_array(),
            Operator::Lt
        )?,
        BoolArray::from_iter([true, true, false, false]),
        &mut ctx
    );
    assert_eq!(
        array.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<u64>(),
        &[0, 1, 3, 7]
    );

    Ok(())
}

#[rstest]
#[case::sorted(PrimitiveArray::from_iter([0u8, 1, 3, 7]))]
#[case::constant(PrimitiveArray::from_iter([5u8, 5, 5, 5]))]
#[case::partially_valid(PrimitiveArray::from_option_iter([None, Some(6u8), Some(2), None]))]
fn test_aggregates_match_canonical(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = widen_bitpacked(values.clone(), 3)?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
    let options = NumericalAggregateOpts::default();

    // Each aggregate gets a fresh `Widen`. A statistic cached by an earlier aggregate, such as the
    // min and max that `min_max` caches, would otherwise answer a later one before its kernel runs.
    let widen = || widen_bitpacked(values.clone(), 3);

    assert_eq!(
        min_max(&widen()?, &mut ctx, options)?,
        min_max(&canonical, &mut ctx, options)?
    );
    assert_eq!(
        is_constant(&widen()?, &mut ctx)?,
        is_constant(&canonical, &mut ctx)?
    );
    assert_eq!(
        is_sorted(&widen()?, &mut ctx)?,
        is_sorted(&canonical, &mut ctx)?
    );
    assert_eq!(
        is_strict_sorted(&widen()?, &mut ctx)?,
        is_strict_sorted(&canonical, &mut ctx)?
    );
    assert_eq!(sum(&widen()?, &mut ctx)?, sum(&canonical, &mut ctx)?);

    Ok(())
}
