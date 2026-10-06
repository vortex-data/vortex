// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Narrow operations on compressed integer children.
//!
//! BitPacked children retain their stored width through selection and comparison. Canonical
//! execution widens them to Narrow's logical dtype.

#![cfg(test)]

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Narrow;
use vortex_array::arrays::NarrowArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::narrow::NarrowArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::integer::integer_dtype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_mask::Mask;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

#[test]
fn test_narrow_bitpacked_child() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let packed = BitPacked::encode(&buffer![0u8, 1, 3, 7].into_array(), 3, &mut ctx)?;
    let array = NarrowArray::try_new(packed.into_array(), PType::U64.into())?.into_array();
    let selected = array.filter(Mask::from_iter([true, false, true, true]))?;

    assert!(selected.is::<Narrow>());
    assert_eq!(selected.as_::<Narrow>().values().dtype(), &PType::U8.into());
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
    assert_eq!(sum(&array, &mut ctx)?, Scalar::from(11u64).into_nullable());
    assert_eq!(
        array.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<u64>(),
        &[0, 1, 3, 7]
    );

    Ok(())
}

#[test]
fn test_decimal_canonicalization_keeps_bitpacked_storage() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DecimalDType::new(76, 2);
    let packed = BitPacked::encode(&buffer![0i32, 1, 3, 7].into_array(), 3, &mut ctx)?;
    let values = NarrowArray::try_new(
        packed.into_array(),
        integer_dtype(DecimalType::I256, Nullability::NonNullable),
    )?
    .into_array();
    let array = DecimalArray::try_new_values(values, dtype)?;
    let canonical = array
        .clone()
        .into_array()
        .execute::<DecimalArray>(&mut ctx)?;
    assert!(
        canonical
            .values()
            .as_::<Narrow>()
            .values()
            .is::<BitPacked>()
    );
    assert_eq!(canonical.values_type(), DecimalType::I32);
    assert!(canonical.as_ref().buffer_handles().is_empty());
    assert_eq!(
        sum(array.as_ref(), &mut ctx)?,
        Scalar::decimal(DecimalValue::I32(11), dtype, Nullability::Nullable)
    );
    assert_arrays_eq!(
        array.filter(Mask::from_iter([true, false, true, true]))?,
        DecimalArray::from_iter([0i32, 3, 7], dtype),
        &mut ctx
    );

    Ok(())
}
