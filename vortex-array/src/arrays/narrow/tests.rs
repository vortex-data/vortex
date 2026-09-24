// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use super::Narrow;
use super::NarrowArray;
use super::NarrowArraySlotsExt;
use crate::ArrayContext;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

#[rstest]
fn test_integer_widths(
    #[values(PType::I8, PType::I16, PType::I32, PType::U8, PType::U16, PType::U32)] storage: PType,
    #[values(PType::I16, PType::I32, PType::I64, PType::U16, PType::U32, PType::U64)]
    logical: PType,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![0i64, 1, 42]
        .into_array()
        .cast(storage.into())?
        .execute::<PrimitiveArray>(&mut ctx)?
        .into_array();
    let result = NarrowArray::try_new(values.clone(), logical.into());
    if storage.is_signed_int() != logical.is_signed_int()
        || storage.byte_width() >= logical.byte_width()
    {
        assert!(result.is_err());
        return Ok(());
    }
    let array = result?;
    assert_eq!(array.dtype(), &DType::from(logical));
    assert_eq!(array.values().dtype(), &DType::from(storage));
    assert_eq!(array.as_ref().nbytes(), 3 * storage.byte_width() as u64);
    let expected = values.cast(logical.into())?;
    assert_arrays_eq!(array, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_reject_nonintegers_and_nullability_changes() {
    assert!(NarrowArray::try_new(buffer![1f32].into_array(), PType::F64.into()).is_err());
    assert!(NarrowArray::try_new(buffer![1i8].into_array(), PType::F64.into()).is_err());
    assert!(
        NarrowArray::try_new(
            buffer![1i8].into_array(),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .is_err()
    );
    assert!(
        NarrowArray::try_new(
            PrimitiveArray::from_option_iter([Some(1i8), None]).into_array(),
            PType::I64.into(),
        )
        .is_err()
    );
}

#[test]
fn test_flatten_nested_wrappers() -> VortexResult<()> {
    let inner = NarrowArray::try_new(buffer![1i8, 2, 3].into_array(), PType::I16.into())?;
    let outer = NarrowArray::try_new(inner.into_array(), PType::I64.into())?;
    assert!(outer.values().is::<Primitive>());
    assert_eq!(outer.values().dtype(), &DType::from(PType::I8));
    Ok(())
}

#[rstest]
#[case(vec![-128i64, 127], PType::I8)]
#[case(vec![0, 255], PType::I16)]
#[case(vec![-32769, 32768], PType::I32)]
#[case(vec![i64::MIN, i64::MAX], PType::I64)]
fn test_encode_signed(#[case] values: Vec<i64>, #[case] storage: PType) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let original = PrimitiveArray::from_iter(values);
    let encoded = NarrowArray::encode(original.clone(), &mut ctx)?;
    assert_eq!(encoded.dtype(), original.dtype());
    if storage == PType::I64 {
        assert!(encoded.is::<Primitive>());
    } else {
        assert_eq!(
            encoded.as_::<Narrow>().values().dtype(),
            &DType::from(storage)
        );
    }
    assert_arrays_eq!(encoded, original, &mut ctx);
    Ok(())
}

#[rstest]
#[case(vec![0u64, 255], PType::U8)]
#[case(vec![0, 256], PType::U16)]
#[case(vec![0, 65536], PType::U32)]
#[case(vec![0, u64::MAX], PType::U64)]
fn test_encode_unsigned(#[case] values: Vec<u64>, #[case] storage: PType) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let original = PrimitiveArray::from_iter(values);
    let encoded = NarrowArray::encode(original.clone(), &mut ctx)?;
    assert_eq!(encoded.dtype(), original.dtype());
    if storage == PType::U64 {
        assert!(encoded.is::<Primitive>());
    } else {
        assert_eq!(
            encoded.as_::<Narrow>().values().dtype(),
            &DType::from(storage)
        );
    }
    assert_arrays_eq!(encoded, original, &mut ctx);
    Ok(())
}

#[test]
fn test_encode_ignores_null_slot_values() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let original = PrimitiveArray::new(
        buffer![i64::MIN, 42, i64::MAX],
        Validity::from_iter([false, true, false]),
    );
    let encoded = NarrowArray::encode(original.clone(), &mut ctx)?;
    assert_eq!(
        encoded.as_::<Narrow>().values().dtype(),
        &DType::Primitive(PType::I8, Nullability::Nullable)
    );
    assert_eq!(
        encoded.execute_scalar(0, &mut ctx)?,
        Scalar::null(original.dtype().clone())
    );
    assert_eq!(
        encoded.execute_scalar(1, &mut ctx)?,
        Scalar::primitive(42i64, Nullability::Nullable)
    );
    assert_arrays_eq!(encoded, original, &mut ctx);
    Ok(())
}

#[rstest]
#[case(Vec::new())]
#[case(vec![None, None])]
fn test_empty_and_all_null(#[case] values: Vec<Option<i64>>) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let original = PrimitiveArray::from_option_iter(values);
    let encoded = NarrowArray::encode(original.clone(), &mut ctx)?;
    assert_eq!(
        encoded.as_::<Narrow>().values().dtype(),
        &DType::Primitive(PType::I8, Nullability::Nullable)
    );
    assert_arrays_eq!(encoded, original, &mut ctx);
    Ok(())
}

#[test]
fn test_selection_preserves_narrow_storage() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = NarrowArray::try_new(buffer![10i8, 20, 30, 40].into_array(), PType::I64.into())?
        .into_array();
    let sliced = array.slice(1..3)?;
    let filtered = array.filter(Mask::from_iter([true, false, true, false]))?;
    let taken = array.take(buffer![3u32, 1].into_array())?;
    for result in [&sliced, &filtered, &taken] {
        assert_eq!(result.dtype(), &DType::from(PType::I64));
        assert_eq!(
            result.as_::<Narrow>().values().dtype(),
            &DType::from(PType::I8)
        );
    }
    assert_arrays_eq!(sliced, buffer![20i64, 30].into_array(), &mut ctx);
    assert_arrays_eq!(filtered, buffer![10i64, 30].into_array(), &mut ctx);
    assert_arrays_eq!(taken, buffer![40i64, 20].into_array(), &mut ctx);

    let indices = PrimitiveArray::from_option_iter([Some(3u32), None, Some(0)]).into_array();
    let taken = array.take(indices)?;
    assert_eq!(
        taken.as_::<Narrow>().values().dtype(),
        &DType::Primitive(PType::I8, Nullability::Nullable)
    );
    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_option_iter([Some(40i64), None, Some(10)]),
        &mut ctx
    );

    let masked = array.mask(BoolArray::from_iter([true, false, true, false]).into_array())?;
    assert_eq!(
        masked.as_::<Narrow>().values().dtype(),
        &DType::Primitive(PType::I8, Nullability::Nullable)
    );
    assert_arrays_eq!(
        masked,
        PrimitiveArray::from_option_iter([Some(10i64), None, Some(30), None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
fn test_compare_signed_constants(
    #[values(-1000i64, -129, -128, 0, 127, 128, 1000)] constant: i64,
    #[values(
        Operator::Eq,
        Operator::NotEq,
        Operator::Lt,
        Operator::Lte,
        Operator::Gt,
        Operator::Gte
    )]
    op: Operator,
    #[values(false, true)] swapped: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values =
        PrimitiveArray::from_option_iter([Some(-128i8), Some(0), None, Some(127)]).into_array();
    let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    let narrow = NarrowArray::try_new(values, dtype)?.into_array();
    let wide =
        PrimitiveArray::from_option_iter([Some(-128i64), Some(0), None, Some(127)]).into_array();
    let rhs =
        ConstantArray::new(Scalar::primitive(constant, Nullability::Nullable), 4).into_array();
    let (actual, expected) = if swapped {
        (rhs.binary(narrow, op)?, rhs.binary(wide, op)?)
    } else {
        (narrow.binary(rhs.clone(), op)?, wide.binary(rhs, op)?)
    };
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[rstest]
fn test_compare_unsigned_constants(
    #[values(0u64, 255, 256, u64::MAX)] constant: u64,
    #[values(
        Operator::Eq,
        Operator::NotEq,
        Operator::Lt,
        Operator::Lte,
        Operator::Gt,
        Operator::Gte
    )]
    op: Operator,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values =
        PrimitiveArray::from_option_iter([Some(0u8), Some(127), None, Some(255)]).into_array();
    let dtype = DType::Primitive(PType::U64, Nullability::Nullable);
    let narrow = NarrowArray::try_new(values, dtype)?.into_array();
    let wide =
        PrimitiveArray::from_option_iter([Some(0u64), Some(127), None, Some(255)]).into_array();
    let rhs = ConstantArray::new(constant, 4).into_array();
    assert_arrays_eq!(
        narrow.binary(rhs.clone(), op)?,
        wide.binary(rhs, op)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_null_constant_and_arithmetic_fallback() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let narrow =
        NarrowArray::try_new(buffer![120i8, 100].into_array(), PType::I64.into())?.into_array();
    let null = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        2,
    )
    .into_array();
    assert_arrays_eq!(
        narrow.binary(null, Operator::Eq)?,
        ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 2),
        &mut ctx
    );
    let sum = narrow.binary(ConstantArray::new(120i64, 2).into_array(), Operator::Add)?;
    assert_arrays_eq!(sum, buffer![240i64, 220].into_array(), &mut ctx);
    let unwrapped = narrow.cast(PType::I8.into())?;
    assert!(unwrapped.is::<Primitive>());
    assert_arrays_eq!(unwrapped, buffer![120i8, 100].into_array(), &mut ctx);
    Ok(())
}

#[test]
fn test_narrow_pair_and_mixed_width_fallback() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs =
        NarrowArray::try_new(buffer![1i8, 5, 8].into_array(), PType::I64.into())?.into_array();
    for values in [
        buffer![2i8, 5, 7].into_array(),
        buffer![2i16, 5, 7].into_array(),
    ] {
        let rhs = NarrowArray::try_new(values, PType::I64.into())?.into_array();
        assert_arrays_eq!(
            lhs.binary(rhs, Operator::Lt)?,
            BoolArray::from_iter([true, false, false]),
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn test_dictionary_child_and_serialization() -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let values = DictArray::try_new(
        buffer![0u8, 1, 0].into_array(),
        buffer![10i8, 20].into_array(),
    )?
    .into_array();
    let array = NarrowArray::try_new(values, PType::I64.into())?.into_array();
    let array_ctx = ArrayContext::empty();
    let buffers = array.serialize(&array_ctx, &session, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(&buffer);
    }
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &session,
    )?;
    let values = decoded.as_::<Narrow>().values().clone();
    assert_eq!(values.dtype(), &DType::from(PType::I8));
    assert!(values.is::<crate::arrays::Dict>());
    assert_arrays_eq!(decoded, buffer![10i64, 20, 10].into_array(), &mut ctx);
    Ok(())
}

#[rstest]
#[case(vec![])]
#[case(vec![255])]
#[case(vec![PType::I8 as u8, 0])]
#[case(vec![PType::U8 as u8])]
#[case(vec![PType::I64 as u8])]
#[case(vec![PType::F32 as u8])]
fn test_reject_invalid_metadata(#[case] metadata: Vec<u8>) {
    let children = vec![buffer![1i8].into_array()];
    assert!(
        Narrow
            .deserialize(
                &PType::I64.into(),
                1,
                &metadata,
                &[],
                &children,
                &array_session(),
            )
            .is_err()
    );
}
