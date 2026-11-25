// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::builtin::cast::CastReduce;
use vortex_error::VortexResult;

use crate::rle::RLE;
use crate::rle::RLEArrayExt;
use crate::rle::RLEArraySlotsExt;

/// Returns `true` if every value of `src` is representable in `dst` without error.
fn is_infallible_primitive_cast(src: PType, dst: PType) -> bool {
    if src == dst {
        return true;
    }
    match (src, dst) {
        (PType::F16, PType::F32 | PType::F64) | (PType::F32, PType::F64) => true,
        (s, d) if s.is_unsigned_int() && d.is_int() => d.byte_width() > s.byte_width(),
        (s, d) if s.is_signed_int() && d.is_signed_int() => d.byte_width() > s.byte_width(),
        _ => false,
    }
}

impl CastReduce for RLE {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(target_ptype, target_nullability) = *dtype else {
            return Ok(None);
        };

        // Dropping nullability cannot be pushed down to `indices`: `indices` covers full
        // 1024-element chunks and may hold nulls outside `offset..offset + len` (from slicing
        // or chunk-tail padding) even when every logical row in the array is valid.
        if !target_nullability.is_nullable() && array.dtype().is_nullable() {
            return Ok(None);
        }

        // Fallible ptype casts cannot be pushed down to `values` when the array does not cover
        // its full chunk storage, because out-of-slice run values could trigger spurious overflow
        // or sign errors. Note: because `indices` is padded to a multiple of `FL_CHUNK_SIZE`
        // (1024), `covers_all_indices` is false whenever `array.len()` is not a multiple of 1024
        // even if unsliced, so fallible narrowing casts decompress in that case (a future
        // optimization could check cached `values` min/max stats to push down safely).
        let src_ptype = array.dtype().as_ptype();
        let covers_all_indices = array.offset() == 0 && array.len() == array.indices().len();
        if !is_infallible_primitive_cast(src_ptype, target_ptype)
            && (!covers_all_indices || array.dtype().is_nullable())
        {
            return Ok(None);
        }

        // Cast RLE values.
        let casted_values = array
            .values()
            .cast(DType::Primitive(target_ptype, Nullability::NonNullable))?;

        // Cast RLE indices such that validity matches the target dtype.
        let casted_indices = array
            .indices()
            .cast(array.indices().dtype().with_nullability(target_nullability))?;

        Ok(Some(
            RLE::try_new(
                casted_values,
                casted_indices,
                array.values_idx_offsets().clone(),
                array.offset(),
                array.len(),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RLEData;
    use crate::rle::RLEArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn rle(primitive: &PrimitiveArray, ctx: &mut ExecutionCtx) -> RLEArray {
        RLEData::encode(primitive.as_view(), ctx).unwrap()
    }

    #[test]
    fn try_cast_rle_success() {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter([10u8, 20, 30, 40, 50]),
            Validity::from_iter([true, true, true, true, true]),
        );
        let encoded = rle(&primitive, &mut ctx);

        let casted = encoded
            .into_array()
            .cast(DType::Primitive(PType::U16, Nullability::NonNullable))
            .unwrap();
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([10u16, 20, 30, 40, 50]),
            &mut ctx
        );
    }

    #[test]
    #[should_panic]
    fn try_cast_rle_fail() {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::new(
            Buffer::from_iter([10u8, 20, 30, 40, 50]),
            Validity::from_iter([true, false, true, true, false]),
        );
        let encoded = rle(&primitive, &mut ctx);
        let result = encoded
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));
        result.unwrap();
    }

    #[rstest]
    #[case::u8(
        PrimitiveArray::new(
            Buffer::from_iter([0u8, 10, 20, 30, 40, 50]),
            Validity::NonNullable,
        )
    )]
    #[case::u8_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u8, 10, 20, 30, 40]),
            Validity::from_iter([true, false, true, false, true]),
        )
    )]
    #[case::u16(
        PrimitiveArray::new(
            Buffer::from_iter([0u16, 100, 200, 300, 400, 500]),
            Validity::NonNullable,
        )
    )]
    #[case::u16_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u16, 100, 200, 300, 400]),
            Validity::from_iter([false, true, false, true, true]),
        )
    )]
    #[case::u32(
        PrimitiveArray::new(
            Buffer::from_iter([0u32, 1000, 2000, 3000, 4000]),
            Validity::NonNullable,
        )
    )]
    #[case::u32_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u32, 1000, 2000, 3000, 4000]),
            Validity::from_iter([true, true, false, false, true]),
        )
    )]
    #[case::u64(
        PrimitiveArray::new(
            Buffer::from_iter([0u64, 10000, 20000, 30000]),
            Validity::NonNullable,
        )
    )]
    #[case::u64_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u64, 10000, 20000, 30000]),
            Validity::from_iter([false, false, true, true]),
        )
    )]
    fn test_cast_rle_conformance(#[case] primitive: PrimitiveArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let rle_array = rle(&primitive, &mut ctx);
        test_cast_conformance(&rle_array.into_array(), &mut ctx);
    }

    /// Regression: casting a nullable RLE slice with no nulls in its logical range to a
    /// non-nullable dtype must succeed even though `indices` still holds nulls outside the slice
    /// and false padding bits in the trailing chunk.
    #[test]
    fn test_cast_rle_nullable_slice_to_non_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0..2500i32).map(|i| (i % 13 != 0).then_some(i / 19 - 50)),
        );
        let encoded = rle(&primitive, &mut ctx).into_array();
        // Rows 1..12 are all valid, while row 0 and chunk padding (2500..3072) are null.
        let sliced = encoded.slice(1..12)?.execute::<ArrayRef>(&mut ctx)?;
        let casted = sliced.cast(DType::Primitive(PType::I32, Nullability::NonNullable))?;
        let expected = primitive
            .into_array()
            .slice(1..12)?
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable))?;
        assert_arrays_eq!(casted, expected, &mut ctx);
        Ok(())
    }

    /// Regression: casting a sliced RLE array whose logical rows are all non-negative to an
    /// unsigned integer must succeed even if earlier out-of-slice rows in the same chunk are
    /// negative.
    #[test]
    fn test_cast_rle_sliced_fallible_ptype_change() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter([-10i32, -10, 0, 5, 5, 5, 20]);
        let encoded = rle(&primitive, &mut ctx).into_array();
        let sliced = encoded.slice(2..7)?.execute::<ArrayRef>(&mut ctx)?;
        let casted = sliced.cast(DType::Primitive(PType::U32, Nullability::NonNullable))?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([0u32, 5, 5, 5, 20]),
            &mut ctx
        );
        Ok(())
    }
}
