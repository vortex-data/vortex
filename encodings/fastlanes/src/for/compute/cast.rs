// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::r#for::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;
impl CastReduce for FoR {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only a nullability change is pushed down: it leaves the values, and so the non-nullable
        // references, as they are.
        //
        // Changing the type would cast the encoded values as values, but they are offsets from the
        // reference modulo 2^n. E.g. `i8` values -128..=127 have the reference -128, so 127 is
        // stored as 255, which wraps to -1, and would decode as -129 in `i16`. Decline, so the
        // decoded values are cast instead.
        if !dtype.is_int() || dtype.as_ptype() != array.ptype() {
            return Ok(None);
        }

        let casted_child = array.encoded().cast(dtype.clone())?;
        Ok(Some(
            FoR::try_new_chunked(casted_child, array.references().clone(), array.offset())?
                .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::cast::CastReduce;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::FoR;
    use crate::FoRArray;
    use crate::FoRArrayExt;
    use crate::FoRArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn for_arr(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[test]
    fn test_cast_for_i32_to_i64() {
        let for_array = for_arr(
            buffer![0i32, 10, 20, 30, 40].into_array(),
            Scalar::from(100i32),
        );

        let casted = for_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        // Verify the values after decoding
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([100i64, 110, 120, 130, 140]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_cast_for_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(0i32), None, Some(20), Some(30), None]);
        let for_array = for_arr(values.into_array(), Scalar::from(50i32));

        let casted = for_array
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    /// FoR with the minimum as the reference, over a BitPacked child of `bit_width` if given.
    fn for_over_bitpacked(values: PrimitiveArray, bit_width: Option<u8>) -> VortexResult<FoRArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let for_array = FoR::encode(values, &mut ctx)?;
        let Some(bit_width) = bit_width else {
            return Ok(for_array);
        };
        let packed = BitPacked::encode(for_array.encoded(), bit_width, &mut ctx)?;
        let reference = for_array
            .constant_reference()
            .vortex_expect("FoR::encode uses one reference");
        FoR::try_new(packed.into_array(), reference)
    }

    /// Every type change declines, and casting the decoded values gives the right answer.
    #[rstest]
    #[case::widen(PrimitiveArray::from_iter(1000i32..3000), PType::I64, Some(11))]
    #[case::widen_unsigned_to_signed(PrimitiveArray::from_iter(0u8..100), PType::I16, Some(7))]
    #[case::narrow_values_fit(PrimitiveArray::from_iter(10i32..60), PType::I8, Some(6))]
    // 127 is stored as 255, which wraps to -1 in `i8`.
    #[case::signed_range_overflows_type(PrimitiveArray::from_iter(i8::MIN..=i8::MAX), PType::I16, None)]
    #[case::wide_signed_range(PrimitiveArray::from_iter([i32::MIN, 0, i32::MAX]), PType::I64, None)]
    // The values fit `i8`, but the differences 0..=199 do not.
    #[case::narrow_differences_overflow(PrimitiveArray::from_iter(-100i32..100), PType::I8, Some(8))]
    fn cast_to_other_type(
        #[case] values: PrimitiveArray,
        #[case] ptype: PType,
        #[case] bit_width: Option<u8>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(ptype, Nullability::NonNullable);
        let array = for_over_bitpacked(values.clone(), bit_width)?;
        assert!(<FoR as CastReduce>::cast(array.as_view(), &dtype)?.is_none());
        assert_arrays_eq!(
            array.into_array().cast(dtype.clone())?,
            values.into_array().cast(dtype)?,
            &mut ctx
        );
        Ok(())
    }

    /// Values that do not fit the target fail to cast, rather than wrapping.
    #[rstest]
    #[case::narrow(PrimitiveArray::from_iter([100i16, 200]), PType::I8, None)]
    #[case::narrow_bitpacked(PrimitiveArray::from_iter(100i32..300), PType::I8, Some(8))]
    #[case::signed_to_unsigned(PrimitiveArray::from_iter(-5i8..10), PType::U8, Some(4))]
    #[case::unsigned_to_signed(PrimitiveArray::from_iter(100u8..200), PType::I8, Some(7))]
    fn cast_values_out_of_range(
        #[case] values: PrimitiveArray,
        #[case] ptype: PType,
        #[case] bit_width: Option<u8>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(ptype, Nullability::NonNullable);
        let array = for_over_bitpacked(values, bit_width)?;
        assert!(<FoR as CastReduce>::cast(array.as_view(), &dtype)?.is_none());
        let cast = array.into_array().cast(dtype)?;
        assert!(cast.execute::<PrimitiveArray>(&mut ctx).is_err());
        Ok(())
    }

    /// Values 5..10 with the reference -5, as remains after filtering away the values below zero.
    #[test]
    fn cast_reference_out_of_range() -> VortexResult<()> {
        let array = for_arr(
            buffer![10i32, 11, 12, 13, 14].into_array(),
            Scalar::from(-5i32),
        );
        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        assert!(<FoR as CastReduce>::cast(array.as_view(), &dtype)?.is_none());
        assert_arrays_eq!(
            array.into_array().cast(dtype)?,
            PrimitiveArray::from_iter([5u32, 6, 7, 8, 9]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// A slice keeps the references of the chunks it overlaps, which can be out of the target's
    /// range while the sliced values are not.
    #[test]
    fn cast_per_chunk_references_out_of_range() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..2048i32).map(|i| i - 5));
        let array = FoR::encode_chunked(values, &mut ctx)?.into_array();
        assert!(array.as_::<FoR>().constant_reference().is_none());
        let sliced = array.slice(10..2048)?;
        let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
        assert!(<FoR as CastReduce>::cast(sliced.as_::<FoR>(), &dtype)?.is_none());
        assert_arrays_eq!(
            sliced.cast(dtype)?,
            PrimitiveArray::from_iter(5u32..2043),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_per_chunk_references_nullability() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..2048i32).map(|i| (i / 1024) * 1_000 + i % 10));
        let array = FoR::encode_chunked(values.clone(), &mut ctx)?;
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let cast = <FoR as CastReduce>::cast(array.as_view(), &dtype)?
            .expect("a nullability change is pushed down");
        assert_eq!(cast.dtype(), &dtype);
        assert_arrays_eq!(cast, values.into_array().cast(dtype)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(for_arr(
        buffer![0i32, 1, 2, 3, 4].into_array(),
        Scalar::from(100i32)
    ))]
    #[case(for_arr(
        buffer![0u64, 10, 20, 30].into_array(),
        Scalar::from(1000u64)
    ))]
    #[case(for_arr(
        PrimitiveArray::from_option_iter([Some(0i16), None, Some(5), Some(10), None]).into_array(),
        Scalar::from(50i16)
    ))]
    #[case(for_arr(
        buffer![-10i32, -5, 0, 5, 10].into_array(),
        Scalar::from(-100i32)
    ))]
    fn test_cast_for_conformance(#[case] array: FoRArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
