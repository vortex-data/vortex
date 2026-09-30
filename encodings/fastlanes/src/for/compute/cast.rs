// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::r#for::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;
impl CastReduce for FoR {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // FoR only supports integer types
        if !dtype.is_int() {
            return Ok(None);
        }

        // References are always non-nullable.
        let casted_references = match array.constant_reference() {
            // A reference can be out of the target's range while the values are not, e.g. after
            // filtering away the values below zero. Decline, so the decoded values are cast.
            Some(reference) => match reference.cast(&dtype.as_nonnullable()) {
                Ok(reference) => {
                    ConstantArray::new(reference, array.references().len()).into_array()
                }
                Err(_) => return Ok(None),
            },
            // Casting per-chunk references would only fail on decode if one is out of range, so
            // only push down a nullability change, which leaves them as they are.
            None if dtype.as_ptype() == array.ptype() => array.references().clone(),
            None => return Ok(None),
        };

        // For type changes between integers, cast the components
        let casted_child = array.encoded().cast(dtype.clone())?;

        Ok(Some(
            FoR::try_new_chunked(casted_child, casted_references, array.offset())?.into_array(),
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

    use crate::FoR;
    use crate::FoRArray;
    use crate::FoRArrayExt;

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
