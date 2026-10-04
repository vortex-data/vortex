// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::for_::FoR;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;
impl CastReduce for FoR {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only push down nullability change.
        if !array.dtype().eq_ignore_nullability(dtype) {
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
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FoR;
    use crate::FoRArray;

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

    #[rstest]
    // 127 is stored as 255, which wraps to -1 in `i8`, and would decode as -129 in `i16`.
    #[case::widen_wrapped_offset(
        FoR::encode(PrimitiveArray::from_iter([-128i8, 127]), &mut SESSION.create_execution_ctx()),
        PType::I16,
        Some(PrimitiveArray::from_iter([-128i16, 127]))
    )]
    // The reference and the offsets fit `i8`, but 200 does not.
    #[case::narrow_out_of_range(
        FoR::encode(PrimitiveArray::from_iter([100i16, 200]), &mut SESSION.create_execution_ctx()),
        PType::I8,
        None
    )]
    // The values fit `u32`, but the reference -5 does not.
    #[case::reference_out_of_range(
        FoR::try_new(buffer![10i32, 11, 12].into_array(), Scalar::from(-5i32)),
        PType::U32,
        Some(PrimitiveArray::from_iter([5u32, 6, 7]))
    )]
    fn cast_type_change(
        #[case] array: VortexResult<FoRArray>,
        #[case] ptype: PType,
        #[case] expected: Option<PrimitiveArray>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let cast = array?
            .into_array()
            .cast(DType::Primitive(ptype, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(&mut ctx);
        match expected {
            Some(expected) => assert_arrays_eq!(cast?, expected, &mut ctx),
            None => assert!(cast.is_err()),
        }
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
