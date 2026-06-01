// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::delta::Delta;
use crate::delta::array::DeltaArrayExt;

impl CastReduce for Delta {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(target_ptype, target_nullability) = dtype else {
            return Ok(None);
        };

        // Only a nullability change at the same primitive type can reuse the encoded
        // components; everything else defers to the generic decompress path.
        if array.dtype().as_ptype() != *target_ptype {
            return Ok(None);
        }
        if *target_nullability == NonNullable && array.dtype().nullability() != NonNullable {
            return Ok(None);
        }

        let casted_bases = array.bases().cast(dtype.with_nullability(NonNullable))?;
        let casted_deltas = array.deltas().cast(dtype.clone())?;

        Ok(Some(
            Delta::try_new(casted_bases, casted_deltas, array.offset(), array.len())?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;
    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_cast_delta_unsigned_widening_wraps() {
        // A valid non-monotonic unsigned sequence whose deltas wrap at the source width.
        // 200u8 -> 50u8 stores delta 0x6A (106); decode does 200 wrapping_add 106 = 306
        // mod 256 = 50. Widening the delta to u32 preserves the value 106 but changes the
        // reconstruction modulus to 2^32, so 200 + 106 = 306 instead of 50. Every index
        // after a wrap is corrupted, so the widening cast must not reuse the components.
        let primitive = PrimitiveArray::from_iter([200u8, 50, 75, 10, 255]);
        let array =
            Delta::try_from_primitive_array(&primitive, &mut SESSION.create_execution_ctx())
                .unwrap();

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();

        assert_arrays_eq!(casted, PrimitiveArray::from_iter([200u32, 50, 75, 10, 255]));
    }

    #[test]
    fn test_cast_delta_same_width_float_target() {
        // u32 and f32 share a bit width but are not interchangeable: the fast path must not
        // reuse the integer components for a float target (that would error in `try_new`).
        // The cast must succeed via the decompress-and-re-encode fallback.
        let primitive = PrimitiveArray::from_iter([10u32, 20, 30, 40, 50]);
        let array =
            Delta::try_from_primitive_array(&primitive, &mut SESSION.create_execution_ctx())
                .unwrap();

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::F32, Nullability::NonNullable))
            .unwrap();

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([10f32, 20.0, 30.0, 40.0, 50.0])
        );
    }

    #[test]
    fn test_cast_delta_add_nullability() -> VortexResult<()> {
        // Same ptype, only adding nullability — handled by the kernel without decompressing.
        let values = PrimitiveArray::from_iter([10u32, 20, 5, 30, 15]);
        let array = Delta::try_from_primitive_array(&values, &mut SESSION.create_execution_ctx())?;

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))?;

        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([Some(10u32), Some(20), Some(5), Some(30), Some(15)])
        );
        Ok(())
    }

    #[test]
    fn test_cast_delta_nullability_preserves_nulls() -> VortexResult<()> {
        // A nullable Delta array carries its validity in the deltas child; a same-ptype
        // nullability cast must round-trip the null positions.
        let values =
            PrimitiveArray::from_option_iter([Some(10u32), None, Some(30), Some(15), None]);
        let array = Delta::try_from_primitive_array(&values, &mut SESSION.create_execution_ctx())?;

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))?;

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([Some(10u32), None, Some(30), Some(15), None])
        );
        Ok(())
    }

    #[test]
    fn test_cast_delta_drop_nullability_with_nulls_errors() -> VortexResult<()> {
        // Dropping nullability when real nulls are present must error, matching the generic
        // cast semantics rather than silently discarding the null mask.
        let values = PrimitiveArray::from_option_iter([Some(10u32), None, Some(30)]);
        let array = Delta::try_from_primitive_array(&values, &mut SESSION.create_execution_ctx())?;

        // The cast is lazy; the nullability check fires when it is executed.
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(
            result.is_err(),
            "dropping nullability with real nulls must error, got {result:?}"
        );
        Ok(())
    }

    #[test]
    fn test_cast_delta_u8_to_u32() {
        let primitive = PrimitiveArray::from_iter([10u8, 20, 30, 40, 50]);
        let array =
            Delta::try_from_primitive_array(&primitive, &mut SESSION.create_execution_ctx())
                .unwrap();

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        // Verify by decoding
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([10u32, 20, 30, 40, 50]));
    }

    #[test]
    fn test_cast_delta_nullable() {
        // A combined ptype + nullability change goes through the generic decompress path.
        let values = PrimitiveArray::new(
            buffer![100u16, 0, 200, 300, 0],
            vortex_array::validity::Validity::NonNullable,
        );
        let array =
            Delta::try_from_primitive_array(&values, &mut SESSION.create_execution_ctx()).unwrap();

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case::u8(
        PrimitiveArray::new(
            buffer![0u8, 10, 20, 30, 40, 50],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u16(
        PrimitiveArray::new(
            buffer![0u16, 100, 200, 300, 400, 500],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u32(
        PrimitiveArray::new(
            buffer![0u32, 1000, 2000, 3000, 4000],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u64(
        PrimitiveArray::new(
            buffer![0u64, 10000, 20000, 30000],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    fn test_cast_delta_conformance(#[case] primitive: PrimitiveArray) {
        let delta_array =
            Delta::try_from_primitive_array(&primitive, &mut SESSION.create_execution_ctx())
                .unwrap();
        test_cast_conformance(&delta_array.into_array());
    }
}
