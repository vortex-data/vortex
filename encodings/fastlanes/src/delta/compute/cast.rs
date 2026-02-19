// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::expr::CastReduce;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::delta::DeltaArray;
use crate::delta::DeltaVTable;

impl CastReduce for DeltaVTable {
    fn cast(array: &DeltaArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Delta encoding stores differences between consecutive values, which requires
        // unsigned integers to avoid overflow issues. Signed integers could produce
        // negative deltas that wouldn't fit in the unsigned delta representation.
        // This encoding is optimized for monotonically increasing sequences.
        let DType::Primitive(target_ptype, _) = dtype else {
            return Ok(None);
        };

        let DType::Primitive(source_ptype, _) = array.dtype() else {
            vortex_panic!("delta should be primitive typed");
        };

        // TODO(DK): narrows can be safe but we must decompress to compute the maximum value.
        if target_ptype.is_signed_int() || source_ptype.bit_width() > target_ptype.bit_width() {
            return Ok(None);
        }

        // Cast both bases and deltas to the target type
        let casted_bases = array.bases().cast(dtype.with_nullability(NonNullable))?;
        let casted_deltas = array.deltas().cast(dtype.clone())?;

        // Create a new DeltaArray with the casted components
        Ok(Some(
            DeltaArray::try_from_delta_compress_parts(casted_bases, casted_deltas)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::Buffer;

    use crate::delta::DeltaArray;

    #[test]
    fn test_cast_delta_u8_to_u32() {
        let primitive = PrimitiveArray::new(
            Buffer::copy_from(vec![10u8, 20, 30, 40, 50]),
            vortex_array::validity::Validity::NonNullable,
        );
        let array = DeltaArray::try_from_primitive_array(&primitive).unwrap();

        let casted = array
            .to_array()
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
        // DeltaArray doesn't support nullable arrays - the validity is handled at the DeltaArray level
        // Create a non-nullable array and then add validity to the DeltaArray
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![100u16, 0, 200, 300, 0]),
            vortex_array::validity::Validity::NonNullable,
        );
        let array = DeltaArray::try_from_primitive_array(&values).unwrap();

        let casted = array
            .to_array()
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
            Buffer::copy_from(vec![0u8, 10, 20, 30, 40, 50]),
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u16(
        PrimitiveArray::new(
            Buffer::copy_from(vec![0u16, 100, 200, 300, 400, 500]),
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u32(
        PrimitiveArray::new(
            Buffer::copy_from(vec![0u32, 1000, 2000, 3000, 4000]),
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u64(
        PrimitiveArray::new(
            Buffer::copy_from(vec![0u64, 10000, 20000, 30000]),
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    fn test_cast_delta_conformance(#[case] primitive: PrimitiveArray) {
        let delta_array = DeltaArray::try_from_primitive_array(&primitive).unwrap();
        test_cast_conformance(delta_array.as_ref());
    }
}
