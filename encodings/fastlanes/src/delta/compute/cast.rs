// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;

use crate::delta::{DeltaArray, DeltaVTable};

impl CastKernel for DeltaVTable {
    fn cast(&self, array: &DeltaArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Delta encoding stores differences between consecutive values, which requires
        // unsigned integers to avoid overflow issues. Signed integers could produce
        // negative deltas that wouldn't fit in the unsigned delta representation.
        // This encoding is optimized for monotonically increasing sequences.
        if !matches!(dtype, DType::Primitive(ptype, _) if ptype.is_unsigned_int()) {
            return Ok(None);
        }

        // Cast both bases and deltas to the target type
        let casted_bases = cast(array.bases(), &dtype.with_nullability(NonNullable))?;
        let casted_deltas = cast(array.deltas(), dtype)?;

        // Create a new DeltaArray with the casted components
        Ok(Some(
            DeltaArray::try_from_delta_compress_parts(casted_bases, casted_deltas)?.into_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(DeltaVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::delta::DeltaArray;

    #[test]
    fn test_cast_delta_u8_to_u32() {
        let primitive = PrimitiveArray::new(
            Buffer::copy_from(vec![10u8, 20, 30, 40, 50]),
            vortex_array::validity::Validity::NonNullable,
        );
        let array = DeltaArray::try_from_primitive_array(&primitive).unwrap();

        let casted = cast(
            array.as_ref(),
            &DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        // Verify by decoding
        let decoded = casted.to_primitive();
        assert_eq!(decoded.as_slice::<u32>(), &[10u32, 20, 30, 40, 50]);
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

        let casted = cast(
            array.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
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
