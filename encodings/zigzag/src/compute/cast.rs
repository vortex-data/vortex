// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ZigZagArray, ZigZagVTable};

impl CastKernel for ZigZagVTable {
    fn cast(&self, array: &ZigZagArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_signed_int() {
            return Ok(None);
        }

        let new_encoded_dtype =
            DType::Primitive(dtype.as_ptype().to_unsigned(), dtype.nullability());
        let new_encoded = cast(array.encoded(), &new_encoded_dtype)?;
        Ok(Some(ZigZagArray::try_new(new_encoded)?.into_array()))
    }
}

register_kernel!(CastKernelAdapter(ZigZagVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::{Array, ToCanonical};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{ZigZagArray, zigzag_encode};

    #[test]
    fn test_cast_zigzag_i32_to_i64() {
        let values = PrimitiveArray::from_iter([-100i32, -1, 0, 1, 100]);
        let zigzag = zigzag_encode(values).unwrap();

        let casted = cast(
            zigzag.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        // Verify the result is still a ZigZagArray (not decoded)
        // Note: The result might be wrapped, so let's check the encoding ID
        assert_eq!(
            casted.encoding().id().as_ref(),
            "vortex.zigzag",
            "Cast should preserve ZigZag encoding"
        );

        let decoded = casted.to_primitive();
        assert_eq!(decoded.as_slice::<i64>(), &[-100i64, -1, 0, 1, 100]);
    }

    #[test]
    fn test_cast_zigzag_width_changes() {
        // Test i32 to i16 (narrowing)
        let values = PrimitiveArray::from_iter([100i32, -50, 0, 25, -100]);
        let zigzag = zigzag_encode(values).unwrap();

        let casted = cast(
            zigzag.as_ref(),
            &DType::Primitive(PType::I16, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.encoding().id().as_ref(),
            "vortex.zigzag",
            "Should remain ZigZag encoded"
        );

        let decoded = casted.to_primitive();
        assert_eq!(decoded.as_slice::<i16>(), &[100i16, -50, 0, 25, -100]);

        // Test i16 to i64 (widening)
        let values16 = PrimitiveArray::from_iter([1000i16, -500, 0, 250, -1000]);
        let zigzag16 = zigzag_encode(values16).unwrap();

        let casted64 = cast(
            zigzag16.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted64.encoding().id().as_ref(),
            "vortex.zigzag",
            "Should remain ZigZag encoded"
        );

        let decoded64 = casted64.to_primitive();
        assert_eq!(decoded64.as_slice::<i64>(), &[1000i64, -500, 0, 250, -1000]);
    }

    #[test]
    fn test_cast_zigzag_nullable() {
        let values =
            PrimitiveArray::from_option_iter([Some(-10i32), None, Some(0), Some(10), None]);
        let zigzag = zigzag_encode(values).unwrap();

        let casted = cast(
            zigzag.as_ref(),
            &DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(zigzag_encode(PrimitiveArray::from_iter([-100i32, -50, -1, 0, 1, 50, 100])).unwrap())]
    #[case(zigzag_encode(PrimitiveArray::from_iter([-1000i64, -1, 0, 1, 1000])).unwrap())]
    #[case(zigzag_encode(PrimitiveArray::from_option_iter([Some(-5i16), None, Some(0), Some(5), None])).unwrap())]
    #[case(zigzag_encode(PrimitiveArray::from_iter([i32::MIN, -1, 0, 1, i32::MAX])).unwrap())]
    fn test_cast_zigzag_conformance(#[case] array: ZigZagArray) {
        test_cast_conformance(array.as_ref());
    }
}
