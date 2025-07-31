// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ZigZagArray, ZigZagVTable};

impl CastKernel for ZigZagVTable {
    fn cast(&self, array: &ZigZagArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // ZigZag encoding maps signed integers to unsigned for better compression.
        // We need to decode back to signed before casting to other types.
        let decoded = array.to_canonical()?;
        cast(decoded.as_ref(), dtype).map(Some)
    }
}

register_kernel!(CastKernelAdapter(ZigZagVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::ToCanonical;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::zigzag_encode;

    #[test]
    fn test_cast_zigzag_i32_to_i64() {
        let values = PrimitiveArray::from_iter([-100i32, -1, 0, 1, 100]);
        let zigzag = zigzag_encode(values).unwrap();
        
        let casted = cast(zigzag.as_ref(), &DType::Primitive(PType::I64, Nullability::NonNullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Primitive(PType::I64, Nullability::NonNullable));
        
        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i64>(), &[-100i64, -1, 0, 1, 100]);
    }

    #[test]
    fn test_cast_zigzag_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(-10i32), None, Some(0), Some(10), None]);
        let zigzag = zigzag_encode(values).unwrap();
        
        let casted = cast(zigzag.as_ref(), &DType::Primitive(PType::I64, Nullability::Nullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Primitive(PType::I64, Nullability::Nullable));
    }

    #[rstest]
    #[case(zigzag_encode(PrimitiveArray::from_iter([-100i32, -50, -1, 0, 1, 50, 100])).unwrap())]
    #[case(zigzag_encode(PrimitiveArray::from_iter([-1000i64, -1, 0, 1, 1000])).unwrap())]
    #[case(zigzag_encode(PrimitiveArray::from_option_iter([Some(-5i16), None, Some(0), Some(5), None])).unwrap())]
    #[case(zigzag_encode(PrimitiveArray::from_iter([i32::MIN, -1, 0, 1, i32::MAX])).unwrap())]
    fn test_cast_zigzag_conformance(#[case] array: crate::ZigZagArray) {
        test_cast_conformance(array.as_ref());
    }
}