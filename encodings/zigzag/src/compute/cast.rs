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
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::ZigZagEncoding;
    use vortex_array::encoding::ArrayEncoding;

    #[test]
    fn test_cast_zigzag_i32_to_i64() {
        let values = buffer![-100i32, -1, 0, 1, 100].into_array();
        let zigzag = ZigZagEncoding.encode(&values, None).unwrap().unwrap();
        
        let casted = cast(zigzag.as_ref(), &DType::Primitive(PType::I64, Nullability::NonNullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Primitive(PType::I64, Nullability::NonNullable));
        
        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i64>(), &[-100i64, -1, 0, 1, 100]);
    }

    #[test]
    fn test_cast_zigzag_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(-10i32), None, Some(0), Some(10), None]).into_array();
        let zigzag = ZigZagEncoding.encode(&values, None).unwrap().unwrap();
        
        let casted = cast(zigzag.as_ref(), &DType::Primitive(PType::I64, Nullability::Nullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Primitive(PType::I64, Nullability::Nullable));
    }

    #[rstest]
    #[case(buffer![-100i32, -50, -1, 0, 1, 50, 100].into_array())]
    #[case(buffer![-1000i64, -1, 0, 1, 1000].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(-5i16), None, Some(0), Some(5), None]).into_array())]
    #[case(buffer![i32::MIN, -1, 0, 1, i32::MAX].into_array())]
    fn test_cast_zigzag_conformance(#[case] array: vortex_array::ArrayRef) {
        let zigzag = ZigZagEncoding.encode(&array, None).unwrap().unwrap();
        test_cast_conformance(zigzag.as_ref());
    }
}