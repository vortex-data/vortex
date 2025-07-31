// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::{ZigZagArray, ZigZagVTable};

impl CastKernel for ZigZagVTable {
    fn cast(&self, array: &ZigZagArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if this is just a nullability change
        if array.dtype().eq_ignore_nullability(dtype) {
            // For nullability-only changes, we can avoid decoding
            // Cast the encoded array to handle nullability
            let new_encoded = cast(array.encoded(), &array.encoded().dtype().with_nullability(dtype.nullability()))?;
            
            Ok(Some(ZigZagArray::try_new(new_encoded)?.into_array()))
        } else if let (DType::Primitive(target_ptype, target_nullability), DType::Primitive(source_ptype, _)) = 
            (dtype, array.dtype()) {
            // ZigZag only works with signed integers, so we can optimize integer width changes
            // by casting the underlying unsigned encoded array
            if source_ptype.is_signed_int() && target_ptype.is_signed_int() {
                // Map signed target type to unsigned for the encoded array
                let encoded_target_ptype = match target_ptype {
                    PType::I8 => PType::U8,
                    PType::I16 => PType::U16,
                    PType::I32 => PType::U32,
                    PType::I64 => PType::U64,
                    _ => unreachable!("Already checked is_signed_int"),
                };
                
                let encoded_target_dtype = DType::Primitive(encoded_target_ptype, *target_nullability);
                let new_encoded = cast(array.encoded(), &encoded_target_dtype)?;
                
                Ok(Some(ZigZagArray::try_new(new_encoded)?.into_array()))
            } else {
                // For non-integer targets (e.g., float), we need to decode
                let decoded = array.to_canonical()?;
                cast(decoded.as_ref(), dtype).map(Some)
            }
        } else {
            // For non-primitive targets, we need to decode
            let decoded = array.to_canonical()?;
            cast(decoded.as_ref(), dtype).map(Some)
        }
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

    use crate::{zigzag_encode, ZigZagArray};

    #[test]
    fn test_cast_zigzag_i32_to_i64() {
        let values = PrimitiveArray::from_iter([-100i32, -1, 0, 1, 100]);
        let zigzag = zigzag_encode(values).unwrap();
        
        let casted = cast(zigzag.as_ref(), &DType::Primitive(PType::I64, Nullability::NonNullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Primitive(PType::I64, Nullability::NonNullable));
        
        // Verify the result is still a ZigZagArray (not decoded)
        // Note: The result might be wrapped, so let's check the encoding ID
        assert_eq!(casted.encoding().id().as_ref(), "vortex.zigzag", "Cast should preserve ZigZag encoding");
        
        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i64>(), &[-100i64, -1, 0, 1, 100]);
    }
    
    #[test]
    fn test_cast_zigzag_width_changes() {
        // Test i32 to i16 (narrowing)
        let values = PrimitiveArray::from_iter([100i32, -50, 0, 25, -100]);
        let zigzag = zigzag_encode(values).unwrap();
        
        let casted = cast(zigzag.as_ref(), &DType::Primitive(PType::I16, Nullability::NonNullable)).unwrap();
        assert_eq!(casted.encoding().id().as_ref(), "vortex.zigzag", "Should remain ZigZag encoded");
        
        let decoded = casted.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i16>(), &[100i16, -50, 0, 25, -100]);
        
        // Test i16 to i64 (widening)
        let values16 = PrimitiveArray::from_iter([1000i16, -500, 0, 250, -1000]);
        let zigzag16 = zigzag_encode(values16).unwrap();
        
        let casted64 = cast(zigzag16.as_ref(), &DType::Primitive(PType::I64, Nullability::NonNullable)).unwrap();
        assert_eq!(casted64.encoding().id().as_ref(), "vortex.zigzag", "Should remain ZigZag encoded");
        
        let decoded64 = casted64.to_canonical().unwrap().into_primitive().unwrap();
        assert_eq!(decoded64.as_slice::<i64>(), &[1000i64, -500, 0, 250, -1000]);
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
    fn test_cast_zigzag_conformance(#[case] array: ZigZagArray) {
        test_cast_conformance(array.as_ref());
    }
}