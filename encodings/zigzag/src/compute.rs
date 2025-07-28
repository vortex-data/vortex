// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, TakeKernel, TakeKernelAdapter, MaskKernel, MaskKernelAdapter, 
    filter, take, mask,
};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ZigZagArray, ZigZagVTable};

impl FilterKernel for ZigZagVTable {
    fn filter(&self, array: &ZigZagArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let encoded = filter(array.encoded(), mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(FilterKernelAdapter(ZigZagVTable).lift());

impl TakeKernel for ZigZagVTable {
    fn take(&self, array: &ZigZagArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let encoded = take(array.encoded(), indices)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(TakeKernelAdapter(ZigZagVTable).lift());

impl MaskKernel for ZigZagVTable {
    fn mask(&self, array: &ZigZagArray, filter_mask: &Mask) -> VortexResult<ArrayRef> {
        let encoded = mask(array.encoded(), filter_mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ZigZagVTable).lift());

pub(crate) trait ZigZagEncoded {
    type Int: zigzag::ZigZag;
}

impl ZigZagEncoded for u8 {
    type Int = i8;
}

impl ZigZagEncoded for u16 {
    type Int = i16;
}

impl ZigZagEncoded for u32 {
    type Int = i32;
}

impl ZigZagEncoded for u64 {
    type Int = i64;
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BooleanBuffer, PrimitiveArray};
    use vortex_array::compute::{filter, take};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::ZigZagEncoding;

    #[test]
    pub fn nullable_scalar_at() {
        let zigzag = ZigZagEncoding
            .encode(
                &PrimitiveArray::new(buffer![-189, -160, 1], Validity::AllValid)
                    .to_canonical()
                    .unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            zigzag.scalar_at(1).unwrap(),
            Scalar::primitive(-160, Nullability::Nullable)
        );
    }

    #[test]
    fn take_zigzag() {
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-189, -160, 1].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();

        let indices = buffer![0, 2].into_array();
        let actual = take(&zigzag, &indices).unwrap().to_primitive().unwrap();
        let expected = ZigZagEncoding
            .encode(&buffer![-189, 1].into_array().to_canonical().unwrap(), None)
            .unwrap()
            .unwrap()
            .to_primitive()
            .unwrap();
        assert_eq!(actual.as_slice::<i32>(), expected.as_slice::<i32>());
    }

    #[test]
    fn filter_zigzag() {
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-189, -160, 1].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        let filter_mask = BooleanBuffer::from(vec![true, false, true]).into();
        let actual = filter(&zigzag, &filter_mask)
            .unwrap()
            .to_primitive()
            .unwrap();
        let expected = ZigZagEncoding
            .encode(&buffer![-189, 1].into_array().to_canonical().unwrap(), None)
            .unwrap()
            .unwrap()
            .to_primitive()
            .unwrap();
        assert_eq!(actual.as_slice::<i32>(), expected.as_slice::<i32>());
    }
    
    #[test]
    fn test_filter_conformance() {
        use vortex_array::compute::conformance::filter::test_filter;
        use vortex_array::compute::conformance::binary_numeric::test_numeric;
        
        // Test with i32 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-189i32, -160, 1, 42, -73].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        test_filter(zigzag.as_ref());
        
        // Test with i64 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![1000i64, -2000, 3000, -4000, 5000].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        test_filter(zigzag.as_ref());
        
        // Test with nullable values
        let array = PrimitiveArray::from_option_iter([Some(-10i16), None, Some(20), Some(-30), None]);
        let zigzag = ZigZagEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_filter(zigzag.as_ref());
    }
    
    #[test]
    fn test_mask_conformance() {
        use vortex_array::compute::conformance::mask::test_mask;
        
        // Test with i32 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-100i32, 200, -300, 400, -500].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        test_mask(zigzag.as_ref());
        
        // Test with i8 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-127i8, 0, 127, -1, 1].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        test_mask(zigzag.as_ref());
    }
    
    #[test]
    fn test_numeric_conformance() {
        use vortex_array::compute::conformance::binary_numeric::test_numeric;
        
        // Test binary numeric operations with i32
        let zigzag1 = ZigZagEncoding
            .encode(
                &buffer![10i32, -20, 30, -40, 50].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        let zigzag2 = ZigZagEncoding
            .encode(
                &buffer![5i32, -10, 15, -20, 25].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        test_numeric(zigzag1.into_array());
        test_numeric(zigzag2.into_array());
    }
}
