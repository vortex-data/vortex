use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, TakeKernel, TakeKernelAdapter, filter, take,
};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ZigZagArray, ZigZagEncoding};

impl FilterKernel for ZigZagEncoding {
    fn filter(&self, array: &ZigZagArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let encoded = filter(array.encoded(), mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(FilterKernelAdapter(ZigZagEncoding).lift());

impl TakeKernel for ZigZagEncoding {
    fn take(&self, array: &ZigZagArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let encoded = take(array.encoded(), indices)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(TakeKernelAdapter(ZigZagEncoding).lift());

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
    use vortex_array::vtable::EncodingVTable;
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
}
