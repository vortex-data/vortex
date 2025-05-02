use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, ScalarAtFn, TakeFn, filter, scalar_at, take,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::{VortexResult, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::{PrimitiveScalar, Scalar};
use zigzag::{ZigZag as ExternalZigZag, ZigZag};

use crate::{ZigZagArray, ZigZagEncoding};

impl ComputeVTable for ZigZagEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

impl FilterKernel for ZigZagEncoding {
    fn filter(&self, array: &ZigZagArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let encoded = filter(array.encoded(), mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(FilterKernelAdapter(ZigZagEncoding).lift());

impl ScalarAtFn<&ZigZagArray> for ZigZagEncoding {
    fn scalar_at(&self, array: &ZigZagArray, index: usize) -> VortexResult<Scalar> {
        let scalar = scalar_at(array.encoded(), index)?;
        if scalar.is_null() {
            return Ok(scalar.reinterpret_cast(array.ptype()));
        }

        let pscalar = PrimitiveScalar::try_from(&scalar)?;
        match_each_unsigned_integer_ptype!(pscalar.ptype(), |$P| {
            Ok(Scalar::primitive(
                <<$P as ZigZagEncoded>::Int>::decode(pscalar.typed_value::<$P>().ok_or_else(|| {
                    vortex_err!(
                        "Cannot decode provided scalar: expected {}, got ptype {}",
                        std::any::type_name::<$P>(),
                        pscalar.ptype()
                    )
                })?),
                array.dtype().nullability(),
            ))
        })
    }
}

impl TakeFn<&ZigZagArray> for ZigZagEncoding {
    fn take(&self, array: &ZigZagArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let encoded = take(array.encoded(), indices)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

trait ZigZagEncoded {
    type Int: ZigZag;
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
    use vortex_array::compute::{
        SearchResult, SearchSortedSide, filter, scalar_at, search_sorted, take,
    };
    use vortex_array::validity::Validity;
    use vortex_array::vtable::EncodingVTable;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::ZigZagEncoding;

    #[test]
    pub fn search_sorted_uncompressed() {
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-189, -160, 1].into_array().to_canonical().unwrap(),
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            search_sorted(&zigzag, -169, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(1)
        );
    }

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
            scalar_at(&zigzag, 1).unwrap(),
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
