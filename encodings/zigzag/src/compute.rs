use vortex_array::compute::{
    filter, scalar_at, slice, take, ComputeVTable, FilterFn, FilterMask, ScalarAtFn, SliceFn,
    TakeFn,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::{PrimitiveScalar, Scalar};
use zigzag::{ZigZag as ExternalZigZag, ZigZag};

use crate::{ZigZagArray, ZigZagEncoding};

impl ComputeVTable for ZigZagEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl FilterFn<ZigZagArray> for ZigZagEncoding {
    fn filter(&self, array: &ZigZagArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let encoded = filter(&array.encoded(), mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

impl ScalarAtFn<ZigZagArray> for ZigZagEncoding {
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

impl SliceFn<ZigZagArray> for ZigZagEncoding {
    fn slice(&self, array: &ZigZagArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ZigZagArray::try_new(slice(array.encoded(), start, stop)?)?.into_array())
    }
}

impl TakeFn<ZigZagArray> for ZigZagEncoding {
    fn take(&self, array: &ZigZagArray, indices: &ArrayData) -> VortexResult<ArrayData> {
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
    use vortex_array::array::{BooleanBuffer, PrimitiveArray};
    use vortex_array::compute::{
        filter, scalar_at, search_sorted, take, SearchResult, SearchSortedSide,
    };
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::ZigZagArray;

    #[test]
    pub fn search_sorted_uncompressed() {
        let zigzag = ZigZagArray::encode(&PrimitiveArray::from(vec![-189, -160, 1]).into_array())
            .unwrap()
            .into_array();
        assert_eq!(
            search_sorted(&zigzag, -169, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(1)
        );
    }

    #[test]
    pub fn nullable_scalar_at() {
        let zigzag = ZigZagArray::encode(
            &PrimitiveArray::from_vec(vec![-189, -160, 1], Validity::AllValid).into_array(),
        )
        .unwrap();
        assert_eq!(
            scalar_at(&zigzag, 1).unwrap(),
            Scalar::primitive(-160, Nullability::Nullable)
        );
    }

    #[test]
    fn take_zigzag() {
        let zigzag =
            ZigZagArray::encode(&PrimitiveArray::from(vec![-189, -160, 1]).into_array()).unwrap();
        let indices = PrimitiveArray::from(vec![0, 2]).into_array();
        let actual = take(zigzag, indices).unwrap().into_primitive().unwrap();
        let expected = ZigZagArray::encode(&PrimitiveArray::from(vec![-189, 1]).into_array())
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(actual.into_buffer(), expected.into_buffer());
    }

    #[test]
    fn filter_zigzag() {
        let zigzag =
            ZigZagArray::encode(&PrimitiveArray::from(vec![-189, -160, 1]).into_array()).unwrap();
        let filter_mask = BooleanBuffer::from(vec![true, false, true]).into();
        let actual = filter(&zigzag.into_array(), filter_mask)
            .unwrap()
            .into_primitive()
            .unwrap();
        let expected = ZigZagArray::encode(&PrimitiveArray::from(vec![-189, 1]).into_array())
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(actual.into_buffer(), expected.into_buffer());
    }
}
