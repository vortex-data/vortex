use arrow_buffer::NullBufferBuilder;
use num_traits::{AsPrimitive, PrimInt};
use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType};
use vortex_error::{vortex_panic, VortexExpect as _};

use crate::array::primitive::PrimitiveArray;
use crate::array::varbin::VarBinArray;
use crate::validity::Validity;
use crate::IntoArrayData;

pub struct VarBinBuilder<O: NativePType> {
    offsets: BufferMut<O>,
    data: BufferMut<u8>,
    validity: NullBufferBuilder,
}

impl<O: NativePType + PrimInt> Default for VarBinBuilder<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: NativePType + PrimInt> VarBinBuilder<O> {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(len: usize) -> Self {
        let mut offsets = BufferMut::with_capacity(len + 1);
        offsets.push(O::zero());
        Self {
            offsets,
            data: BufferMut::empty(),
            validity: NullBufferBuilder::new(len),
        }
    }

    #[inline]
    pub fn push(&mut self, value: Option<&[u8]>) {
        match value {
            Some(v) => self.push_value(v),
            None => self.push_null(),
        }
    }

    #[inline]
    pub fn push_value(&mut self, value: impl AsRef<[u8]>) {
        let slice = value.as_ref();
        self.offsets
            .push(O::from(self.data.len() + slice.len()).unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert sum of {} and {} to offset of type {}",
                    self.data.len(),
                    slice.len(),
                    std::any::type_name::<O>()
                )
            }));
        self.data.extend_from_slice(slice);
        self.validity.append_non_null();
    }

    #[inline]
    pub fn push_null(&mut self) {
        self.offsets.push(self.offsets[self.offsets.len() - 1]);
        self.validity.append_null();
    }

    #[inline]
    pub fn push_values(&mut self, values: &[u8], end_offsets: impl Iterator<Item = O>, num: usize)
    where
        O: 'static,
        usize: AsPrimitive<O>,
    {
        self.offsets
            .extend(end_offsets.map(|offset| offset + self.data.len().as_()));
        self.data.extend_from_slice(values);
        self.validity.append_n_non_nulls(num);
    }

    pub fn finish(mut self, dtype: DType) -> VarBinArray {
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        let nulls = self.validity.finish();

        let validity = if dtype.is_nullable() {
            nulls.map(Validity::from).unwrap_or(Validity::AllValid)
        } else {
            assert!(nulls.is_none(), "dtype and validity mismatch");
            Validity::NonNullable
        };

        VarBinArray::try_new(offsets.into_array(), self.data.freeze(), dtype, validity)
            .vortex_expect("Unexpected error while building VarBinArray")
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::array::varbin::builder::VarBinBuilder;
    use crate::compute::scalar_at;
    use crate::{ArrayDType, IntoArrayData};

    #[test]
    fn test_builder() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(0);
        builder.push(Some(b"hello"));
        builder.push(None);
        builder.push(Some(b"world"));
        let array = builder.finish(DType::Utf8(Nullable)).into_array();

        assert_eq!(array.len(), 3);
        assert_eq!(array.dtype().nullability(), Nullable);
        assert_eq!(
            scalar_at(&array, 0).unwrap(),
            Scalar::utf8("hello".to_string(), Nullable)
        );
        assert!(scalar_at(&array, 1).unwrap().is_null());
    }
}
