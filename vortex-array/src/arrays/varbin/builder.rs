// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::{BitBufferMut, BufferMut};
use vortex_dtype::{DType, IntegerPType};
use vortex_error::vortex_panic;

use crate::IntoArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::varbin::VarBinArray;
use crate::validity::Validity;

pub struct VarBinBuilder<O: IntegerPType> {
    offsets: BufferMut<O>,
    data: BufferMut<u8>,
    validity: BitBufferMut,
}

impl<O: IntegerPType> Default for VarBinBuilder<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: IntegerPType> VarBinBuilder<O> {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(len: usize) -> Self {
        let mut offsets = BufferMut::with_capacity(len + 1);
        offsets.push(O::zero());
        Self {
            offsets,
            data: BufferMut::empty(),
            validity: BitBufferMut::with_capacity(len),
        }
    }

    #[inline]
    pub fn append(&mut self, value: Option<&[u8]>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    #[inline]
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) {
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
        self.validity.append_true();
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.offsets.push(self.offsets[self.offsets.len() - 1]);
        self.validity.append_false();
    }

    #[inline]
    pub fn append_n_nulls(&mut self, n: usize) {
        self.offsets.push_n(self.offsets[self.offsets.len() - 1], n);
        self.validity.append_n(false, n);
    }

    #[inline]
    pub fn append_values(&mut self, values: &[u8], end_offsets: impl Iterator<Item = O>, num: usize)
    where
        O: 'static,
        usize: AsPrimitive<O>,
    {
        self.offsets
            .extend(end_offsets.map(|offset| offset + self.data.len().as_()));
        self.data.extend_from_slice(values);
        self.validity.append_n(true, num);
    }

    pub fn finish(self, dtype: DType) -> VarBinArray {
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        let nulls = self.validity.freeze();

        let validity = Validity::from_bit_buffer(nulls, dtype.nullability());

        // SAFETY: The builder maintains all invariants:
        // - Offsets are monotonically increasing starting from 0 (guaranteed by builder logic).
        // - Bytes buffer contains exactly the data referenced by offsets.
        // - Validity matches the dtype nullability.
        // - UTF-8 validity is ensured by the caller when using DType::Utf8.
        unsafe {
            VarBinArray::new_unchecked(offsets.into_array(), self.data.freeze(), dtype, validity)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::arrays::varbin::builder::VarBinBuilder;

    #[test]
    fn test_builder() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(0);
        builder.append(Some(b"hello"));
        builder.append(None);
        builder.append(Some(b"world"));
        let array = builder.finish(DType::Utf8(Nullable));

        assert_eq!(array.len(), 3);
        assert_eq!(array.dtype().nullability(), Nullable);
        assert_eq!(
            array.scalar_at(0),
            Scalar::utf8("hello".to_string(), Nullable)
        );
        assert!(array.scalar_at(1).is_null());
    }
}
