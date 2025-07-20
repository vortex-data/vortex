// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::NullBufferBuilder;
use num_traits::{AsPrimitive, PrimInt};
use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexExpect as _, vortex_panic};

use crate::IntoArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::varbin::VarBinArray;
use crate::validity::Validity;

/// Builder for variable-length binary arrays.
///
/// This builder allows efficient construction of VarBinArray by accumulating
/// values and building the offsets and data buffers incrementally.
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
    /// Create a new builder with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new builder with the specified capacity.
    ///
    /// This pre-allocates space for `len` elements, which can improve performance
    /// when the final size is known.
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
    /// Append a value to the builder.
    ///
    /// The value can be `None` to append a null value, or `Some(bytes)` to append
    /// the given byte slice.
    pub fn append(&mut self, value: Option<&[u8]>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    #[inline]
    /// Append a non-null value to the builder.
    ///
    /// The value will be converted to bytes and appended to the data buffer.
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
        self.validity.append_non_null();
    }

    #[inline]
    /// Append a null value to the builder.
    pub fn append_null(&mut self) {
        self.offsets.push(self.offsets[self.offsets.len() - 1]);
        self.validity.append_null();
    }

    #[inline]
    /// Append multiple null values to the builder.
    ///
    /// This is more efficient than calling `append_null()` multiple times.
    pub fn append_n_nulls(&mut self, n: usize) {
        self.offsets.push_n(self.offsets[self.offsets.len() - 1], n);
        self.validity.append_n_nulls(n);
    }

    #[inline]
    /// Append multiple values from a single buffer with pre-computed offsets.
    ///
    /// This is an efficient way to append multiple values when you already have
    /// the concatenated data and end offsets.
    pub fn append_values(&mut self, values: &[u8], end_offsets: impl Iterator<Item = O>, num: usize)
    where
        O: 'static,
        usize: AsPrimitive<O>,
    {
        self.offsets
            .extend(end_offsets.map(|offset| offset + self.data.len().as_()));
        self.data.extend_from_slice(values);
        self.validity.append_n_non_nulls(num);
    }

    /// Finish building and return the constructed VarBinArray.
    ///
    /// This consumes the builder and returns the final array with the specified data type.
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
            array.scalar_at(0).unwrap(),
            Scalar::utf8("hello".to_string(), Nullable)
        );
        assert!(array.scalar_at(1).unwrap().is_null());
    }
}
