use std::any::Any;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;

use crate::arrays::{BoolArray, PrimitiveArray};
use crate::builders::ArrayBuilder;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef};

/// Builder for [`PrimitiveArray`].
pub struct PrimitiveBuilder<T> {
    values: BufferMut<T>,
    nulls: LazyNullBufferBuilder,
    dtype: DType,
}

impl<T: NativePType> PrimitiveBuilder<T> {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, 1024) // Same as Arrow builders
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Primitive(T::PTYPE, nullability),
        }
    }

    /// Append a `Mask` to the null buffer.
    pub fn append_mask(&mut self, mask: Mask) {
        self.nulls.append_validity_mask(mask);
    }

    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.nulls.append(true);
    }

    pub fn append_option(&mut self, value: Option<T>) {
        match value {
            Some(value) => {
                self.values.push(value);
                self.nulls.append(true);
            }
            None => self.append_null(),
        }
    }

    pub fn values(&self) -> &[T] {
        self.values.as_ref()
    }

    /// Create a new handle to the next `len` uninitialized values in the builder.
    ///
    /// All reads/writes through the handle to the values buffer or the validity buffer will operate
    /// on indices relative to the start of the range.
    ///
    ///
    /// ## Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
    /// use vortex_dtype::Nullability;
    ///
    /// // Create a new builder.
    /// let mut builder: PrimitiveBuilder<i32> = PrimitiveBuilder::with_capacity(Nullability::NonNullable, 5);
    ///
    /// // Populate the values in reverse order.
    /// let mut range = builder.uninit_range(5);
    /// for i in [4, 3, 2, 1, 0] {
    ///     range[i] = MaybeUninit::new(i as i32);
    /// }
    /// range.finish();
    ///
    /// let built = builder.finish_into_primitive();
    ///
    /// assert_eq!(built.as_slice::<i32>(), &[0i32, 1, 2, 3, 4]);
    /// ```
    pub fn uninit_range(&mut self, len: usize) -> UninitRange<T> {
        let offset = self.values.len();
        assert!(
            offset + len <= self.values.capacity(),
            "uninit_range of len {len} exceeds builder capacity"
        );

        UninitRange {
            offset,
            len,
            builder: self,
        }
    }

    pub fn finish_into_primitive(&mut self) -> PrimitiveArray {
        let nulls = self.nulls.finish();

        if let Some(null_buf) = nulls.as_ref() {
            assert_eq!(
                null_buf.len(),
                self.values.len(),
                "null buffer length must equal value buffer length"
            );
        }

        let validity = match (nulls, self.dtype().nullability()) {
            (None, Nullability::NonNullable) => Validity::NonNullable,
            (Some(_), Nullability::NonNullable) => {
                vortex_panic!("Non-nullable builder has null values")
            }
            (None, Nullability::Nullable) => Validity::AllValid,
            (Some(nulls), Nullability::Nullable) => {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::Array(BoolArray::from(nulls.into_inner()).into_array())
                }
            }
        };

        PrimitiveArray::new(std::mem::take(&mut self.values).freeze(), validity)
    }

    pub fn extend_with_iterator(&mut self, iter: impl IntoIterator<Item = T>, mask: Mask) {
        self.values.extend(iter);
        self.extend_with_validity_mask(mask)
    }

    fn extend_with_validity_mask(&mut self, validity_mask: Mask) {
        self.nulls.append_validity_mask(validity_mask);
    }
}

impl<T: NativePType> ArrayBuilder for PrimitiveBuilder<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.values.push_n(T::default(), n);
        self.nulls.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.values.push_n(T::default(), n);
        self.nulls.append_n_nulls(n);
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        let array = array.to_canonical()?.into_primitive()?;
        if array.ptype() != T::PTYPE {
            vortex_bail!("Cannot extend from array with different ptype");
        }

        self.values.extend_from_slice(array.as_slice::<T>());

        self.extend_with_validity_mask(array.validity_mask()?);

        Ok(())
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_primitive().into_array()
    }
}

pub struct UninitRange<'a, T> {
    offset: usize,
    len: usize,
    builder: &'a mut PrimitiveBuilder<T>,
}

impl<T> Deref for UninitRange<'_, T> {
    type Target = [MaybeUninit<T>];

    fn deref(&self) -> &[MaybeUninit<T>] {
        let start = self.builder.values.as_ptr();
        unsafe {
            // SAFETY: start + len is checked on construction to be in range.
            let dst = std::slice::from_raw_parts(start, self.len);

            // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
            let dst: &[MaybeUninit<T>] = std::mem::transmute(dst);

            dst
        }
    }
}

impl<T> DerefMut for UninitRange<'_, T> {
    fn deref_mut(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.builder.values.spare_capacity_mut()[..self.len]
    }
}

impl<T> UninitRange<'_, T> {
    /// Set a validity bit at the given index. The index is relative to the start of this range
    /// of the builder.
    pub fn set_bit(&mut self, index: usize, v: bool) {
        self.builder.nulls.set_bit(self.offset + index, v);
    }

    /// Set values from an initialized range.
    pub fn copy_from_init(&mut self, offset: usize, len: usize, src: &[T])
    where
        T: Copy,
    {
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
        let uninit_src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(src) };

        let dst = &mut self[offset..][..len];
        dst.copy_from_slice(uninit_src);
    }

    /// Finish building this range, marking it as initialized and advancing the length of the
    /// underlying values buffer.
    pub fn finish(self) {
        // SAFETY: constructor enforces that offset + len does not exceed the capacity of the array.
        unsafe { self.builder.values.set_len(self.offset + self.len) };
    }
}
