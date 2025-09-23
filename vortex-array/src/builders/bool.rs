// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use arrow_buffer::BooleanBufferBuilder;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_ensure};
use vortex_mask::Mask;
use vortex_scalar::{BoolScalar, Scalar};

use crate::arrays::BoolArray;
use crate::builders::{ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder};
use crate::canonical::{Canonical, ToCanonical};
use crate::{Array, ArrayRef, IntoArray};

pub struct BoolBuilder {
    dtype: DType,
    inner: BooleanBufferBuilder,
    nulls: LazyNullBufferBuilder,
}

impl BoolBuilder {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, DEFAULT_BUILDER_CAPACITY)
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: BooleanBufferBuilder::new(capacity),
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Bool(nullability),
        }
    }

    /// Appends a boolean value to the builder.
    pub fn append_value(&mut self, value: bool) {
        self.append_values(value, 1)
    }

    /// Appends the same boolean value multiple times to the builder.
    ///
    /// This method appends the given boolean value `n` times.
    pub fn append_values(&mut self, value: bool, n: usize) {
        self.inner.append_n(n, value);
        self.nulls.append_n_non_nulls(n)
    }

    /// Finishes the builder directly into a [`BoolArray`].
    pub fn finish_into_bool(&mut self) -> BoolArray {
        assert_eq!(
            self.nulls.len(),
            self.inner.len(),
            "Null count and value count should match when calling BoolBuilder::finish."
        );

        BoolArray::from_bool_buffer(
            self.inner.finish(),
            self.nulls.finish_with_nullability(self.dtype.nullability()),
        )
    }
}

impl ArrayBuilder for BoolBuilder {
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
        self.inner.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.append_values(false, n)
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.inner.append_n(n, false);
        self.nulls.append_n_nulls(n)
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "BoolBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        let bool_scalar = BoolScalar::try_from(scalar)?;
        match bool_scalar.value() {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }

        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let bool_array = array.to_bool();

        self.inner.append_buffer(bool_array.boolean_buffer());
        self.nulls.append_validity_mask(bool_array.validity_mask());
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.inner.capacity() {
            self.inner.reserve(capacity - self.inner.capacity());
            self.nulls.ensure_capacity(capacity);
        }
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_bool().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::Bool(self.finish_into_bool())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::{BoolArray, ChunkedArray};
    use crate::builders::{ArrayBuilder, BoolBuilder, builder_with_capacity};
    use crate::canonical::ToCanonical;
    use crate::vtable::ValidityHelper;
    use crate::{ArrayRef, IntoArray};

    fn make_opt_bool_chunks(len: usize, chunk_count: usize) -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(0);

        (0..chunk_count)
            .map(|_| {
                BoolArray::from_iter((0..len).map(|_| match rng.random_range::<u8, _>(0..=2) {
                    0 => Some(false),
                    1 => Some(true),
                    2 => None,
                    _ => unreachable!(),
                }))
                .into_array()
            })
            .collect::<ChunkedArray>()
            .into_array()
    }

    #[test]
    fn tests() {
        let len = 1000;
        let chunk_count = 10;
        let chunk = make_opt_bool_chunks(len, chunk_count);

        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.clone().append_to_builder(builder.as_mut());

        let canon_into = builder.finish().to_bool();
        let into_canon = chunk.to_bool();

        assert_eq!(canon_into.validity(), into_canon.validity());
        assert_eq!(canon_into.boolean_buffer(), into_canon.boolean_buffer());
    }

    #[test]
    fn test_append_scalar() {
        let mut builder = BoolBuilder::with_capacity(Nullability::Nullable, 10);

        // Test appending true value.
        let true_scalar = Scalar::bool(true, Nullability::Nullable);
        builder.append_scalar(&true_scalar).unwrap();

        // Test appending false value.
        let false_scalar = Scalar::bool(false, Nullability::Nullable);
        builder.append_scalar(&false_scalar).unwrap();

        // Test appending null value.
        let null_scalar = Scalar::null(DType::Bool(Nullability::Nullable));
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_bool();
        assert_eq!(array.len(), 3);

        // Check actual values.
        assert!(array.boolean_buffer().value(0));
        assert!(!array.boolean_buffer().value(1));
        // The third value is null, but the buffer might have any value.

        // Check validity - first two should be valid, third should be null.
        assert!(array.validity().is_valid(0));
        assert!(array.validity().is_valid(1));
        assert!(!array.validity().is_valid(2));

        // Test wrong dtype error.
        let mut builder = BoolBuilder::with_capacity(Nullability::NonNullable, 10);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
