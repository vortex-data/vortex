// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::mem;

use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
#[expect(deprecated)]
use crate::canonical::ToCanonical as _;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

pub struct BoolBuilder {
    dtype: DType,
    inner: BitBufferMut,
    nulls: LazyBitBufferBuilder,
}

impl BoolBuilder {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, DEFAULT_BUILDER_CAPACITY)
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: BitBufferMut::with_capacity(capacity),
            nulls: LazyBitBufferBuilder::new(capacity),
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
        self.inner.append_n(value, n);
        self.nulls.append_n_non_nulls(n)
    }

    /// Finishes the builder directly into a [`BoolArray`].
    pub fn finish_into_bool(&mut self) -> BoolArray {
        assert_eq!(
            self.nulls.len(),
            self.inner.len(),
            "Null count and value count should match when calling BoolBuilder::finish."
        );

        BoolArray::new(
            mem::take(&mut self.inner).freeze(),
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
        self.inner.append_n(false, n);
        self.nulls.append_n_nulls(n)
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "BoolBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        match scalar.as_bool().value() {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }

        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        #[expect(deprecated)]
        let bool_array = array.to_bool();

        self.inner.append_buffer(&bool_array.to_bit_buffer());
        self.nulls.append_validity_mask(
            bool_array
                .as_ref()
                .validity()
                .vortex_expect("validity_mask")
                .to_mask(
                    bool_array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .vortex_expect("Failed to compute validity mask"),
        );
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
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
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::BoolBuilder;
    use crate::builders::bool::BoolArray;
    use crate::builders::builder_with_capacity;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

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
    fn tests() -> VortexResult<()> {
        let len = 1000;
        let chunk_count = 10;
        let chunk = make_opt_bool_chunks(len, chunk_count);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk
            .clone()
            .append_to_builder(builder.as_mut(), &mut ctx)?;

        #[expect(deprecated)]
        let canon_into = builder.finish().to_bool();
        #[expect(deprecated)]
        let into_canon = chunk.to_bool();

        assert!(
            canon_into
                .validity()?
                .mask_eq(&into_canon.validity()?, &mut ctx)?
        );
        assert_eq!(canon_into.to_bit_buffer(), into_canon.to_bit_buffer());
        Ok(())
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
        let expected = BoolArray::from_iter([Some(true), Some(false), None]);
        assert_arrays_eq!(&array, &expected);

        // Test wrong dtype error.
        let mut builder = BoolBuilder::with_capacity(Nullability::NonNullable, 10);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
