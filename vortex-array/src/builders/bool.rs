use std::any::Any;

use arrow_buffer::BooleanBufferBuilder;
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_panic, VortexResult};
use vortex_mask::AllOr;

use crate::array::BoolArray;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::ArrayBuilder;
use crate::validity::Validity;
use crate::{Array, Canonical, IntoArray, IntoCanonical};

pub struct BoolBuilder {
    inner: BooleanBufferBuilder,
    nulls: LazyNullBufferBuilder,
    nullability: Nullability,
    dtype: DType,
}

impl BoolBuilder {
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, 1024) // Same as Arrow builders
    }

    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            inner: BooleanBufferBuilder::new(capacity),
            nulls: LazyNullBufferBuilder::new(capacity),
            nullability,
            dtype: DType::Bool(nullability),
        }
    }

    pub fn append_value(&mut self, value: bool) {
        self.append_values(value, 1)
    }

    pub fn append_values(&mut self, value: bool, n: usize) {
        self.inner.append_n(n, value);
        self.nulls.append_n_non_nulls(n)
    }

    pub fn append_option(&mut self, value: Option<bool>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
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

    fn append_nulls(&mut self, n: usize) {
        self.inner.append_n(n, false);
        self.nulls.append_n_nulls(n)
    }

    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        let array = array.into_canonical()?;
        let Canonical::Bool(array) = array else {
            vortex_bail!("Expected Canonical::Bool, found {:?}", array);
        };

        self.inner.append_buffer(&array.boolean_buffer());

        match array.validity_mask()?.boolean_buffer() {
            AllOr::All => {
                self.nulls.append_n_non_nulls(array.len());
            }
            AllOr::None => self.nulls.append_n_nulls(array.len()),
            AllOr::Some(validity) => self.nulls.append_buffer(validity.clone()),
        }

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        let bools = self.inner.finish();

        let nulls = self.nulls.finish();
        let validity = match (self.nullability, nulls) {
            (NonNullable, None) => Validity::NonNullable,
            (Nullable, None) => Validity::AllValid,
            (Nullable, Some(arr)) => Validity::from(arr),
            _ => vortex_panic!("Invalid nullability/nulls combination"),
        };

        Ok(BoolArray::try_new(bools, validity)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::array::{BoolArray, ChunkedArray};
    use crate::builders::builder_with_capacity;
    use crate::{Array, IntoArray, IntoCanonical};

    fn make_opt_bool_chunks(len: usize, chunk_count: usize) -> Array {
        let mut rng = StdRng::seed_from_u64(0);

        (0..chunk_count)
            .map(|_| {
                BoolArray::from_iter((0..len).map(|_| match rng.gen_range::<u8, _>(0..=2) {
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
        chunk.clone().canonicalize_into(builder.as_mut()).unwrap();
        let canon_into = builder
            .finish()
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_bool()
            .unwrap();

        let into_canon = chunk.clone().into_canonical().unwrap().into_bool().unwrap();

        assert_eq!(canon_into.validity(), into_canon.validity());
        assert_eq!(canon_into.boolean_buffer(), into_canon.boolean_buffer());
    }
}
