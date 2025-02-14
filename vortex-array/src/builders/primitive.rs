use std::any::Any;

use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_dtype::{match_each_unsigned_integer_ptype, DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

use crate::array::{BoolArray, PrimitiveArray};
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::ArrayBuilder;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray, IntoArrayVariant as _, IntoCanonical};

pub struct PrimitiveBuilder<T: NativePType> {
    pub values: BufferMut<T>,
    pub nulls: LazyNullBufferBuilder,
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

    pub fn patch(&mut self, patches: Patches, starting_at: usize) -> VortexResult<()> {
        let (array_len, indices_offset, indices, values) = patches.into_parts();
        assert!(starting_at + array_len == self.len());
        let indices = indices.into_primitive()?;
        let values = values.into_primitive()?;
        let validity = values.validity();
        let values = values.as_slice::<T>();
        match_each_unsigned_integer_ptype!(indices.ptype(), |$P| {
            self.insert_values_and_validity_at_indices::<$P>(indices, values, validity, starting_at, indices_offset)
        })
    }

    fn insert_values_and_validity_at_indices<IndexT: NativePType + AsPrimitive<usize>>(
        &mut self,
        indices: PrimitiveArray,
        values: &[T],
        validity: Validity,
        starting_at: usize,
        indices_offset: usize,
    ) -> VortexResult<()> {
        match validity {
            Validity::NonNullable => {
                for (compressed_index, decompressed_index) in
                    indices.as_slice::<IndexT>().iter().enumerate()
                {
                    let decompressed_index = decompressed_index.as_();
                    let out_index = starting_at + decompressed_index - indices_offset;
                    self.values[out_index] = values[compressed_index];
                }
            }
            _ => {
                let validity = validity.to_logical(indices.len())?;
                for (compressed_index, decompressed_index) in
                    indices.as_slice::<IndexT>().iter().enumerate()
                {
                    let decompressed_index = decompressed_index.as_();
                    let out_index = starting_at + decompressed_index - indices_offset;
                    self.values[out_index] = values[compressed_index];
                    self.nulls.set_bit(out_index, validity.value(out_index));
                }
            }
        }

        Ok(())
    }

    pub fn finish_into_primitive(&mut self) -> VortexResult<PrimitiveArray> {
        let validity = match (self.nulls.finish(), self.dtype().nullability()) {
            (None, Nullability::NonNullable) => Validity::NonNullable,
            (Some(_), Nullability::NonNullable) => {
                vortex_bail!("Non-nullable builder has null values")
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

        Ok(PrimitiveArray::new(
            std::mem::take(&mut self.values).freeze(),
            validity,
        ))
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

    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        let array = array.into_canonical()?.into_primitive()?;
        if array.ptype() != T::PTYPE {
            vortex_bail!("Cannot extend from array with different ptype");
        }

        self.values.extend_from_slice(array.as_slice::<T>());

        self.extend_with_validity_mask(array.validity_mask()?);

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<Array> {
        self.finish_into_primitive().map(IntoArray::into_array)
    }
}
