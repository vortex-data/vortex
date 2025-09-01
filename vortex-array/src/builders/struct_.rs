// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use itertools::Itertools;
use vortex_dtype::{DType, Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::StructScalar;

use crate::arrays::StructArray;
use crate::builders::{
    ArrayBuilder, ArrayBuilderExt, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder,
    builder_with_capacity,
};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// The builder for building a [`StructArray`].
pub struct StructBuilder {
    dtype: DType,
    builders: Vec<Box<dyn ArrayBuilder>>,
    nulls: LazyNullBufferBuilder,
}

impl StructBuilder {
    /// Creates a new `StructBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(struct_dtype: StructFields, nullability: Nullability) -> Self {
        Self::with_capacity(struct_dtype, nullability, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a new `StructBuilder` with the given `capacity`.
    pub fn with_capacity(
        struct_dtype: StructFields,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        let builders = struct_dtype
            .fields()
            .map(|dt| builder_with_capacity(&dt, capacity))
            .collect();

        Self {
            builders,
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Struct(struct_dtype, nullability),
        }
    }

    /// Appends a struct `value` to the builder.
    pub fn append_value(&mut self, struct_scalar: StructScalar) -> VortexResult<()> {
        if !self.dtype.is_nullable() && struct_scalar.is_null() {
            vortex_bail!("Tried to append a null `StructScalar` to a non-nullable struct builder",);
        }

        if struct_scalar.struct_fields() != self.struct_fields() {
            vortex_bail!(
                "Tried to append a `StructScalar` with fields {} to a \
                    struct builder with fields {}",
                struct_scalar.struct_fields(),
                self.struct_fields()
            );
        }

        if let Some(fields) = struct_scalar.fields() {
            for (builder, field) in self.builders.iter_mut().zip_eq(fields) {
                builder.append_scalar(&field)?;
            }
            self.nulls.append_non_null();
        } else {
            self.append_null()
        }

        Ok(())
    }

    /// Finishes the builder directly into a [`StructArray`].
    pub fn finish_into_struct(&mut self) -> StructArray {
        let len = self.len();
        let fields = self
            .builders
            .iter_mut()
            .map(|builder| builder.finish())
            .collect::<Vec<_>>();

        if fields.len() > 1 {
            let expected_length = fields[0].len();
            for (index, field) in fields[1..].iter().enumerate() {
                assert_eq!(
                    field.len(),
                    expected_length,
                    "Field {index} does not have expected length {expected_length}"
                );
            }
        }

        let validity = self.nulls.finish_with_nullability(self.dtype.nullability());

        StructArray::try_new_with_dtype(fields, self.struct_fields().clone(), len, validity)
            .vortex_expect("Fields must all have same length.")
    }

    /// The [`StructFields`] of this struct builder.
    pub fn struct_fields(&self) -> &StructFields {
        let DType::Struct(struct_fields, _) = &self.dtype else {
            vortex_panic!("`StructBuilder` somehow had dtype {}", self.dtype);
        };

        struct_fields
    }
}

impl ArrayBuilder for StructBuilder {
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
        self.nulls.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.builders
            .iter_mut()
            .for_each(|builder| builder.append_zeros(n));
        self.nulls.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.builders
            .iter_mut()
            // We push zero values into our children when appending a null in case the children are
            // themselves non-nullable.
            .for_each(|builder| builder.append_zeros(n));
        self.nulls.append_null();
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        if !self.dtype.eq_with_nullability_superset(array.dtype()) {
            vortex_bail!(
                "tried to extend a builder with `DType` {} with an array with `DType {}",
                self.dtype,
                array.dtype()
            );
        }

        let array = array.to_struct()?;

        for (a, builder) in (0..array.struct_fields().nfields())
            .map(|i| &array.fields()[i])
            .zip_eq(self.builders.iter_mut())
        {
            a.append_to_builder(builder.as_mut())?;
        }

        self.nulls.append_validity_mask(array.validity_mask());
        Ok(())
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.builders.iter_mut().for_each(|builder| {
            builder.ensure_capacity(capacity);
        });
        self.nulls.ensure_capacity(capacity);
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_struct().into_array()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructFields};
    use vortex_scalar::Scalar;

    use crate::builders::ArrayBuilder;
    use crate::builders::struct_::StructBuilder;

    #[test]
    fn test_struct_builder() {
        let sdt = StructFields::new(
            vec![Arc::from("a"), Arc::from("b")].into(),
            vec![I32.into(), I32.into()],
        );
        let dtype = DType::Struct(sdt.clone(), Nullability::NonNullable);
        let mut builder = StructBuilder::with_capacity(sdt, Nullability::NonNullable, 0);

        builder
            .append_value(Scalar::struct_(dtype.clone(), vec![1.into(), 2.into()]).as_struct())
            .unwrap();

        let struct_ = builder.finish();
        assert_eq!(struct_.len(), 1);
        assert_eq!(struct_.dtype(), &dtype);
    }

    #[test]
    fn test_append_nullable_struct() {
        let sdt = StructFields::new(
            vec![Arc::from("a"), Arc::from("b")].into(),
            vec![I32.into(), I32.into()],
        );
        let dtype = DType::Struct(sdt.clone(), Nullability::Nullable);
        let mut builder = StructBuilder::with_capacity(sdt, Nullability::Nullable, 0);

        builder
            .append_value(Scalar::struct_(dtype.clone(), vec![1.into(), 2.into()]).as_struct())
            .unwrap();

        let struct_ = builder.finish();
        assert_eq!(struct_.len(), 1);
        assert_eq!(struct_.dtype(), &dtype);
    }
}
