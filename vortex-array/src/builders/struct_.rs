// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use itertools::Itertools;
use vortex_dtype::{DType, Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_ensure, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{Scalar, StructScalar};

use crate::arrays::StructArray;
use crate::builders::{
    ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyBitBufferBuilder, builder_with_capacity,
};
use crate::canonical::{Canonical, ToCanonical};
use crate::{Array, ArrayRef, IntoArray};

/// The builder for building a [`StructArray`].
pub struct StructBuilder {
    dtype: DType,
    builders: Vec<Box<dyn ArrayBuilder>>,
    nulls: LazyBitBufferBuilder,
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
            nulls: LazyBitBufferBuilder::new(capacity),
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

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.builders
            .iter_mut()
            // We push zero values into our children when appending a null in case the children are
            // themselves non-nullable.
            .for_each(|builder| builder.append_defaults(n));
        self.nulls.append_null();
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "StructBuilder expected scalar with dtype {:?}, got {:?}",
            self.dtype(),
            scalar.dtype()
        );

        let struct_scalar = StructScalar::try_from(scalar)?;
        self.append_value(struct_scalar)
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let array = array.to_struct();

        for (a, builder) in array
            .fields()
            .iter()
            .cloned()
            .zip_eq(self.builders.iter_mut())
        {
            a.append_to_builder(builder.as_mut());
        }

        self.nulls.append_validity_mask(array.validity_mask());
    }

    fn reserve_exact(&mut self, capacity: usize) {
        self.builders.iter_mut().for_each(|builder| {
            builder.reserve_exact(capacity);
        });
        self.nulls.reserve_exact(capacity);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_struct().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::Struct(self.finish_into_struct())
    }
}

#[cfg(test)]
mod tests {

    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructFields};
    use vortex_scalar::Scalar;

    use crate::builders::ArrayBuilder;
    use crate::builders::struct_::StructBuilder;

    #[test]
    fn test_struct_builder() {
        let sdt = StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]);
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
        let sdt = StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]);
        let dtype = DType::Struct(sdt.clone(), Nullability::Nullable);
        let mut builder = StructBuilder::with_capacity(sdt, Nullability::Nullable, 0);

        builder
            .append_value(Scalar::struct_(dtype.clone(), vec![1.into(), 2.into()]).as_struct())
            .unwrap();

        let struct_ = builder.finish();
        assert_eq!(struct_.len(), 1);
        assert_eq!(struct_.dtype(), &dtype);
    }

    #[test]
    fn test_append_scalar() {
        use vortex_scalar::Scalar;

        let dtype = DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(I32, Nullability::Nullable)),
                ("b", DType::Utf8(Nullability::Nullable)),
            ]),
            Nullability::Nullable,
        );

        let struct_fields = match &dtype {
            DType::Struct(fields, _) => fields.clone(),
            _ => panic!("Expected struct dtype"),
        };
        let mut builder = StructBuilder::new(struct_fields, Nullability::Nullable);

        // Test appending a valid struct value.
        let struct_scalar1 = Scalar::struct_(
            dtype.clone(),
            vec![
                Scalar::primitive(42i32, Nullability::Nullable),
                Scalar::utf8("hello", Nullability::Nullable),
            ],
        );
        builder.append_scalar(&struct_scalar1).unwrap();

        // Test appending another struct value.
        let struct_scalar2 = Scalar::struct_(
            dtype.clone(),
            vec![
                Scalar::primitive(84i32, Nullability::Nullable),
                Scalar::utf8("world", Nullability::Nullable),
            ],
        );
        builder.append_scalar(&struct_scalar2).unwrap();

        // Test appending null value.
        let null_scalar = Scalar::null(dtype.clone());
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_struct();
        assert_eq!(array.len(), 3);

        // Check actual values using scalar_at.

        let scalar0 = array.scalar_at(0);
        let struct0 = scalar0.as_struct();
        if let Some(fields0) = struct0.fields() {
            let fields0 = fields0.collect::<Vec<_>>();
            assert_eq!(fields0[0].as_primitive().typed_value::<i32>(), Some(42));
            assert_eq!(fields0[1].as_utf8().value().as_deref(), Some("hello"));
        }

        let scalar1 = array.scalar_at(1);
        let struct1 = scalar1.as_struct();
        if let Some(fields1) = struct1.fields() {
            let fields1 = fields1.collect::<Vec<_>>();
            assert_eq!(fields1[0].as_primitive().typed_value::<i32>(), Some(84));
            assert_eq!(fields1[1].as_utf8().value().as_deref(), Some("world"));
        }

        let scalar2 = array.scalar_at(2);
        let struct2 = scalar2.as_struct();
        assert!(struct2.fields().is_none()); // Null struct has no fields.

        // Check validity - first two should be valid, third should be null.
        use crate::vtable::ValidityHelper;
        assert!(array.validity().is_valid(0));
        assert!(array.validity().is_valid(1));
        assert!(!array.validity().is_valid(2));

        // Test wrong dtype error.
        let struct_fields = match &dtype {
            DType::Struct(fields, _) => fields.clone(),
            _ => panic!("Expected struct dtype"),
        };
        let mut builder = StructBuilder::new(struct_fields, Nullability::NonNullable);
        let wrong_scalar = Scalar::from(42i32);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
