use std::any::Any;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::{DType, Nullability, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::StructScalar;

use crate::arrays::StructArray;
use crate::builders::lazy_validity_builder::LazyNullBufferBuilder;
use crate::builders::{ArrayBuilder, ArrayBuilderExt, builder_with_capacity};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

pub struct StructBuilder {
    builders: Vec<Box<dyn ArrayBuilder>>,
    validity: LazyNullBufferBuilder,
    struct_dtype: Arc<StructDType>,
    nullability: Nullability,
    dtype: DType,
}

impl StructBuilder {
    pub fn with_capacity(
        struct_dtype: Arc<StructDType>,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        let builders = struct_dtype
            .fields()
            .map(|dt| builder_with_capacity(&dt, capacity))
            .collect();

        Self {
            builders,
            validity: LazyNullBufferBuilder::new(capacity),
            struct_dtype: struct_dtype.clone(),
            nullability,
            dtype: DType::Struct(struct_dtype, nullability),
        }
    }

    pub fn append_value(&mut self, struct_scalar: StructScalar) -> VortexResult<()> {
        if struct_scalar.dtype() != &DType::Struct(self.struct_dtype.clone(), self.nullability) {
            vortex_bail!(
                "Expected struct scalar with dtype {:?}, found {:?}",
                self.struct_dtype,
                struct_scalar.dtype()
            )
        }

        if let Some(fields) = struct_scalar.fields() {
            for (builder, field) in self.builders.iter_mut().zip_eq(fields) {
                builder.append_scalar(&field)?;
            }
            self.validity.append_non_null();
        } else {
            self.append_null()
        }

        Ok(())
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
        self.validity.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.builders
            .iter_mut()
            .for_each(|builder| builder.append_zeros(n));
        self.validity.append_n_non_nulls(n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.builders
            .iter_mut()
            // We push zero values into our children when appending a null in case the children are
            // themselves non-nullable.
            .for_each(|builder| builder.append_zeros(n));
        self.validity.append_null();
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        let array = array.to_struct()?;

        if array.dtype() != self.dtype() {
            vortex_bail!(
                "Cannot extend from array with different dtype: expected {}, found {}",
                self.dtype(),
                array.dtype()
            );
        }

        for (a, builder) in (0..array.struct_dtype().nfields())
            .map(|i| &array.fields()[i])
            .zip_eq(self.builders.iter_mut())
        {
            a.append_to_builder(builder.as_mut())?;
        }

        self.validity.append_validity_mask(array.validity_mask()?);
        Ok(())
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        self.builders.iter_mut().for_each(|builder| {
            builder.ensure_capacity(capacity);
        });
        self.validity.ensure_capacity(capacity);
    }

    fn set_validity(&mut self, validity: Mask) {
        self.validity = LazyNullBufferBuilder::new(validity.len());
        self.validity.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
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

        let validity = self.validity.finish_with_nullability(self.nullability);

        StructArray::try_new_with_dtype(fields, self.struct_dtype.clone(), len, validity)
            .vortex_expect("Fields must all have same length.")
            .into_array()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructDType};
    use vortex_scalar::Scalar;

    use crate::builders::ArrayBuilder;
    use crate::builders::struct_::StructBuilder;

    #[test]
    fn test_struct_builder() {
        let sdt = Arc::new(StructDType::new(
            vec![Arc::from("a"), Arc::from("b")].into(),
            vec![I32.into(), I32.into()],
        ));
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
        let sdt = Arc::new(StructDType::new(
            vec![Arc::from("a"), Arc::from("b")].into(),
            vec![I32.into(), I32.into()],
        ));
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
