use std::any::Any;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::{DType, Nullability, StructDType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::StructScalar;

use crate::arrays::StructArray;
use crate::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt, BoolBuilder};
use crate::validity::Validity;
use crate::variants::StructArrayTrait;
use crate::{Array, ArrayRef, Canonical};

pub struct StructBuilder {
    builders: Vec<Box<dyn ArrayBuilder>>,
    // TODO(ngates): this should be a NullBufferBuilder? Or mask builder?
    validity: BoolBuilder,
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
            validity: BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
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
            self.validity.append_value(true);
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
        self.validity.append_values(true, n);
    }

    fn append_nulls(&mut self, n: usize) {
        self.builders
            .iter_mut()
            // We push zero values into our children when appending a null in case the children are
            // themselves non-nullable.
            .for_each(|builder| builder.append_zeros(n));
        self.validity.append_value(false);
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        let array = array.to_canonical()?;
        let Canonical::Struct(array) = array else {
            vortex_bail!("Expected Canonical::Struct, found {:?}", array);
        };
        if array.dtype() != self.dtype() {
            vortex_bail!(
                "Cannot extend from array with different dtype: expected {:?}, found {:?}",
                self.dtype(),
                array.dtype()
            );
        }

        for (a, builder) in (0..array.nfields())
            .map(|i| {
                array
                    .maybe_null_field_by_idx(i)
                    .vortex_expect("out of bounds")
            })
            .zip_eq(self.builders.iter_mut())
        {
            a.append_to_builder(builder.as_mut())?;
        }

        match array.validity() {
            Validity::NonNullable | Validity::AllValid => {
                self.validity.append_values(true, array.len());
            }
            Validity::AllInvalid => {
                self.validity.append_values(false, array.len());
            }
            Validity::Array(validity) => {
                validity.append_to_builder(&mut self.validity)?;
            }
        }

        Ok(())
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
                    "Field {} does not have expected length {}",
                    index,
                    expected_length
                );
            }
        }

        let validity = match self.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::Array(self.validity.finish()),
        };

        StructArray::try_new(self.struct_dtype.names().clone(), fields, len, validity)
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

    use crate::builders::struct_::StructBuilder;
    use crate::builders::ArrayBuilder;

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
}
