use std::any::Any;

use itertools::Itertools;
use vortex_dtype::{DType, Nullability, StructDType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::StructScalar;

use crate::array::StructArray;
use crate::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt, BoolBuilder};
use crate::validity::Validity;
use crate::{ArrayData, IntoArrayData};

pub struct StructBuilder {
    builders: Vec<Box<dyn ArrayBuilder>>,
    validity: BoolBuilder,
    struct_dtype: StructDType,
    nullability: Nullability,
    dtype: DType,
}

impl StructBuilder {
    pub fn with_capacity(
        struct_dtype: StructDType,
        nullability: Nullability,
        capacity: usize,
    ) -> Self {
        let builders = struct_dtype
            .dtypes()
            .iter()
            .map(|dt| builder_with_capacity(dt, capacity))
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
            for (builder, field) in self.builders.iter_mut().zip(fields) {
                builder.append_scalar(&field)?;
            }
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

    fn finish(&mut self) -> VortexResult<ArrayData> {
        let fields: Vec<ArrayData> = self
            .builders
            .iter_mut()
            .map(|builder| builder.finish())
            .try_collect()?;

        let validity = match self.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::Array(self.validity.finish()?),
        };

        Ok(StructArray::try_new(
            self.struct_dtype.names().clone(),
            fields,
            self.len(),
            validity,
        )?
        .into_array())
    }
}
