// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Schema};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_ensure};

use crate::arrays::StructArray;
use crate::arrow::compute::{to_arrow, to_arrow_preferred};
use crate::{Array, Canonical};

impl TryFrom<&dyn Array> for RecordBatch {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> VortexResult<Self> {
        let Canonical::Struct(struct_array) = value.to_canonical() else {
            vortex_bail!("RecordBatch can only be constructed from ")
        };

        vortex_ensure!(
            struct_array.all_valid(),
            "RecordBatch can only be constructed from StructArray with no nulls"
        );

        let array_ref = to_arrow_preferred(struct_array.as_ref())?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}

impl StructArray {
    pub fn into_record_batch_with_schema(
        self,
        schema: impl AsRef<Schema>,
    ) -> VortexResult<RecordBatch> {
        let data_type = DataType::Struct(schema.as_ref().fields.clone());
        let array_ref = to_arrow(self.as_ref(), &data_type)?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, FieldRef, Schema};
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::arrays::StructArray;
    use crate::builders::{ArrayBuilder, ListBuilder};

    #[test]
    fn test_into_rb_with_schema() {
        let mut xs = ListBuilder::<u32>::new(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        );

        xs.append_scalar(&Scalar::list(
            xs.element_dtype().clone(),
            vec![1i32.into(), 2i32.into(), 3i32.into()],
            Nullability::Nullable,
        ))
        .unwrap();
        xs.append_null();
        xs.append_zero();

        let xs = xs.finish();

        let array = StructArray::from_fields(&[("xs", xs)]).unwrap();

        // Explicitly request a conversion to LargeListView type instead of the preferred type.
        let arrow_schema = Arc::new(Schema::new(vec![Field::new(
            "xs",
            DataType::LargeListView(FieldRef::new(Field::new_list_field(DataType::Int32, false))),
            true,
        )]));
        let rb = array.into_record_batch_with_schema(arrow_schema).unwrap();

        let xs = rb.column(0);
        assert_eq!(
            xs.data_type(),
            &DataType::LargeListView(FieldRef::new(Field::new_list_field(DataType::Int32, false)))
        );
    }
}
