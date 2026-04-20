// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Schema;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_error::VortexResult;

use crate::ArrowArrayExecutor;

/// Extension trait on [`StructArray`] for constructing an Arrow [`RecordBatch`].
pub trait StructArrayRecordBatchExt {
    /// Convert this [`StructArray`] into an Arrow [`RecordBatch`] with the given schema.
    fn into_record_batch_with_schema(
        self,
        schema: impl AsRef<Schema>,
    ) -> VortexResult<RecordBatch>;
}

impl StructArrayRecordBatchExt for StructArray {
    fn into_record_batch_with_schema(
        self,
        schema: impl AsRef<Schema>,
    ) -> VortexResult<RecordBatch> {
        let data_type = DataType::Struct(schema.as_ref().fields.clone());
        let array_ref = self
            .into_array()
            .execute_arrow(Some(&data_type), &mut LEGACY_SESSION.create_execution_ctx())?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::FieldRef;
    use arrow_schema::Schema;
    use vortex_array::arrays::StructArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::ListBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;

    use super::StructArrayRecordBatchExt;

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
