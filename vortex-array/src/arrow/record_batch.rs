// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Schema;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::IntoArray;
use crate::arrays::StructArray;
use crate::arrow::ArrowArrayExecutor;

impl StructArray {
    /// Convert a [`StructArray`] to a [`RecordBatch`] with the given schema, using `session`.
    pub fn into_record_batch_with_schema_with_session(
        self,
        schema: impl AsRef<Schema>,
        session: &VortexSession,
    ) -> VortexResult<RecordBatch> {
        let data_type = DataType::Struct(schema.as_ref().fields.clone());
        let array_ref = self
            .into_array()
            .execute_arrow(Some(&data_type), &mut session.create_execution_ctx())?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }

    /// Convert a [`StructArray`] to a [`RecordBatch`] using the legacy global session.
    #[deprecated(note = "Use `into_record_batch_with_schema_with_session` instead")]
    pub fn into_record_batch_with_schema(
        self,
        schema: impl AsRef<Schema>,
    ) -> VortexResult<RecordBatch> {
        self.into_record_batch_with_schema_with_session(schema, &LEGACY_SESSION)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::FieldRef;
    use arrow_schema::Schema;

    use crate::LEGACY_SESSION;
    use crate::arrow::record_batch::StructArray;
    use crate::builders::ArrayBuilder;
    use crate::builders::ListBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

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
        let rb = array
            .into_record_batch_with_schema_with_session(arrow_schema, &LEGACY_SESSION)
            .unwrap();

        let xs = rb.column(0);
        assert_eq!(
            xs.data_type(),
            &DataType::LargeListView(FieldRef::new(Field::new_list_field(DataType::Int32, false)))
        );
    }
}
