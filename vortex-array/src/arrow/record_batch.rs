// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Schema;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::Canonical;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::IntoArray;
use crate::arrays::StructArray;
use crate::arrow::ArrowArrayExecutor;
use crate::validity::Validity;

// deprecated(note = "Use ArrowArrayExecutor::execute_record_batch instead")
impl TryFrom<&ArrayRef> for RecordBatch {
    type Error = VortexError;

    fn try_from(value: &ArrayRef) -> VortexResult<Self> {
        let Canonical::Struct(struct_array) = value.to_canonical()? else {
            vortex_bail!("RecordBatch can only be constructed from ")
        };

        vortex_ensure!(
            matches!(struct_array.validity()?, Validity::AllValid),
            "RecordBatch can only be constructed from StructArray with no nulls"
        );

        let data_type = struct_array.dtype().to_arrow_dtype()?;
        let array_ref = struct_array
            .into_array()
            .execute_arrow(Some(&data_type), &mut LEGACY_SESSION.create_execution_ctx())?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}

impl StructArray {
    pub fn into_record_batch_with_schema(
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
        let rb = array.into_record_batch_with_schema(arrow_schema).unwrap();

        let xs = rb.column(0);
        assert_eq!(
            xs.data_type(),
            &DataType::LargeListView(FieldRef::new(Field::new_list_field(DataType::Int32, false)))
        );
    }
}
