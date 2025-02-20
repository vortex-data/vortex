use arrow_array::cast::AsArray;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::arrays::StructArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, TryIntoArray};

impl TryIntoArray for RecordBatch {
    fn try_into_array(self) -> VortexResult<ArrayRef> {
        Ok(StructArray::try_new(
            self.schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str().into())
                .collect(),
            self.columns()
                .iter()
                .zip(self.schema().fields())
                .map(|(array, field)| ArrayRef::from_arrow(array.clone(), field.is_nullable()))
                .collect(),
            self.num_rows(),
            Validity::NonNullable, // Must match FromArrowType<SchemaRef> for DType
        )?
        .into_array())
    }
}

impl TryFrom<&dyn Array> for RecordBatch {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> VortexResult<Self> {
        let struct_arr = value.to_struct().map_err(|err| {
            vortex_err!("RecordBatch can only be constructed from a Vortex StructArray: {err}")
        })?;

        struct_arr.into_record_batch()
    }
}

impl StructArray {
    pub fn into_record_batch(self) -> VortexResult<RecordBatch> {
        let array_ref = self.into_array().into_arrow_preferred()?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }

    pub fn into_record_batch_with_schema(self, schema: &Schema) -> VortexResult<RecordBatch> {
        let data_type = DataType::Struct(schema.fields.clone());
        let array_ref = self.into_array().into_arrow(&data_type)?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}
