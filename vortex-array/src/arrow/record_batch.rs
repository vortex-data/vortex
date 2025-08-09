// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Schema};
use vortex_dtype::arrow::FromArrowType;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult, vortex_err};

use crate::arrays::StructArray;
use crate::arrow::compute::{to_arrow, to_arrow_preferred};
use crate::compute::cast;
use crate::{Array, ToCanonical};

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
        let array_ref = to_arrow_preferred(self.as_ref())?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }

    pub fn into_record_batch_with_schema(self, schema: &Schema) -> VortexResult<RecordBatch> {
        let data_type = DataType::Struct(schema.fields.clone());
        let dtype = DType::from_arrow((&data_type, Nullability::NonNullable));
        let casted = cast(self.as_ref(), &dtype)?;
        let array_ref = to_arrow(casted.as_ref(), &data_type)?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}
