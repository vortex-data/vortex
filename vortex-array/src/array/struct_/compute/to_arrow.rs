use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray as ArrowStructArray};
use arrow_schema::{DataType, Field, Fields};
use itertools::Itertools;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{StructArray, StructEncoding};
use crate::compute::{to_arrow, ToArrowFn};
use crate::validity::ArrayValidity;
use crate::variants::StructArrayTrait;
use crate::{ArrayLen, IntoCanonical};

impl ToArrowFn<StructArray> for StructEncoding {
    fn to_arrow(
        &self,
        array: &StructArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrayRef>> {
        let target_fields = match data_type {
            DataType::Struct(fields) => fields,
            _ => vortex_bail!("Unsupported data type: {data_type}"),
        };

        let field_arrays = target_fields
            .iter()
            .zip_eq(array.children())
            .map(|(field, arr)| {
                to_arrow(arr, field.data_type()).map_err(|err| {
                    err.with_context(format!("Failed to canonicalize field {}", field))
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let nulls = array.logical_validity()?.to_null_buffer();

        if field_arrays.is_empty() {
            Ok(Some(Arc::new(ArrowStructArray::new_empty_fields(
                array.len(),
                nulls,
            ))))
        } else {
            let arrow_fields = array
                .names()
                .iter()
                .zip(field_arrays.iter())
                .zip(array.dtypes().iter())
                .map(|((name, arrow_field), vortex_field)| {
                    Field::new(
                        &**name,
                        arrow_field.data_type().clone(),
                        vortex_field.is_nullable(),
                    )
                })
                .map(Arc::new)
                .collect::<Fields>();

            Ok(Some(Arc::new(ArrowStructArray::try_new(
                arrow_fields,
                field_arrays,
                nulls,
            )?)))
        }
    }
}
