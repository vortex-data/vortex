use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field, FieldRef};
use vortex_dtype::PType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{ListArray, ListEncoding};
use crate::arrow::IntoArrowArray;
use crate::compute::{ToArrowFn, cast};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ToCanonical};

impl ToArrowFn<&ListArray> for ListEncoding {
    fn to_arrow(&self, array: &ListArray, data_type: &DataType) -> VortexResult<Option<ArrayRef>> {
        let (cast_ptype, element_dtype) = match data_type {
            DataType::List(field) => (PType::I32, field.data_type()),
            DataType::LargeList(field) => (PType::I64, field.data_type()),
            _ => {
                vortex_bail!("Unsupported data type: {data_type}");
            }
        };

        let offsets = array
            .offsets()
            .to_primitive()
            .map_err(|err| err.with_context("Failed to canonicalize offsets"))?;

        let arrow_offsets = cast(&offsets, cast_ptype.into())
            .map_err(|err| err.with_context("Failed to cast offsets to PrimitiveArray"))?
            .to_primitive()?;

        let values = array.elements().clone().into_arrow(element_dtype)?;

        let field_ref = FieldRef::new(Field::new_list_field(
            values.data_type().clone(),
            array.elements().dtype().nullability().into(),
        ));

        let nulls = array.validity_mask()?.to_null_buffer();

        Ok(Some(match arrow_offsets.ptype() {
            PType::I32 => Arc::new(arrow_array::ListArray::try_new(
                field_ref,
                arrow_offsets.buffer::<i32>().into_arrow_offset_buffer(),
                values,
                nulls,
            )?),
            PType::I64 => Arc::new(arrow_array::LargeListArray::try_new(
                field_ref,
                arrow_offsets.buffer::<i64>().into_arrow_offset_buffer(),
                values,
                nulls,
            )?),
            _ => vortex_bail!("Invalid offsets type {}", arrow_offsets.ptype()),
        }))
    }
}
