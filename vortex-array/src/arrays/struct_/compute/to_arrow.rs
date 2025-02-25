use std::sync::Arc;

use arrow_array::{ArrayRef, StructArray as ArrowStructArray};
use arrow_schema::{DataType, Field, Fields};
use itertools::Itertools;
use vortex_error::{VortexResult, vortex_bail};

use crate::Array;
use crate::arrays::{StructArray, StructEncoding};
use crate::compute::{ToArrowFn, to_arrow};
use crate::variants::StructArrayTrait;

impl ToArrowFn<&StructArray> for StructEncoding {
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
            .zip_eq(array.fields())
            .map(|(field, arr)| {
                // We check that the Vortex array nullability is compatible with the field
                // nullability. In other words, make sure we don't return any nulls for a
                // non-nullable field.
                if arr.dtype().is_nullable() && !field.is_nullable() && !arr.all_valid()? {
                    vortex_bail!(
                        "Field {} is non-nullable but has nulls {}",
                        field,
                        arr.tree_display()
                    );
                }

                to_arrow(arr, field.data_type()).map_err(|err| {
                    err.with_context(format!("Failed to canonicalize field {}", field))
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let nulls = array.validity_mask()?.to_null_buffer();

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
                .zip(target_fields.iter())
                .map(|((name, field_array), target_field)| {
                    Field::new(
                        &**name,
                        field_array.data_type().clone(),
                        target_field.is_nullable(),
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

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;

    use super::*;
    use crate::arrays::PrimitiveArray;
    use crate::arrow::IntoArrowArray;
    use crate::validity::Validity;

    #[test]
    fn nullable_non_null_to_arrow() {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::AllValid);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into()]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        struct_a.into_array().into_arrow(&arrow_dt).unwrap();
    }

    #[test]
    fn nullable_with_nulls_to_arrow() {
        let xs =
            PrimitiveArray::from_option_iter(vec![Some(0_i64), Some(1), Some(2), None, Some(3)]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into()]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        assert!(struct_a.into_array().into_arrow(&arrow_dt).is_err());
    }
}
