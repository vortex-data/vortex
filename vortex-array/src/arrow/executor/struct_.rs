// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray as ArrowStructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::Field;
use arrow_schema::Fields;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Chunked;
use crate::arrays::ScalarFnVTable;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayParts;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::StructFields;
use crate::dtype::arrow::FromArrowType;
use crate::scalar_fn::fns::pack::Pack;

pub(super) fn to_arrow_struct(
    array: ArrayRef,
    target_fields: Option<&Fields>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();

    // If the array is chunked, then we invert the chunk-of-struct to struct-of-chunk.
    let array = match array.try_into::<Chunked>() {
        Ok(array) => {
            // NOTE(ngates): this currently uses the old into_canonical code path, but we should
            //  just call directly into the swizzle-chunks function.
            array.into_array().execute::<StructArray>(ctx)?.into_array()
        }
        Err(array) => array,
    };

    // Attempt to short-circuit if the array is already a Struct:
    let array = match array.try_into::<Struct>() {
        Ok(array) => {
            let len = array.len();
            let StructArrayParts {
                validity,
                fields,
                struct_fields,
                ..
            } = array.into_parts();
            let validity = to_arrow_null_buffer(validity, len, ctx)?;
            return create_from_fields(
                target_fields.ok_or_else(|| struct_fields.names().clone()),
                &fields,
                validity,
                len,
                ctx,
            );
        }
        Err(array) => array,
    };

    // We can also short-circuit if the array is a `pack` scalar function:
    if let Some(array) = array.as_opt::<ScalarFnVTable>()
        && let Some(_pack_options) = array.scalar_fn().as_opt::<Pack>()
    {
        let DType::Struct(struct_fields, _) = array.dtype() else {
            unreachable!("Pack must have Struct dtype");
        };
        return create_from_fields(
            target_fields.ok_or_else(|| struct_fields.names().clone()),
            &array.children(),
            None, // Pack is never null,
            len,
            ctx,
        );
    }

    // Otherwise, we fall back to executing to a StructArray.
    let array = if let Some(fields) = target_fields {
        let vx_fields = StructFields::from_arrow(fields);
        // We apply a cast to ensure we push down casting where possible into the struct fields.
        array.cast(DType::Struct(
            vx_fields,
            crate::dtype::Nullability::Nullable,
        ))?
    } else {
        array
    };

    let struct_array = array.execute::<StructArray>(ctx)?;
    let len = struct_array.len();
    let StructArrayParts {
        validity,
        fields,
        struct_fields,
        ..
    } = struct_array.into_parts();

    let validity = to_arrow_null_buffer(validity, len, ctx)?;
    create_from_fields(
        target_fields.ok_or_else(|| struct_fields.names().clone()),
        &fields,
        validity,
        len,
        ctx,
    )
}

fn create_from_fields(
    fields: Result<&Fields, FieldNames>,
    vortex_fields: &[ArrayRef],
    null_buffer: Option<NullBuffer>,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    match fields {
        Ok(fields) => {
            vortex_ensure!(
                vortex_fields.len() == fields.len(),
                "StructArray has {} fields, but target Arrow type has {} fields",
                vortex_fields.len(),
                fields.len()
            );

            let mut arrow_arrays = Vec::with_capacity(vortex_fields.len());
            for (field, vx_field) in fields.iter().zip_eq(vortex_fields.iter()) {
                let arrow_field = vx_field
                    .clone()
                    .execute_arrow(Some(field.data_type()), ctx)?;
                vortex_ensure!(
                    field.is_nullable() || arrow_field.null_count() == 0,
                    "Cannot convert field '{}' to non-nullable Arrow field because it contains nulls",
                    field.name()
                );
                arrow_arrays.push(arrow_field);
            }

            Ok(Arc::new(unsafe {
                ArrowStructArray::new_unchecked_with_length(
                    fields.clone(),
                    arrow_arrays,
                    null_buffer,
                    len,
                )
            }))
        }
        Err(names) => {
            // No target fields specified - use preferred types for each child
            let mut arrow_arrays = Vec::with_capacity(vortex_fields.len());
            for vx_field in vortex_fields.iter() {
                let arrow_array = vx_field.clone().execute_arrow(None, ctx)?;
                arrow_arrays.push(arrow_array);
            }

            // Build the Arrow fields from the resulting arrays
            let arrow_fields: Fields = names
                .iter()
                .zip_eq(arrow_arrays.iter())
                .zip_eq(vortex_fields.iter().map(|f| f.dtype().is_nullable()))
                .map(|((name, arr), vx_nullable)| {
                    Arc::new(Field::new(
                        name.as_ref(),
                        arr.data_type().clone(),
                        vx_nullable,
                    ))
                })
                .collect();

            Ok(Arc::new(unsafe {
                ArrowStructArray::new_unchecked_with_length(
                    arrow_fields,
                    arrow_arrays,
                    null_buffer,
                    len,
                )
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef;
    use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
    use arrow_array::StringViewArray;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::types::Int32Type;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::array;
    use crate::arrays;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::FromArrowArray;
    use crate::arrow::IntoArrowArray;
    use crate::dtype::FieldNames;
    use crate::validity::Validity;

    #[test]
    fn struct_nullable_non_null_to_arrow() -> VortexResult<()> {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::AllValid);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )?;

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        struct_a.into_array().into_arrow(&arrow_dt)?;
        Ok(())
    }

    #[test]
    fn struct_nullable_with_nulls_to_arrow() -> VortexResult<()> {
        let xs =
            PrimitiveArray::from_option_iter(vec![Some(0_i64), Some(1), Some(2), None, Some(3)]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )?;

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        assert!(struct_a.into_array().into_arrow(&arrow_dt).is_err());
        Ok(())
    }

    #[test]
    fn struct_to_arrow_with_schema_mismatch() -> VortexResult<()> {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::AllValid);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )?;

        let fields = vec![
            Field::new("xs", DataType::Int8, false),
            Field::new("ys", DataType::Int64, false),
        ];
        let arrow_dt = DataType::Struct(fields.into());

        let err = struct_a.into_array().into_arrow(&arrow_dt).err().unwrap();
        assert!(
            err.to_string()
                .contains("StructArray has 1 fields, but target Arrow type has 2 fields")
        );
        Ok(())
    }

    #[test]
    fn test_to_arrow() -> VortexResult<()> {
        let array = StructArray::from_fields(
            vec![
                (
                    "a",
                    PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)]).into_array(),
                ),
                (
                    "b",
                    arrays::varbinview::VarBinViewArray::from_iter_str(vec!["a", "b", "c"])
                        .into_array(),
                ),
            ]
            .as_slice(),
        )?;

        let arrow_array: ArrayRef = Arc::new(ArrowStructArray::try_from(vec![
            (
                "a",
                Arc::new(
                    ArrowPrimitiveArray::<Int32Type>::from_iter_values_with_nulls(
                        vec![1, 0, 2],
                        Some(NullBuffer::from(vec![true, false, true])),
                    ),
                ) as ArrayRef,
            ),
            (
                "b",
                Arc::new(StringViewArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ),
        ])?);

        let arrow_dtype = array.dtype().to_arrow_dtype()?;
        assert_eq!(
            &array.into_array().execute_arrow(
                Some(&arrow_dtype),
                &mut LEGACY_SESSION.create_execution_ctx()
            )?,
            &arrow_array
        );
        Ok(())
    }

    #[test]
    fn to_arrow_with_non_nullable_fields() -> VortexResult<()> {
        let array = StructArray::from_fields(
            vec![
                (
                    "a",
                    PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)]).into_array(),
                ),
                (
                    "b",
                    arrays::varbinview::VarBinViewArray::from_iter_str(vec!["a", "b", "c"])
                        .into_array(),
                ),
            ]
            .as_slice(),
        )?;
        let orig_dtype = array.dtype().clone();
        let arrow_array = array.into_array().into_arrow_preferred()?;
        let from_arrow = array::ArrayRef::from_arrow(arrow_array.as_ref(), false)?;
        assert_eq!(&orig_dtype, from_arrow.dtype());
        Ok(())
    }
}
