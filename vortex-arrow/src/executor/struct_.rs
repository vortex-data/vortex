// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray as ArrowStructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::Field;
use arrow_schema::Fields;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::struct_::StructDataParts;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::matcher::Matcher;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrowArrayExecutor;
use crate::executor::validity::to_arrow_null_buffer;
use crate::session::ArrowSessionExt;

/// Matches the encodings [`to_arrow_struct`] requires for export.
struct ArrowStructExportable;

impl Matcher for ArrowStructExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Struct>()
            || array.is::<Chunked>()
            || array
                .as_opt::<ScalarFn>()
                .is_some_and(|scalar_fn| scalar_fn.scalar_fn().as_opt::<Pack>().is_some()))
        .then_some(array)
    }
}

pub(super) fn to_arrow_struct(
    array: ArrayRef,
    target_fields: Option<&Fields>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();

    let array = array.execute_until::<ArrowStructExportable>(ctx)?;

    // If the array is chunked, then we invert the chunk-of-struct to struct-of-chunk.
    let array = match array.try_downcast::<Chunked>() {
        Ok(array) => {
            // NOTE(ngates): this currently uses the old into_canonical code path, but we should
            //  just call directly into the swizzle-chunks function.
            array.into_array().execute::<StructArray>(ctx)?.into_array()
        }
        Err(array) => array,
    };

    // Attempt to short-circuit if the array is already a Struct:
    let array = match array.try_downcast::<Struct>() {
        Ok(array) => {
            let StructDataParts {
                validity,
                fields,
                struct_fields,
                ..
            } = array.into_data_parts();
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
    if let Some(array) = array.as_opt::<ScalarFn>()
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
        let vx_fields = ctx.session().arrow().from_arrow_fields(fields)?;
        // We apply a cast to ensure we push down casting where possible into the struct fields.
        array.cast(DType::Struct(
            vx_fields,
            vortex_array::dtype::Nullability::Nullable,
        ))?
    } else {
        array
    };

    let struct_array = array.execute::<StructArray>(ctx)?;
    let StructDataParts {
        validity,
        fields,
        struct_fields,
        ..
    } = struct_array.into_data_parts();

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
                // Route through the session with the full Field (not just data_type) so any
                // ARROW:extension:name metadata reaches the export-plugin dispatcher.
                let arrow_field = ctx.session().clone().arrow().execute_arrow(
                    vx_field.clone(),
                    Some(field.as_ref()),
                    ctx,
                )?;
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

    use arrays::varbinview::VarBinViewArray;
    use arrow_array::Array;
    use arrow_array::ArrayRef;
    use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
    use arrow_array::StringViewArray;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::types::Int32Type;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_array as array;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::dtype::FieldNames;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::fns::mask::Mask;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrowArrayExecutor;
    use crate::FromArrowArray;
    use crate::dtype::to_data_type_naive;

    #[test]
    fn struct_nullable_non_null_to_arrow() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::AllValid);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            5,
            Validity::AllValid,
        )?;

        let fields = vec![Field::new("xs", DataType::Int64, false)];
        let arrow_dt = DataType::Struct(fields.into());

        struct_a
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;
        Ok(())
    }

    #[test]
    fn struct_nullable_with_nulls_to_arrow() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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

        assert!(
            struct_a
                .into_array()
                .execute_arrow(Some(&arrow_dt), &mut ctx)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn struct_to_arrow_with_schema_mismatch() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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

        let err = struct_a
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)
            .err()
            .unwrap();
        assert!(
            err.to_string()
                .contains("StructArray has 1 fields, but target Arrow type has 2 fields")
        );
        Ok(())
    }

    #[test]
    fn test_to_arrow() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = StructArray::from_fields(
            vec![
                (
                    "a",
                    PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)]).into_array(),
                ),
                (
                    "b",
                    VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array(),
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

        let arrow_dtype = to_data_type_naive(array.dtype())?;
        assert_eq!(
            &array
                .into_array()
                .execute_arrow(Some(&arrow_dtype), &mut ctx)?,
            &arrow_array
        );
        Ok(())
    }

    #[test]
    fn mask_wrapped_struct_exports_via_struct_fast_path() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // A struct behind a lazy `mask` scalar-fn nulling out row 1 — the shape a scan
        // produces when a row mask is applied to a top-level struct batch.
        let xs = PrimitiveArray::new(buffer![1i64, 2, 3], Validity::NonNullable);
        let struct_array = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            3,
            Validity::NonNullable,
        )?;
        let mask = BoolArray::from_iter([true, false, true]);
        let masked = Mask.try_new_array(
            3,
            EmptyOptions,
            [struct_array.into_array(), mask.into_array()],
        )?;

        let arrow = masked.execute_arrow(None, &mut ctx)?;

        let arrow_struct = arrow
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .expect("struct array");
        assert_eq!(arrow_struct.len(), 3);
        assert!(!arrow_struct.is_null(0));
        assert!(arrow_struct.is_null(1));
        assert!(!arrow_struct.is_null(2));
        let xs_col = arrow_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<arrow_array::types::Int64Type>>()
            .expect("int64 column");
        assert_eq!(xs_col.values(), &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn to_arrow_with_non_nullable_fields() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = StructArray::from_fields(
            vec![
                (
                    "a",
                    PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2)]).into_array(),
                ),
                (
                    "b",
                    VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array(),
                ),
            ]
            .as_slice(),
        )?;
        let orig_dtype = array.dtype().clone();
        let arrow_array = array.into_array().execute_arrow(None, &mut ctx)?;
        let from_arrow = array::ArrayRef::from_arrow(arrow_array.as_ref(), false)?;
        assert_eq!(&orig_dtype, from_arrow.dtype());
        Ok(())
    }
}
