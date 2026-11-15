// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Export run-end arrays without expanding their logical values.
//!
//! Vortex slices retain absolute run ends. Adjusting before narrowing to Arrow's requested integer
//! width keeps small slices exportable even when their source indices are large. The values field
//! retains extension metadata so imports recover the same logical type.

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RunArray;
use arrow_array::cast::AsArray;
use arrow_array::make_array;
use arrow_array::new_null_array;
use arrow_array::types::*;
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::RunEndArray;
use vortex_array::arrays::runend::RunEndArrayExt;
use vortex_array::arrays::runend::RunEndArraySlotsExt;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::matcher::Matcher;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrowExportOptions;
use crate::ExecuteArrowExt;
use crate::executor::execute_arrow_naive;

/// Matches the encodings [`to_arrow_run_end`] requires for export.
struct ArrowRunEndExportable;

impl Matcher for ArrowRunEndExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<RunEnd>() || array.is::<Constant>()).then_some(array)
    }
}

pub(super) fn to_arrow_run_end(
    array: ArrayRef,
    ends_type: &DataType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let data_type = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", ends_type.clone(), false)),
        Arc::new(values_field.clone()),
    );
    if array.is_empty() {
        return Ok(new_null_array(&data_type, 0));
    }

    // Arrow's RunArray constructors synthesize a values field without extension metadata.
    // Retain the caller's field so the array remains consistent with its export schema.
    let exported = build_run_end_array(array, ends_type, values_field, options, ctx)?;

    Ok(make_array(
        exported
            .to_data()
            .into_builder()
            .data_type(data_type)
            .build()?,
    ))
}

fn build_run_end_array(
    array: ArrayRef,
    ends_type: &DataType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = array.execute_until::<ArrowRunEndExportable>(ctx)?;

    let array = match array.try_downcast::<Constant>() {
        Ok(constant) => {
            return constant_to_run_end(constant, ends_type, values_field, options, ctx);
        }
        Err(array) => array,
    };
    let array = match array.try_downcast::<RunEnd>() {
        Ok(run_end) => return run_end_to_arrow(run_end, ends_type, values_field, options, ctx),
        Err(array) => array,
    };

    // Fallback: canonicalize to flat Arrow, then cast to REE.
    let flat = array.execute_arrow_with_options(Some(values_field), options, ctx)?;
    let ree_type = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", ends_type.clone(), false)),
        Arc::new(values_field.clone()),
    );
    arrow_cast::cast(&flat, &ree_type).map_err(VortexError::from)
}

fn run_end_to_arrow(
    array: RunEndArray,
    ends_type: &DataType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let (runs, ends) = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        let run_ends = ends.as_slice::<E>();
        let length = E::try_from(array.len())?;

        if array.offset() == 0
            && run_ends.first().is_some_and(|&end| end > 0)
            && run_ends.last() == Some(&length)
        {
            (0..run_ends.len(), ends.clone().into_array())
        } else {
            let offset = E::try_from(array.offset())?;

            // A run ending at the offset contributes no rows. Retaining it would give Arrow a
            // zero end, and retaining trailing runs would give duplicate ends after clamping.
            let start = run_ends.partition_point(|&end| end <= offset);
            let end = start + run_ends[start..].partition_point(|&end| end - offset < length);
            let end = (end + 1).min(run_ends.len());
            let adjusted = PrimitiveArray::from_iter(
                run_ends[start..end]
                    .iter()
                    .map(|&end| (end - offset).min(length)),
            )
            .into_array();

            (start..end, adjusted)
        }
    });
    let values = if runs.start == 0 && runs.end == array.values().len() {
        array.values().clone()
    } else {
        array.values().slice(runs)?
    };

    // Narrow only the relative ends: absolute indices can exceed the target's range even when
    // this slice fits. Unadjusted ends and values keep the existing buffer-sharing export path.
    let arrow_ends = execute_arrow_naive(ends, Some(ends_type), options, ctx)?;
    let arrow_values = values.execute_arrow_with_options(Some(values_field), options, ctx)?;

    match ends_type {
        DataType::Int16 => Ok(Arc::new(RunArray::<Int16Type>::try_new(
            arrow_ends.as_primitive::<Int16Type>(),
            &arrow_values,
        )?)),
        DataType::Int32 => Ok(Arc::new(RunArray::<Int32Type>::try_new(
            arrow_ends.as_primitive::<Int32Type>(),
            &arrow_values,
        )?)),
        DataType::Int64 => Ok(Arc::new(RunArray::<Int64Type>::try_new(
            arrow_ends.as_primitive::<Int64Type>(),
            &arrow_values,
        )?)),
        _ => vortex_bail!("Unsupported run-end index type: {:?}", ends_type),
    }
}

/// Convert a constant array to a run-end encoded array with a single run.
fn constant_to_run_end(
    array: ConstantArray,
    ends_type: &DataType,
    values_field: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();
    let scalar = array.scalar();

    if scalar.is_null() {
        let ree_type = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ends_type.clone(), false)),
            Arc::new(values_field.clone()),
        );
        return Ok(new_null_array(&ree_type, len));
    }

    let values = ConstantArray::new(scalar.clone(), 1).execute_arrow_with_options(
        Some(values_field),
        options,
        ctx,
    )?;

    match ends_type {
        DataType::Int16 => build_constant_run_array::<Int16Type>(len, &values),
        DataType::Int32 => build_constant_run_array::<Int32Type>(len, &values),
        DataType::Int64 => build_constant_run_array::<Int64Type>(len, &values),
        _ => vortex_bail!("Unsupported run-end index type: {:?}", ends_type),
    }
}

fn build_constant_run_array<R: RunEndIndexType>(
    len: usize,
    values: &ArrowArrayRef,
) -> VortexResult<ArrowArrayRef> {
    let end = R::Native::from_usize(len)
        .ok_or_else(|| vortex_err!("Array length {len} exceeds run-end index capacity"))?;
    let run_ends = arrow_array::PrimitiveArray::<R>::from_value(end, 1);
    Ok(Arc::new(RunArray::<R>::try_new(&run_ends, values)?) as ArrowArrayRef)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Int16Array;
    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_array::RunArray;
    use arrow_array::types::Int16Type;
    use arrow_array::types::Int32Type;
    use arrow_array::types::Int64Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrowSessionExt;
    use crate::ExecuteArrowExt;
    use crate::executor::run_end::ConstantArray;
    use crate::executor::run_end::RunEnd;

    #[track_caller]
    fn check_temporal_run_end_round_trip(len: usize, sliced: bool) -> VortexResult<()> {
        let session = VortexSession::empty();
        let arrow = session.arrow();
        let mut ctx = session.create_execution_ctx();
        let dtype = Date::new(TimeUnit::Days, Nullability::NonNullable).erased();
        let values = ExtensionArray::try_new(
            dtype,
            PrimitiveArray::from_iter((0..i32::try_from(len)?).map(|i| 7 + i * 4)).into_array(),
        )?
        .into_array();
        let source = RunEnd::try_new(
            PrimitiveArray::from_iter((1..=u32::try_from(len)?).map(|i| i * 2)).into_array(),
            values.clone(),
            &mut ctx,
        )?
        .into_array();
        let source = if sliced && !source.is_empty() {
            source.slice(1..source.len())?
        } else {
            source
        };

        let start = u32::from(sliced && len > 0);
        let expected = values.take(
            PrimitiveArray::from_iter((start..u32::try_from(len * 2)?).map(|i| i / 2)).into_array(),
        )?;
        let values_field = arrow.to_arrow_field("dates", source.dtype())?;
        let field = Field::new(
            "dates",
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                Arc::new(values_field),
            ),
            false,
        );
        let exported = source.execute_arrow(Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());
        let imported = arrow.from_arrow_array(exported, &field)?;
        vortex_array::assert_arrays_eq!(imported, expected, &mut ctx);

        Ok(())
    }

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn ree_type(ends: DataType, values_dtype: DataType) -> DataType {
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ends, false)),
            Arc::new(Field::new("values", values_dtype, true)),
        )
    }

    fn execute(array: ArrayRef, dt: &DataType) -> VortexResult<arrow_array::ArrayRef> {
        let field = Field::new("", dt.clone(), array.dtype().is_nullable());
        array.execute_arrow(Some(&field), &mut SESSION.create_execution_ctx())
    }

    fn constant_i32_with_i16_ends() -> arrow_array::ArrayRef {
        Arc::new(
            RunArray::<Int16Type>::try_new(
                &Int16Array::from(vec![5i16]),
                &Int32Array::from(vec![42]),
            )
            .expect("valid run-end test array"),
        ) as arrow_array::ArrayRef
    }

    fn constant_f64_with_i64_ends() -> arrow_array::ArrayRef {
        Arc::new(
            RunArray::<Int64Type>::try_new(
                &Int64Array::from(vec![7i64]),
                &arrow_array::Float64Array::from(vec![1.5]),
            )
            .expect("valid run-end test array"),
        ) as arrow_array::ArrayRef
    }

    #[rstest]
    #[case::i32_with_i16_ends(
        ConstantArray::new(Scalar::from(42i32), 5).into_array(),
        ree_type(DataType::Int16, DataType::Int32),
        constant_i32_with_i16_ends(),
    )]
    #[case::f64_with_i64_ends(
        ConstantArray::new(Scalar::from(1.5f64), 7).into_array(),
        ree_type(DataType::Int64, DataType::Float64),
        constant_f64_with_i64_ends(),
    )]
    #[case::null(
        ConstantArray::new(Scalar::null(DType::Primitive(PType::I32, Nullable)), 4).into_array(),
        ree_type(DataType::Int32, DataType::Int32),
        arrow_array::new_null_array(
            &ree_type(DataType::Int32, DataType::Int32),
            4,
        ),
    )]
    #[case::empty(
        ConstantArray::new(Scalar::from(42i32), 0).into_array(),
        ree_type(DataType::Int32, DataType::Int32),
        arrow_array::new_null_array(
            &ree_type(DataType::Int32, DataType::Int32),
            0,
        ),
    )]
    fn constant_to_ree(
        #[case] input: ArrayRef,
        #[case] target_type: DataType,
        #[case] expected: arrow_array::ArrayRef,
    ) -> VortexResult<()> {
        let result = execute(input, &target_type)?;
        assert_eq!(result.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn nested_wrappers_reveal_run_end_fast_path() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Runs [7, 7, 8, 8] behind two nested lazy Slice wrappers. A single execution step
        // only merges the slices, so the exporter must keep stepping to reveal the
        // RunEndArray underneath.
        let ends = PrimitiveArray::new(buffer![2u32, 4], Validity::NonNullable);
        let values = PrimitiveArray::new(buffer![7i64, 8], Validity::NonNullable);
        let values_ptr = values.as_slice::<i64>().as_ptr();
        let ree = RunEnd::try_new(ends.into_array(), values.into_array(), &mut ctx)?;
        let wrapped = SliceArray::new(SliceArray::new(ree.into_array(), 0..4).into_array(), 0..4)
            .into_array();

        let result = execute(wrapped, &ree_type(DataType::Int32, DataType::Int64))?;

        let ree_arrow = result
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .ok_or_else(|| vortex_err!("expected Int32 run-end array"))?;
        assert_eq!(ree_arrow.run_ends().values(), &[2, 4]);

        // The RunEnd fast path exports the values buffer zero-copy; the flat-then-recode
        // fallback would materialize new buffers.
        let arrow_values = ree_arrow
            .values()
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| vortex_err!("expected Int64 values"))?;
        assert_eq!(arrow_values.values().as_ptr(), values_ptr);
        let expected = RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![2, 4]),
            &Int64Array::from(vec![7, 8]),
        )?;
        assert_eq!(ree_arrow, &expected);
        Ok(())
    }

    #[test]
    fn primitive_to_ree() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(vec![10i32, 10, 20, 20, 20]).into_array();
        let target = ree_type(DataType::Int32, DataType::Int32);
        let result = execute(array, &target)?;

        let expected = RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![2, 5]),
            &Int32Array::from(vec![10, 20]),
        )?;
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    #[rstest]
    #[case::offset_zero(        &[3, 5, 10],      0,     5, &[3, 5],    &[100, 200])]
    #[case::nonzero_offset(     &[3, 5, 10],      2,     3, &[1, 3],    &[100, 200])]
    #[case::clipped_last_run(   &[3, 5, 10],      0,     8, &[3, 5, 8], &[100, 200, 300])]
    #[case::run_boundary(       &[3, 5, 10],      3,     4, &[2, 4],    &[200, 300])]
    #[case::large_absolute_ends(&[40000, 40002], 39999, 3, &[1, 3],    &[100, 200])]
    #[case::empty_retained_runs(&[3, 5, 10],      11,    0, &[],        &[])]
    fn run_end_export_trims_runs(
        #[case] ends: &[u32],
        #[case] offset: usize,
        #[case] length: usize,
        #[case] expected_ends: &[i16],
        #[case] expected_values: &[i64],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            PrimitiveArray::from_iter(ends.iter().copied()).into_array(),
            PrimitiveArray::from_iter((0..ends.len()).map(|i| (i as i64 + 1) * 100)).into_array(),
            offset,
            length,
            &mut ctx,
        )?;

        let result = execute(
            array.into_array(),
            &ree_type(DataType::Int16, DataType::Int64),
        )?;
        assert_eq!(result.len(), length);
        let expected = RunArray::<Int16Type>::try_new(
            &Int16Array::from(expected_ends.to_vec()),
            &Int64Array::from(expected_values.to_vec()),
        )?;
        assert_eq!(result.as_ref(), &expected);

        Ok(())
    }

    #[rstest]
    #[case::empty(0, false)]
    #[case::sliced(3, true)]
    fn temporal_run_end_round_trip(#[case] len: usize, #[case] sliced: bool) -> VortexResult<()> {
        check_temporal_run_end_round_trip(len, sliced)
    }
}
