// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RunArray;
use arrow_array::cast::AsArray;
use arrow_array::new_null_array;
use arrow_array::types::*;
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use arrow_schema::Field;
use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayVisitor;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrow::ArrowArrayExecutor;

/// The encoding ID used by `vortex-runend`. We match on this string to avoid a crate dependency.
const VORTEX_RUNEND_ID: &str = "vortex.runend";

/// Mirror of `RunEndMetadata` from the `vortex-runend` crate. Same prost field tags so we can
/// decode the serialized metadata without depending on that crate.
#[derive(Clone, prost::Message)]
struct RunEndMetadata {
    #[prost(int32, tag = "1")]
    pub ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
}

pub(super) fn to_arrow_run_end(
    array: ArrayRef,
    ends_type: &DataType,
    values_type: &Field,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = match array.try_into::<Constant>() {
        Ok(constant) => {
            return constant_to_run_end(constant, ends_type, values_type, ctx);
        }
        Err(array) => array,
    };

    // Execute to unwrap any wrapper VTables (Slice, Filter, etc.) which may
    // reveal a RunEndArray.
    let array = array.execute::<ArrayRef>(ctx)?;
    if array.encoding_id().as_ref() == VORTEX_RUNEND_ID {
        // NOTE(ngates): while this module still lives in vortex-array, we cannot depend on the
        //  vortex-runend crate. Therefore, we match on the encoding ID string and extract the children
        //  and metadata directly.
        return run_end_to_arrow(array, ends_type, values_type, ctx);
    }

    // Fallback: canonicalize to flat Arrow, then cast to REE.
    let flat = array.execute_arrow(Some(values_type.data_type()), ctx)?;
    let ree_type = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", ends_type.clone(), false)),
        Arc::new(values_type.clone()),
    );
    arrow_cast::cast(&flat, &ree_type).map_err(VortexError::from)
}

fn run_end_to_arrow(
    array: ArrayRef,
    ends_type: &DataType,
    values_type: &Field,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let length = array.len();
    let metadata_bytes = array
        .metadata()?
        .ok_or_else(|| vortex_err!("RunEndArray missing metadata"))?;
    let metadata = RunEndMetadata::decode(&*metadata_bytes)
        .map_err(|e| vortex_err!("Failed to decode RunEndMetadata: {e}"))?;
    let offset = usize::try_from(metadata.offset)
        .map_err(|_| vortex_err!("RunEndArray offset {} overflows usize", metadata.offset))?;

    let children = array.children();
    vortex_ensure!(
        children.len() == 2,
        "Expected RunEndArray to have 2 children, got {}",
        children.len()
    );

    let arrow_ends = children[0].clone().execute_arrow(Some(ends_type), ctx)?;
    let arrow_values = children[1]
        .clone()
        .execute_arrow(Some(values_type.data_type()), ctx)?;

    match ends_type {
        DataType::Int16 => build_run_array::<Int16Type>(&arrow_ends, &arrow_values, offset, length),
        DataType::Int32 => build_run_array::<Int32Type>(&arrow_ends, &arrow_values, offset, length),
        DataType::Int64 => build_run_array::<Int64Type>(&arrow_ends, &arrow_values, offset, length),
        _ => vortex_bail!("Unsupported run-end index type: {:?}", ends_type),
    }
}

fn build_run_array<R: RunEndIndexType>(
    ends: &ArrowArrayRef,
    values: &ArrowArrayRef,
    offset: usize,
    length: usize,
) -> VortexResult<ArrowArrayRef>
where
    R::Native: std::ops::Sub<Output = R::Native> + Ord,
{
    let offset_native = R::Native::from_usize(offset)
        .ok_or_else(|| vortex_err!("Offset {offset} exceeds run-end index capacity"))?;
    let length_native = R::Native::from_usize(length)
        .ok_or_else(|| vortex_err!("Length {length} exceeds run-end index capacity"))?;

    let ends_prim = ends.as_primitive::<R>();
    if offset == 0 && ends_prim.values().last() == Some(&length_native) {
        // Fast path: no trimming or adjustment needed.
        return Ok(Arc::new(RunArray::<R>::try_new(ends_prim, values)?) as ArrowArrayRef);
    }

    // Trim to only include runs covering the [offset, offset+length) range.
    // Runs beyond this would produce duplicate adjusted ends, violating
    // Arrow's strict-ordering requirement for RunArray.
    // Run ends are strictly increasing, so we can binary search.
    let num_runs = (ends_prim
        .values()
        .partition_point(|&e| e - offset_native < length_native)
        + 1)
    .min(ends_prim.len());

    let trimmed_ends = ends.slice(0, num_runs);
    let trimmed_values = values.slice(0, num_runs);

    let adjusted = trimmed_ends
        .as_primitive::<R>()
        .unary(|end| (end - offset_native).min(length_native));

    Ok(Arc::new(RunArray::<R>::try_new(&adjusted, &trimmed_values)?) as ArrowArrayRef)
}

/// Convert a constant array to a run-end encoded array with a single run.
fn constant_to_run_end(
    array: ConstantArray,
    ends_type: &DataType,
    values_type: &Field,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();
    let scalar = array.scalar();

    if scalar.is_null() || len == 0 {
        let ree_type = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ends_type.clone(), false)),
            Arc::new(values_type.clone()),
        );
        return Ok(new_null_array(&ree_type, len));
    }

    let values = ConstantArray::new(scalar.clone(), 1)
        .into_array()
        .execute_arrow(Some(values_type.data_type()), ctx)?;

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
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::executor::run_end::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::executor::VortexSessionExecute;
    use crate::scalar::Scalar;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn ree_type(ends: DataType, values_dtype: DataType) -> DataType {
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ends, false)),
            Arc::new(Field::new("values", values_dtype, true)),
        )
    }

    fn execute(array: crate::ArrayRef, dt: &DataType) -> VortexResult<arrow_array::ArrayRef> {
        array.execute_arrow(Some(dt), &mut SESSION.create_execution_ctx())
    }

    #[rstest]
    #[case::i32_with_i16_ends(
        ConstantArray::new(Scalar::from(42i32), 5).into_array(),
        ree_type(DataType::Int16, DataType::Int32),
        Arc::new(RunArray::<Int16Type>::try_new(
            &Int16Array::from(vec![5i16]),
            &Int32Array::from(vec![42]),
        ).unwrap()) as arrow_array::ArrayRef,
    )]
    #[case::f64_with_i64_ends(
        ConstantArray::new(Scalar::from(1.5f64), 7).into_array(),
        ree_type(DataType::Int64, DataType::Float64),
        Arc::new(RunArray::<Int64Type>::try_new(
            &Int64Array::from(vec![7i64]),
            &arrow_array::Float64Array::from(vec![1.5]),
        ).unwrap()) as arrow_array::ArrayRef,
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
        #[case] input: crate::ArrayRef,
        #[case] target_type: DataType,
        #[case] expected: arrow_array::ArrayRef,
    ) -> VortexResult<()> {
        let result = execute(input, &target_type)?;
        assert_eq!(result.as_ref(), expected.as_ref());
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

    /// Regression: build_run_array must trim excess trailing runs and
    /// respect the `length` parameter. This happens when a vortex
    /// RunEndArray is sliced to fewer rows than the physical run_ends cover.
    #[rstest]
    #[case::offset_zero(0, 5, &[3, 5], &[100, 200])]
    #[case::nonzero_offset(2, 3, &[1, 3], &[100, 200])]
    #[case::all_runs_needed_but_last_exceeds(0, 8, &[3, 5, 8], &[100, 200, 300])]
    fn build_run_array_trims_excess_runs(
        #[case] offset: usize,
        #[case] length: usize,
        #[case] expected_ends: &[i32],
        #[case] expected_values: &[i64],
    ) -> VortexResult<()> {
        // 3 runs covering 10 rows: [0..3), [3..5), [5..10)
        let ends: arrow_array::ArrayRef = Arc::new(Int32Array::from(vec![3i32, 5, 10]));
        let values: arrow_array::ArrayRef = Arc::new(Int64Array::from(vec![100i64, 200, 300]));

        let result = super::build_run_array::<Int32Type>(&ends, &values, offset, length)?;
        assert_eq!(result.len(), length);

        let ree = result
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .unwrap();
        assert_eq!(ree.run_ends().values(), expected_ends);
        let values = ree.values().as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(values.values(), expected_values);
        Ok(())
    }
}
