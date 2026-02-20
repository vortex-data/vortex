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

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayVisitor;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
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
    let array = match array.try_into::<ConstantVTable>() {
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
    if offset == 0 {
        return Ok(
            Arc::new(RunArray::<R>::try_new(ends.as_primitive::<R>(), values)?) as ArrowArrayRef,
        );
    }

    let offset_native = R::Native::from_usize(offset)
        .ok_or_else(|| vortex_err!("Offset {offset} exceeds run-end index capacity"))?;
    let length_native = R::Native::from_usize(length)
        .ok_or_else(|| vortex_err!("Length {length} exceeds run-end index capacity"))?;

    let adjusted = ends
        .as_primitive::<R>()
        .unary(|end| (end - offset_native).min(length_native));

    Ok(Arc::new(RunArray::<R>::try_new(&adjusted, values)?) as ArrowArrayRef)
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
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrow::ArrowArrayExecutor;
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
}
