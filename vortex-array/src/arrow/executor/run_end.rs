// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrow::ArrowArrayExecutor;

/// Fallback conversion to Arrow run-end encoded. The encoding's `to_arrow_array` is tried first
/// by the executor; this handles remaining cases via `arrow_cast`.
pub(crate) fn to_arrow_run_end(
    array: ArrayRef,
    ends_type: &DataType,
    values_type: &Field,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let flat = array.execute_arrow(Some(values_type.data_type()), ctx)?;
    let ree_type = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", ends_type.clone(), false)),
        Arc::new(values_type.clone()),
    );
    arrow_cast::cast(&flat, &ree_type).map_err(VortexError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Int32Array;
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
            &arrow_array::Int16Array::from(vec![5i16]),
            &Int32Array::from(vec![42]),
        ).unwrap()) as arrow_array::ArrayRef,
    )]
    #[case::f64_with_i64_ends(
        ConstantArray::new(Scalar::from(1.5f64), 7).into_array(),
        ree_type(DataType::Int64, DataType::Float64),
        Arc::new(RunArray::<Int64Type>::try_new(
            &arrow_array::Int64Array::from(vec![7i64]),
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
