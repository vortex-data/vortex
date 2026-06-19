// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::slice::SliceReduce;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl SliceReduce for Dict {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        if let Some(code) = array.codes().as_opt::<Constant>() {
            return slice_constant_code(array, code.scalar(), range.len());
        }

        let sliced_code = if let Some(codes) = array.codes().as_typed::<Primitive>() {
            let sliced_code = <Primitive as SliceReduce>::slice(codes, range)?
                .vortex_expect("Primitive SliceReduce should always return Some");
            // Because we specialize the primitive branch here, we have to make sure to handle the stat inheritance
            inherit_slice_stats(array.codes(), &sliced_code);
            sliced_code
        } else {
            array.codes().slice(range)?
        };

        // TODO(joe): if the range is size 1 replace with a constant array
        if let Some(code) = sliced_code.as_opt::<Constant>() {
            return slice_constant_code(array, code.scalar(), sliced_code.len());
        }
        // SAFETY: slicing the codes preserves invariants.
        let array =
            unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()).into_array() };

        Ok(Some(array))
    }
}

fn inherit_slice_stats(source: &ArrayRef, sliced: &ArrayRef) {
    source.statistics().with_iter(|iter| {
        sliced
            .statistics()
            .inherit(iter.filter(|(stat, value)| is_inheritable_true_slice_stat(*stat, value)));
    });
}

fn is_inheritable_true_slice_stat(stat: Stat, value: &Precision<ScalarValue>) -> bool {
    matches!(
        stat,
        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted
    ) && value
        .as_ref()
        .as_exact()
        .is_some_and(|value| matches!(value, ScalarValue::Bool(true)))
}

fn slice_constant_code(
    array: ArrayView<'_, Dict>,
    code: &Scalar,
    len: usize,
) -> VortexResult<Option<ArrayRef>> {
    let code = code.as_primitive().as_::<usize>();
    if let Some(code) = code {
        let values = array.values().slice(code..code + 1)?;
        Ok(Some(
            DictArray::new(ConstantArray::new(0u8, len).into_array(), values).into_array(),
        ))
    } else {
        Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), len).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::compute::slice::ConstantArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

    #[test]
    fn slice_constant_valid_code() -> VortexResult<()> {
        let assertion_session = crate::array_session();
        let mut assertion_ctx = assertion_session.create_execution_ctx();
        let dict = DictArray::new(
            ConstantArray::new(1u8, 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected = PrimitiveArray::from_iter([20i32, 20, 20]).into_array();
        assert_arrays_eq!(sliced, expected, &mut assertion_ctx);
        Ok(())
    }

    #[test]
    fn slice_constant_null_code() -> VortexResult<()> {
        let assertion_session = crate::array_session();
        let mut assertion_ctx = assertion_session.create_execution_ctx();
        let dict = DictArray::new(
            ConstantArray::new(Scalar::null(DType::Primitive(PType::U8, Nullable)), 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected =
            PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]).into_array();
        assert_arrays_eq!(sliced, expected, &mut assertion_ctx);
        Ok(())
    }
}
