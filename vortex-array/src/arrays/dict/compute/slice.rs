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
            inherit_selection_stats(array.codes(), &sliced_code);
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

/// Copies the stats of `source` that hold for any order-preserving selection of its rows, such as a
/// slice or a filter.
pub(super) fn inherit_selection_stats(source: &ArrayRef, selected: &ArrayRef) {
    source.statistics().with_iter(|iter| {
        selected
            .statistics()
            .inherit(iter.filter(|(stat, value)| is_inheritable_selection_stat(*stat, value)));
    });
}

fn is_inheritable_selection_stat(stat: Stat, value: &Precision<ScalarValue>) -> bool {
    matches!(
        stat,
        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted
    ) && value
        .as_ref()
        .as_exact()
        .is_some_and(|value| matches!(value, ScalarValue::Bool(true)))
}

pub(super) fn slice_constant_code(
    array: ArrayView<'_, Dict>,
    code: &Scalar,
    len: usize,
) -> VortexResult<Option<ArrayRef>> {
    let code = code.as_primitive().as_::<usize>();
    if let Some(code) = code {
        let values = array.values().slice(code..code + 1)?;
        // Keep the codes' nullability so the result's dtype matches the original dict's dtype.
        let codes = ConstantArray::new(
            Scalar::primitive(0u8, array.codes().dtype().nullability()),
            len,
        );
        Ok(Some(
            DictArray::new(codes.into_array(), values).into_array(),
        ))
    } else {
        Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), len).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

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

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn slice_constant_valid_code() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::new(
            ConstantArray::new(1u8, 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected = PrimitiveArray::from_iter([20i32, 20, 20]).into_array();
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn slice_nullable_constant_valid_code() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::new(
            ConstantArray::new(Scalar::primitive(1u8, Nullable), 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected = PrimitiveArray::from_option_iter([Some(20i32), Some(20), Some(20)]);
        assert_arrays_eq!(sliced, expected.into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn slice_constant_null_code() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::new(
            ConstantArray::new(Scalar::null(DType::Primitive(PType::U8, Nullable)), 5).into_array(),
            buffer![10i32, 20, 30].into_array(),
        );
        let sliced = dict.slice(1..4)?;
        let expected =
            PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]).into_array();
        assert_arrays_eq!(sliced, expected, &mut ctx);
        Ok(())
    }
}
