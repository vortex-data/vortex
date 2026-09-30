// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::Narrow;
use super::NarrowArray;
use super::NarrowArraySlotsExt;
use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::binary::execute_compare;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::scalar_fn::fns::operators::CompareOperator;

pub(super) const PARENT_RULES: ParentRuleSet<Narrow> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&CastReduceAdaptor(Narrow)),
]);

pub(crate) fn initialize(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        Binary.id(),
        Narrow,
        CompareExecuteAdaptor(Narrow),
    );
}

fn rewrap(array: ArrayView<'_, Narrow>, values: ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let dtype = array.dtype().with_nullability(values.dtype().nullability());
    Ok(Some(NarrowArray::try_new(values, dtype)?.into_array()))
}

impl SliceReduce for Narrow {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().slice(range)?)
    }
}

impl FilterReduce for Narrow {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().filter(mask.clone())?)
    }
}

impl TakeReduce for Narrow {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().take(indices.clone())?)
    }
}

impl MaskReduce for Narrow {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().clone().mask(mask.clone())?)
    }
}

impl CastReduce for Narrow {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if dtype == array.values().dtype() {
            return Ok(Some(array.values().clone()));
        }
        if super::validate_dtypes(array.values().dtype(), dtype).is_ok() {
            return Ok(Some(
                NarrowArray::try_new(array.values().clone(), dtype.clone())?.into_array(),
            ));
        }
        Ok(None)
    }
}

impl CompareKernel for Narrow {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(rhs) = rhs.as_opt::<Narrow>()
            && lhs
                .values()
                .dtype()
                .eq_ignore_nullability(rhs.values().dtype())
        {
            return compare_values(lhs.values(), rhs.values(), operator, ctx).map(Some);
        }
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let storage_dtype = lhs.values().dtype().with_nullability(nullability);
        if let Ok(value) = constant.cast(&storage_dtype) {
            return compare_values(
                lhs.values(),
                &ConstantArray::new(value, lhs.len()).into_array(),
                operator,
                ctx,
            )
            .map(Some);
        }

        // A non-null integer that cannot fit is outside the entire child's type range.
        let below = lhs.dtype().is_signed_int() && i64::try_from(&constant)? < 0;
        let result = match operator {
            CompareOperator::Eq => false,
            CompareOperator::NotEq => true,
            CompareOperator::Lt | CompareOperator::Lte => !below,
            CompareOperator::Gt | CompareOperator::Gte => below,
        };
        let validity = lhs
            .values()
            .validity()?
            .cast_nullability(nullability, lhs.len(), ctx)?;
        Ok(Some(
            BoolArray::try_new(
                BitBuffer::full_in(result, lhs.len(), ctx.allocator().clone()),
                validity,
            )?
            .into_array(),
        ))
    }
}

fn compare_values(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Primitive children need no further encoding dispatch or intermediate expression.
    if lhs.is::<Primitive>() && (rhs.is::<Primitive>() || rhs.is::<Constant>()) {
        execute_compare(lhs, rhs, operator, ctx)
    } else {
        lhs.binary(rhs.clone(), operator.into())
    }
}
