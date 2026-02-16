// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Binary;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ExactScalarFn;
use crate::arrays::ScalarFnArrayView;
use crate::arrays::ScalarFnVTable;
use crate::compute::Operator;
use crate::kernel::ExecuteParentKernel;
use crate::vtable::VTable;

/// Trait for encoding-specific comparison kernels that operate in encoded space.
///
/// Implementations can compare an encoded array against another array (typically a constant)
/// without first decompressing. The adaptor normalizes operand order so `array` is always
/// the left-hand side, swapping the operator when necessary.
pub trait CompareKernel: VTable {
    fn compare(
        array: &Self::Array,
        other: &dyn crate::Array,
        operator: Operator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that bridges [`CompareKernel`] implementations to [`ExecuteParentKernel`].
///
/// When a `ScalarFnArray(Binary, cmp_op)` wraps a child that implements `CompareKernel`,
/// this adaptor extracts the comparison operator and other operand, normalizes operand order
/// (swapping the operator if the encoded array is on the RHS), and delegates to the kernel.
#[derive(Default, Debug)]
pub struct CompareExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for CompareExecuteAdaptor<V>
where
    V: CompareKernel,
{
    type Parent = ExactScalarFn<Binary>;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle comparison operators
        let Some(cmp_op) = parent.options.maybe_cmp_operator() else {
            return Ok(None);
        };

        // Get the ScalarFnArray to access children
        let Some(scalar_fn_array) = parent.as_opt::<ScalarFnVTable>() else {
            return Ok(None);
        };
        let children = scalar_fn_array.children();

        // Normalize so `array` is always LHS, swapping the operator if needed
        let (cmp_op, other) = match child_idx {
            0 => (cmp_op, &children[1]),
            1 => (cmp_op.swap(), &children[0]),
            _ => return Ok(None),
        };

        V::compare(array, other.as_ref(), cmp_op, ctx)
    }
}
