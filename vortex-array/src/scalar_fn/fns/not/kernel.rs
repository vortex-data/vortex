// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::not::Not as NotExpr;

/// Invert a boolean array without reading buffers.
///
/// This trait is for invert implementations that can operate purely on array metadata
/// and structure without needing to read or execute on the underlying buffers.
/// Implementations should return `None` if the operation requires buffer access.
pub trait NotReduce: VTable {
    fn invert(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`NotReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct NotReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for NotReduceAdaptor<V>
where
    V: NotReduce,
{
    type Parent = ExactScalarFn<NotExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: ScalarFnArrayView<'_, NotExpr>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        <V as NotReduce>::invert(array)
    }
}
