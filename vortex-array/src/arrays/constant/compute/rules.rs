// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::not::NotReduceAdaptor;

/// Registers the parent-reduce rules for [`Constant`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(
        Between.id(),
        Constant.id(),
        BetweenReduceAdaptor(Constant),
    );
    kernels.register_reduce_parent_rule(Cast.id(), Constant.id(), CastReduceAdaptor(Constant));
    kernels.register_reduce_parent_rule(Filter.id(), Constant.id(), ConstantFilterRule);
    kernels.register_reduce_parent_rule(
        FillNull.id(),
        Constant.id(),
        FillNullReduceAdaptor(Constant),
    );
    kernels.register_reduce_parent_rule(Filter.id(), Constant.id(), FilterReduceAdaptor(Constant));
    kernels.register_reduce_parent_rule(Not.id(), Constant.id(), NotReduceAdaptor(Constant));
    kernels.register_reduce_parent_rule(Slice.id(), Constant.id(), SliceReduceAdaptor(Constant));
    kernels.register_reduce_parent_rule(Dict.id(), Constant.id(), TakeReduceAdaptor(Constant));
}

#[derive(Debug)]
struct ConstantFilterRule;

impl ArrayParentReduceRule<Constant> for ConstantFilterRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, Constant>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(child.scalar.clone(), parent.len()).into_array(),
        ))
    }
}
