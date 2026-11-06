// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
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
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::not::NotReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(
        Between,
        Constant,
        const { &BetweenReduceAdaptor(Constant) },
    );
    kernels.register_reduce_parent_rule(Cast, Constant, const { &CastReduceAdaptor(Constant) });
    kernels.register_reduce_parent_rule(Filter, Constant, const { &ConstantFilterRule });
    kernels.register_reduce_parent_rule(
        FillNull,
        Constant,
        const { &FillNullReduceAdaptor(Constant) },
    );
    kernels.register_reduce_parent_rule(Filter, Constant, const { &FilterReduceAdaptor(Constant) });
    kernels.register_reduce_parent_rule(Not, Constant, const { &NotReduceAdaptor(Constant) });
    kernels.register_reduce_parent_rule(Slice, Constant, const { &SliceReduceAdaptor(Constant) });
    kernels.register_reduce_parent_rule(Dict, Constant, const { &TakeReduceAdaptor(Constant) });
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
