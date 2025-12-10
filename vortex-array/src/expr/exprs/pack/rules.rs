// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ExactScalarFn;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ScalarFnArrayExt;
use crate::arrays::ScalarFnArrayView;
use crate::expr::Pack;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::Exact;
use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::IntoArray;

/// Pack expression should always push-down filter, regardless of cost.
#[derive(Debug)]
pub(crate) struct PackFilterPushdown;

impl ArrayParentReduceRule<ExactScalarFn<Pack>, Exact<FilterVTable>> for PackFilterPushdown {
    fn child(&self) -> ExactScalarFn<Pack> {
        ExactScalarFn::from(&Pack)
    }

    fn parent(&self) -> Exact<FilterVTable> {
        Exact::from(&FilterVTable)
    }

    fn reduce_parent(
        &self,
        child: ScalarFnArrayView<Pack>,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_children: Vec<_> = child
            .children()
            .into_iter()
            .map(|child| FilterArray::new(child, parent.mask().clone()).into_array())
            .collect();
        Ok(Some(
            Pack.try_new_array(parent.len(), child.options.clone(), new_children)?
                .into_array(),
        ))
    }
}
