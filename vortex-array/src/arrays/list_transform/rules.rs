// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::ListTransform;
use crate::arrays::ListTransformArray;
use crate::arrays::ListTransformArrayExt;
use crate::arrays::Slice;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<ListTransform> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ListTransformSlicePushDown),
    ParentRuleSet::lift(&ListTransformFilterPushDown),
    ParentRuleSet::lift(&TakeReduceAdaptor(ListTransform)),
]);

#[derive(Debug)]
struct ListTransformSlicePushDown;

impl ArrayParentReduceRule<ListTransform> for ListTransformSlicePushDown {
    type Parent = Slice;

    fn reduce_parent(
        &self,
        transform: ArrayView<'_, ListTransform>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range().clone();
        Ok(Some(
            ListTransformArray::try_new_from_parts(
                transform.list().slice(range.clone())?,
                transform.body().clone(),
                transform
                    .captures()
                    .map(|capture| capture.slice(range.clone()))
                    .collect::<VortexResult<Vec<_>>>()?,
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ListTransformFilterPushDown;

impl ArrayParentReduceRule<ListTransform> for ListTransformFilterPushDown {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        transform: ArrayView<'_, ListTransform>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask().clone();
        Ok(Some(
            ListTransformArray::try_new_from_parts(
                transform.list().filter(mask.clone())?,
                transform.body().clone(),
                transform
                    .captures()
                    .map(|capture| capture.filter(mask.clone()))
                    .collect::<VortexResult<Vec<_>>>()?,
            )?
            .into_array(),
        ))
    }
}

impl TakeReduce for ListTransform {
    fn take(transform: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ListTransformArray::try_new_from_parts(
                transform.list().take(indices.clone())?,
                transform.body().clone(),
                transform
                    .captures()
                    .map(|capture| capture.take(indices.clone()))
                    .collect::<VortexResult<Vec<_>>>()?,
            )?
            .into_array(),
        ))
    }
}
