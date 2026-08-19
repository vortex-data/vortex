// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeReduceAdaptor;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use super::DenseUnion;

pub(crate) const PARENT_RULES: ParentRuleSet<DenseUnion> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(DenseUnion)),
    ParentRuleSet::lift(&MaskReduceAdaptor(DenseUnion)),
    ParentRuleSet::lift(&SliceReduceAdaptor(DenseUnion)),
    ParentRuleSet::lift(&TakeReduceAdaptor(DenseUnion)),
]);
