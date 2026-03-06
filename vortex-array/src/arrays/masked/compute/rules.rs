// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::masked::MaskedVTable;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<MaskedVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(MaskedVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(MaskedVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(MaskedVTable)),
    ParentRuleSet::lift(&TakeReduceAdaptor(MaskedVTable)),
]);
