// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Masked;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Masked> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Masked)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Masked)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Masked)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Masked)),
]);
