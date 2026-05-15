// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Variant;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const RULES: ParentRuleSet<Variant> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Variant)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Variant)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Variant)),
]);
