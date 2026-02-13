// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeListVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::compute::CastReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<FixedSizeListVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(FixedSizeListVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FixedSizeListVTable)),
]);
