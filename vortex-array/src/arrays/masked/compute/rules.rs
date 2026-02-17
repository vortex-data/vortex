// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FilterReduceAdaptor;
use crate::arrays::MaskedVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::compute::MaskReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<MaskedVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(MaskedVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(MaskedVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(MaskedVTable)),
]);
