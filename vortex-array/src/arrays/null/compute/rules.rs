// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FilterReduceAdaptor;
use crate::arrays::NullVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::arrays::TakeReduceAdaptor;
use crate::expr::CastReduceAdaptor;
use crate::expr::MaskReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<NullVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(NullVTable)),
    ParentRuleSet::lift(&CastReduceAdaptor(NullVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(NullVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(NullVTable)),
    ParentRuleSet::lift(&TakeReduceAdaptor(NullVTable)),
]);
