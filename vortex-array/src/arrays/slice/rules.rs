// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::SliceVTable;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<SliceVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(SliceVTable))]);
