// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::NullVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<NullVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(NullVTable))]);
