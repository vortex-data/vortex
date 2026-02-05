use crate::arrays::SliceReduceAdaptor;
use crate::arrays::VarBinViewVTable;
use crate::optimizer::rules::ParentRuleSet;

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
pub(crate) const PARENT_RULES: ParentRuleSet<VarBinViewVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(VarBinViewVTable))]);
