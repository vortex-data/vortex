// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::SliceReduceAdaptor;
use crate::arrays::VarBinVTable;
use crate::compute::CastReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<VarBinVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(VarBinVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(VarBinVTable)),
]);
