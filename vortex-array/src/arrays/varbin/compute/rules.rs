// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::SliceReduceAdaptor;
use crate::arrays::VarBinVTable;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::CastReduceAdaptor;
use crate::scalar_fn::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<VarBinVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(VarBinVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(VarBinVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(VarBinVTable)),
]);
