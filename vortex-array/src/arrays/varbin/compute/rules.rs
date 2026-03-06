// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::slice::SliceReduceAdaptor;
use crate::arrays::varbin::VarBinVTable;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<VarBinVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(VarBinVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(VarBinVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(VarBinVTable)),
]);
