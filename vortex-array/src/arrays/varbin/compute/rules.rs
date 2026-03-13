// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::VarBin;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<VarBin> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(VarBin)),
    ParentRuleSet::lift(&MaskReduceAdaptor(VarBin)),
    ParentRuleSet::lift(&SliceReduceAdaptor(VarBin)),
]);
